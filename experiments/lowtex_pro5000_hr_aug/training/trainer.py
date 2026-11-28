import math
import time
from pathlib import Path
import numpy as np
import torch
from .optimizer import optimizer,freeze_phase,seed_all,rng_state,restore_rng
from .runtime import MemoryPressure
from ..data.dataset_adapter import TrainPool,loader,manifest
from ..evaluation.eval_hr import double_dev
from ..models.slim_wrapper import CCSMatcher
from ..utils.run_io import append,atomic_path,digest,now,write,read


def save_checkpoint(path,payload):
    with atomic_path(path) as tmp:
        torch.save(payload,tmp)


def score(dev):
    md,tp = dev["megadepth"],dev["texturepoor"]
    if md["status"]!="PASS" or tp["status"]!="PASS":
        return None
    return md["scene_equal"].get("lt_recall")


def update_summary(rows):
    keys = set().union(*(r.keys() for r in rows))
    s = {k:sum(r.get(k,0) for r in rows) for k in keys}
    for k in s:
        if k.endswith("_loss"):
            s[k] /= len(rows)
    for name,num,den in (("route_recall","route_hits","route_den"),("halo_coverage","predicted_halo_hits","route_den"),
                         ("route_gt_add_fraction","injected_blocks","gt_blocks"),
                         ("topm_gt_add_fraction","injected_hypotheses","retained_gt"),
                         ("topm_recall","topm_predicted_hits","route_den")):
        if num in s and s.get(den):
            s[name] = s[num]/s[den]
    return s


def train_stage(args,cfg,job,guard,probe=False):
    seed_all(cfg["seed_model"])
    side = job["side"]
    directory = Path(job["directory"])
    directory.mkdir(parents=True,exist_ok=True)
    report = Path(job["report"])
    calib = report/"calibration"
    stage_start,total,end = job["start"],job["total"],job["end"]
    micro = job["micro"]
    effective = cfg["effective_batch_size"]
    if effective%micro:
        raise ValueError("effective batch is not divisible by the microbatch")
    precision = job["precision"]
    configured_hash=digest(cfg)
    cfg=dict(cfg)
    for key in ("correlation_chunk","graph_chunk_size"):
        if key in job:
            cfg[key]=job[key]
    model = CCSMatcher(cfg,args.weights).cuda()
    model.checkpointing = job["checkpointing"]
    opt,sched = optimizer(model,total)
    scaler = torch.amp.GradScaler("cuda",enabled=precision=="float16")
    amp = getattr(torch,precision)
    pool = manifest("train_pool")
    seen = np.zeros(pool["unique_unordered"],dtype=np.bool_)
    step,cursor,backbone_steps,invalid = 0,0,0,0
    best,best_step = None,None
    quick_best,quick_step = None,None
    last = directory/"checkpoint_last.ckpt"
    overflows = 0
    identity = job["identity"]
    warm = True
    recent_dev = {}
    if not probe:
        source = last if last.exists() else (Path(job["previous"]) if job.get("previous") else None)
        if source is not None:
            saved = torch.load(source,map_location="cpu",weights_only=False)
            if saved["identity"]!=identity or saved["config_hash"]!=configured_hash:
                raise ValueError("incompatible checkpoint source, settings, data or budget")
            if source==last and any(saved["job"].get(k)!=job.get(k) for k in
                                     ("start","end","total","formal","model","micro","precision","checkpointing","correlation_chunk","graph_chunk_size")):
                raise ValueError("execution profile changed during stage resume")
            if source!=last and (not saved["job"].get("formal") or saved["job"]["total"]!=total):
                raise ValueError("previous checkpoint is not the fixed-budget progressive training stage")
            if source==last and saved["side"]!=side:
                raise ValueError("checkpoint resolution mismatch")
            model.load_state_dict(saved["model"])
            opt.load_state_dict(saved["optimizer"])
            sched.load_state_dict(saved["scheduler"])
            if saved["precision"]==precision:
                scaler.load_state_dict(saved["scaler"])
            step,cursor,backbone_steps,invalid = (saved[k] for k in ("step","cursor","backbone_steps","invalid"))
            seen = np.unpackbits(saved["seen"],count=len(seen)).astype(bool)
            restore_rng(saved["rng"])
            if source==last:
                best,best_step = saved["best"],saved["best_step"]
                quick_best,quick_step = saved.get("quick_best"),saved.get("quick_step")
                recent_dev = saved.get("recent_dev",{})
            elif step!=stage_start:
                raise ValueError("resolution transition must use fixed-step checkpoint_last")
            del saved
        elif stage_start:
            raise FileNotFoundError("previous fixed-step stage checkpoint")
    warm = freeze_phase(model,step,total)
    def checkpoint_now():
        if probe:
            return
        save_checkpoint(last,dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),
                     scaler=scaler.state_dict(),step=step,cursor=cursor,seen=np.packbits(seen),invalid=invalid,
                     backbone_steps=backbone_steps,warm=warm,best=best,best_step=best_step,quick_best=quick_best,quick_step=quick_step,recent_dev=recent_dev,
                     side=side,rng=rng_state(),precision=precision,identity=identity,config=cfg,
                     config_hash=configured_hash,job=job,coverage=coverage()))
    def coverage():
        rows = min(pool["legal_rows"],cursor)
        return dict(legal_rows=pool["legal_rows"],unique_legal_pairs=pool["unique_unordered"],
                    visited_rows=rows,unique_pairs=unique,visits=cursor,repeat_visits=cursor-unique,
                    row_coverage=rows/pool["legal_rows"],physical_coverage=unique/pool["unique_unordered"],invalid=invalid)
    unique = int(seen.sum())
    def validate(label,fast=False):
        nonlocal best,best_step,quick_best,quick_step,recent_dev
        v = double_dev(model,args,side,directory/label,calib,cfg["eval_precision"],guard,quick=fast,probe=probe)
        recent_dev = {domain:{k:z for k,z in res.get("scene_equal",{}).items()
                               if k in ("auc@5","auc@10","auc@20","lt_recall","other_recall","lt_epe","lt_route_recall","lt_topm_recall")}
                      for domain,res in v.items()}
        if not probe:
            write(report/"live_status.json",dict(model=cfg["model"],resolution=side,step=step,updated=now(),development=recent_dev))
        if not probe and label.startswith("quick_"):
            value=score(v)
            if value is not None and (quick_best is None or value>quick_best):
                quick_best,quick_step=value,step
                save_checkpoint(directory/"nominations"/f"step_{step}.ckpt",dict(model=model.state_dict(),identity=identity,
                                side=side,step=step,score=value,config=cfg))
        return v
    if not probe and step==stage_start and not (directory/"before_done.json").exists():
        initial = validate("before")
        write(directory/"before_done.json",{k:v["status"] for k,v in initial.items()})
        checkpoint_now()
    cache = Path(job["cache"])
    def remaining_batches():
        dataset = TrainPool(args.data_root,cache,side,cfg,start=cursor,stop=cursor+(end-step)*effective)
        return iter(loader(dataset,micro,cfg))
    iterator = remaining_batches()
    times = []
    while step<end:
        try:
            guard.check()
        except MemoryPressure:
            checkpoint_now()
            raise
        before_rng = rng_state()
        warm = freeze_phase(model,step,total)
        # Probe explicitly exercises one warm then two fully unfrozen optimizer updates.
        if probe:
            warm = freeze_phase(model,0 if step==0 else total,total)
        inject = max(0.,1-step/max(1,total*cfg["injection_end_ratio"]))
        start = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        rows,ids = [],[]
        loss_sum = 0.
        try:
            for _ in range(effective//micro):
                host = next(iterator)
                ids.extend(host["pair_id"].tolist())
                actual_shapes={k:list(v.shape) for k,v in host.items() if isinstance(v,torch.Tensor)}
                if not probe and len(ids)==micro:
                    write(report/"current_batch.json",dict(model=cfg["model"],side=side,step=step,shapes=actual_shapes,
                          pair_ids=host["pair_id"].tolist(),micro=micro,precision=precision))
                batch = {k:v.to("cuda",non_blocking=cfg["non_blocking"]) for k,v in host.items()}
                with torch.autocast("cuda",dtype=amp):
                    loss,stats = model.training_loss(batch,inject)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite loss at {step}, pair IDs {ids}, stats {stats}")
                scaler.scale(loss*(micro/effective)).backward()
                rows.extend(stats)
                loss_sum += float(loss.detach())*micro/effective
                del batch,loss
                guard.check()
        except MemoryPressure:
            opt.zero_grad(set_to_none=True)
            restore_rng(before_rng)
            checkpoint_now()
            raise
        scaler.unscale_(opt)
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["gradient_clip_norm"],error_if_nonfinite=False)
        if not torch.isfinite(grad):
            overflows += 1
            append(directory/"nonfinite.jsonl",dict(step=step,pair_ids=ids,precision=precision,scale=scaler.get_scale(),event="gradient overflow"))
            if not scaler.is_enabled() or overflows>12:
                raise FloatingPointError(f"nonfinite gradient, step {step}, see nonfinite.jsonl")
            opt.zero_grad(set_to_none=True)
            scaler.update(new_scale=scaler.get_scale()*.5)
            restore_rng(before_rng)
            del iterator
            iterator=remaining_batches()
            continue
        overflows=0
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
        cursor += effective
        backbone_steps += int(not warm)
        invalid += sum(r["valid_geometry"]==0 for r in rows)
        newly = np.unique(ids)
        unique += int((~seen[newly]).sum())
        seen[newly] = True
        torch.cuda.synchronize()
        duration = time.perf_counter()-start
        times.append(duration)
        memory = guard.snapshot()
        live = dict(model=cfg["model"],resolution=side,step=step,max_steps=end,percent=100*(step-stage_start)/(end-stage_start),
                    updated=now(),loss=loss_sum,grad_norm=float(grad),lr=[g["lr"] for g in opt.param_groups],
                    warm=warm,backbone_steps=backbone_steps,injection_probability=inject,precision=precision,
                    shapes=actual_shapes,micro_batch=micro,accumulation=effective//micro,seconds_per_step=duration,pairs_per_s=effective/duration,
                    eta_seconds=(end-step)*np.mean(times[-50:]),metrics=update_summary(rows),memory=memory,
                    development=recent_dev,coverage=coverage(),finite=True)
        print(f"[{cfg['model']} | R{side}] {step}/{end} {live['percent']:.1f}% loss={loss_sum:.5f} "
              f"LR={live['lr']} reserved={memory['reserved_gib']:.2f} GiB {effective/duration:.2f} pairs/s "
              f"{duration:.2f}s/update ETA={live['eta_seconds']:.0f}s",live["metrics"],recent_dev,flush=True)
        if not probe:
            write(report/"live_status.json",live)
            append(directory/"training.jsonl",live)
            quick_due = step%cfg["validation_interval"]==0 and step<end
            if quick_due:
                validate(f"quick_{step}",True)
            if quick_due or step%cfg["checkpoint_interval"]==0 or step==end:
                checkpoint_now()
        try:
            guard.check()
        except MemoryPressure:
            checkpoint_now()
            raise
    if not probe:
        if args.mode=="full" and job.get("formal") and step==total and min(coverage()["row_coverage"],coverage()["physical_coverage"])<1:
            raise RuntimeError(f"incomplete full coverage: {coverage()}")
        after = validate("after")
        last_score = score(after)
        best,best_step = last_score,step
        # Write the exact transition state before loading any nominated development candidate.
        checkpoint_now()
        selected_dev = after
        nomination = directory/"nominations"/f"step_{quick_step}.ckpt"
        if quick_step is not None and nomination.is_file():
            quick = torch.load(nomination,map_location="cpu",weights_only=False)
            if quick["step"]!=quick_step or quick["identity"]!=identity:
                raise ValueError("quick nomination identity/step mismatch")
            model.load_state_dict(quick["model"])
            candidate = validate("nominated_full_dev")
            candidate_score = score(candidate)
            if candidate_score is not None and (best is None or candidate_score>=best):
                best,best_step,selected_dev=candidate_score,quick_step,candidate
            else:
                model.load_state_dict(torch.load(last,map_location="cpu",weights_only=False)["model"])
        save_checkpoint(directory/"checkpoint_best.ckpt",dict(model=model.state_dict(),identity=identity,
                        side=side,step=best_step,score=best,split="full_dev",config=cfg))
        for domain,result in selected_dev.items():
            label="after" if best_step==step else "nominated_full_dev"
            write(directory/"selected"/(domain+".json"),dict(result,pair_records=f"{label}/{domain}_pairs.jsonl"))
        # Keep last at the fixed update, including the selected validation metadata.
        saved = torch.load(last,map_location="cpu",weights_only=False)
        saved.update(best=best,best_step=best_step,recent_dev=recent_dev)
        save_checkpoint(last,saved)
    else:
        probe_development = validate("probe_dev",True)
        if probe_development["megadepth"]["status"]!="PASS":
            raise RuntimeError("probe requires a successful real MegaDepth validation switch")
    development_status={k:v["status"] for k,v in (probe_development if probe else selected_dev).items()}
    result = dict(status="PASS",model=cfg["model"],side=side,step=step,precision=precision,coverage=coverage(),
                  development_status=development_status,
                  backbone_steps=backbone_steps,best=best,best_step=best_step,config=cfg,job=job,memory=guard.snapshot(),
                  seconds_per_step=float(np.mean(times[-2:])) if times else None,
                  pairs_per_s=effective/np.mean(times[-2:]) if times else None)
    write(directory/"training_summary.json",result)
    return result
