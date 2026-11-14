"""Pair-weighted BCE training over the complete immutable manifest."""
from collections import Counter
import gc
import os
import hashlib
import json
import math
from pathlib import Path
import time
import uuid
import numpy as np
import torch
from .runtime import (atomic_json,append_json,seed_all,save_checkpoint,load_checkpoint,
                      digest,now,resource_sample,state_hash,sha)

def array(value):
    if torch.is_tensor(value):return value.detach().cpu().numpy()
    return np.asarray(value)

def label_arrays(labels,device):
    y=torch.as_tensor(labels["y"],device=device,dtype=torch.float32).reshape(-1)
    valid=torch.as_tensor(labels["valid"],device=device,dtype=torch.bool).reshape(-1)
    return y,valid

def pair_seed(seed,epoch,pair_id):
    return int.from_bytes(hashlib.sha256(f"{seed}:{epoch}:{pair_id}".encode()).digest()[:8],"big")%(2**63-1)

def permutation(n,seed,epoch):
    return np.random.default_rng(np.random.SeedSequence([seed,epoch])).permutation(n)

def heartbeat(run,stage,event,**kwargs):
    atomic_json(Path(run)/"heartbeat.json",dict(pid=__import__("os").getpid(),time=now(),monotonic=time.monotonic(),stage=stage,event=event,**kwargs))

def train_step(runtime,record,epoch,position,log_path):
    config=runtime.config
    heartbeat(runtime.run,"training","pair_start",pair_id=record["pair_id"],epoch=epoch,position=position)
    started=time.perf_counter()
    runtime.head.train();runtime.optimizer.zero_grad(set_to_none=True)
    data,labels=runtime.forward(record,full=False)
    y,valid=label_arrays(labels,"cuda")
    if Path(log_path).parent.name=="engineering":
        case_dir=Path(log_path).parent/"label_cases";case_dir.mkdir(exist_ok=True)
        case=case_dir/(record["pair_id"]+".npz")
        if not case.exists():
            tmp=case.with_suffix(".tmp.npz")
            arrays=dict(raw_coarse_match_id=array(data["coarse_match_id"]),i_idx=array(data["i_idx_c"]),j_idx=array(data["j_idx_c"]),
                source_input=array(data["coarse_coord_0"]),target_input=array(data["coarse_coord_1"]),
                gt_target_input=labels["points1_gt_input"],labels=labels["labels"],ignore_reasons=labels["reasons"],
                residual_input=labels["residual_input"],stride_xy=labels["stride_xy"],
                native_confidence=array(data["coarse_confidence"]))
            with tmp.open("wb") as f:
                np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
            os.replace(tmp,case)
            atomic_json(case.with_suffix(".json"),dict(pair_id=record["pair_id"],direction=record.get("direction","0to1"),
                sha256=sha(case),identity_sha256=digest(runtime.run_identity),
                plan0=data["_gt"]["plan0"],plan1=data["_gt"]["plan1"],K0=data["_gt"]["K0"].tolist(),K1=data["_gt"]["K1"].tolist(),
                T_0to1=data["_gt"]["T_0to1"].tolist(),ignore_counts=labels["ignore_counts"],
                note="All actual parents on selected engineering pair; no descriptor/features cached"))
    count=int(valid.sum().item())
    result=dict(pair_id=record["pair_id"],scene=record["scene"],epoch=epoch,position=position,valid_parents=count,
                parents=len(y),ignore=len(y)-count,updated=False,timing=data["_timing"],ignore_reasons=labels.get("ignore_counts",{}))
    grad_copy=None
    if count:
        ids=torch.nonzero(valid).flatten()
        if count>config["max_parent_samples"]:
            gen=torch.Generator(device="cpu");gen.manual_seed(pair_seed(config["seed"],epoch,record["pair_id"]))
            selection=torch.randperm(count,generator=gen)[:config["max_parent_samples"]].to(ids.device)
            ids=ids[selection]
        # Slice before head construction, bounding live train graphs to 1024 parents.
        small={k:data[k][ids] for k in ("coarse_descriptor0","coarse_descriptor1","coarse_confidence")}
        torch.cuda.synchronize();head_started=time.perf_counter()
        logits=runtime.logits(small)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y[ids],reduction="mean")
        if not torch.isfinite(loss):raise FloatingPointError("Nonfinite BCE")
        loss.backward();runtime.calls["head_backward"]+=1
        grads=[p.grad for p in runtime.head.parameters()]
        if any(g is None or not torch.isfinite(g).all() for g in grads):raise FloatingPointError("Missing/nonfinite new gradient")
        if any(p.grad is not None for p in runtime.model.parameters()):raise AssertionError("Original model received gradient")
        grad_copy=torch.cat([g.detach().reshape(-1).cpu() for g in grads])
        if Path(log_path).parent.name=="engineering":
            ledger=Path(log_path).parent/"optimizer_intents.jsonl"
            with ledger.open("a") as f:
                f.write(json.dumps(dict(event="optimizer_intent",pair_id=record["pair_id"],epoch=epoch,position=position,time=now()))+"\n")
                f.flush();os.fsync(f.fileno())
        runtime.optimizer.step()
        if any(not torch.isfinite(p).all() for p in runtime.head.parameters()):raise FloatingPointError("Nonfinite new parameter")
        torch.cuda.synchronize()
        result.update(updated=True,loss=float(loss.detach()),sampled_parents=len(ids),
                      grad_norm=float(torch.linalg.vector_norm(grad_copy)),head_forward_backward_step_seconds=time.perf_counter()-head_started)
        del logits,loss,small,grads,ids
    else:
        result["skip_reason"]="no_coarse_parents" if len(y)==0 else "no_visible_depth_support"
        result["abnormal_gt_label_skip"]=False  # Contract/hash/anchor violations raise; visibility ignore is normal.
    del data,labels,y,valid
    runtime.optimizer.zero_grad(set_to_none=True)
    gc.collect();torch.cuda.synchronize()
    result.update(seconds=time.perf_counter()-started,resources=resource_sample())
    append_json(log_path,result)
    heartbeat(runtime.run,"training","pair_committed",pair_id=record["pair_id"],epoch=epoch,position=position,updated=result["updated"])
    return result,grad_copy

class Predictor:
    def __init__(self,runtime,full=False):
        self.runtime=runtime;self.full=full
    def __call__(self,record,mode="A_score"):
        runtime=self.runtime
        runtime.head.eval()
        data,labels=runtime.forward(record,full=self.full,texture=mode=="A_texture",fraction=.2 if mode=="A_texture" else 0)
        with torch.no_grad():
            torch.cuda.synchronize();t=time.perf_counter()
            logits=runtime.logits(data) if mode=="A_score" else torch.zeros(len(data["coarse_match_id"]),device="cuda")
            torch.cuda.synchronize();head_seconds=time.perf_counter()-t if mode=="A_score" else 0.
        from src.texture_budget import local_texture
        texture_started=time.perf_counter()
        texture,count=local_texture(data["image0"],data["valid_mask0"],data["hw0_c"])
        low=(texture[0]<=runtime.model.texture_threshold)&data["mask0"].flatten()&(count[0]>0)
        data["source_grid_low_texture"]=low.detach().cpu().numpy()
        torch.cuda.synchronize();texture_diagnostic_seconds=time.perf_counter()-texture_started
        y,valid=label_arrays(labels,"cuda")
        y_export=y.to(torch.int8);y_export[~valid]=-1
        result=dict(pair_id=record["pair_id"],scene=record["scene"],
            labels=array(y_export),logits=array(logits),coarse_confidence=array(data["coarse_confidence"]),
            norm0=array(data["coarse_descriptor0"].float().norm(dim=-1)),norm1=array(data["coarse_descriptor1"].float().norm(dim=-1)),
            coarse_match_id=array(data["coarse_match_id"]),ignore_reasons=labels.get("ignore_counts",{}),low_texture_mask=array(low[data["i_idx_c"]]),
            timing={**data["_timing"],"head":head_seconds,"texture_diagnostic":texture_diagnostic_seconds},
            memory=resource_sample())
        for key in ("candidate_counts","correlation_shapes","coarse_row_id","fine_parent_id"):
            if key in data:result[key]=array(data[key]) if torch.is_tensor(data[key]) else data[key]
        if self.full:
            from reliability_data import projection_diagnostics
            result.update(projection_diagnostics(data,labels))
        del data,labels,logits,y,valid,y_export
        gc.collect()
        return result

def validate(runtime,records,out_dir,checkpoint_id,run_identity):
    from .evaluation import evaluate_reliability
    runtime.head.eval()
    heartbeat(runtime.run,"validation","start",count=len(records),checkpoint_id=checkpoint_id)
    summary=evaluate_reliability(records,Predictor(runtime),out_dir,checkpoint_id=checkpoint_id,identity=run_identity)
    heartbeat(runtime.run,"validation","committed",count=len(records),checkpoint_id=checkpoint_id)
    return summary

def full_train(runtime,records,val_records,run_identity,resume=False):
    cfg=runtime.config;run=runtime.run;out=run/"full_train";out.mkdir(exist_ok=True)
    accepted=json.loads((run/"engineering_acceptance.json").read_text())
    if accepted["status"]!="pass" or accepted["identity"]!=run_identity:raise RuntimeError("Engineering acceptance missing/stale; full training blocked")
    last=out/"last.json";best=out/"best.json"
    if last.exists() and not resume:raise RuntimeError("Existing formal training: use --resume, never overwrite")
    state=dict(epoch=0,next_pos=0,global_step=0,attempted_pairs=0,skip_counts={},best_bce=None,best_epoch=None,
               epoch_updates=0,epoch_skips={},epoch_loss_sum=0.0,epoch_loss_count=0,validation_pending=False,abnormal_gt_label_skips=0)
    if resume:
        if not last.exists():raise RuntimeError("No formal checkpoint available for resume")
        obj=load_checkpoint(last,runtime.head,runtime.optimizer,run_identity)
        state=obj["position"]
        append_json(out/"events.jsonl",dict(event="resume",time=now(),checkpoint=sha(last),position=state,
            replay_note="Visits after the committed cursor, if any, are physical replays; logical coverage follows committed cursor only"))
    else:
        # Runtime initializes from seed2026, independent of engineering smoke.
        seed_all(cfg["seed"])
        initial=validate(runtime,val_records,out/"validation_init","initial:"+state_hash(runtime.head),run_identity)
        state["initial_pair_bce"]=initial.get("pair_bce")
        save_checkpoint(last,runtime.head,runtime.optimizer,state,run_identity)
    total=len(records)*cfg["epochs"]
    stage_start=time.perf_counter();step_times=[];resource_history=[]
    atomic_json(out/"status.json",dict(status="running",expected_epochs=cfg["epochs"],expected_visits=total,state=state,time=now()))
    if state["epoch"]==cfg["epochs"]:
        if state["attempted_pairs"]!=total:raise RuntimeError("Completed checkpoint has incomplete full-manifest visit count")
        atomic_json(out/"status.json",dict(status="committed",state=state,expected_epochs=cfg["epochs"],expected_visits=total,time=now(),calls=runtime.calls,resumed_completed_checkpoint=True))
        return state
    for epoch in range(state["epoch"],cfg["epochs"]):
        order=permutation(len(records),cfg["seed"],epoch)
        order_path=out/f"epoch_{epoch:02d}_order.npy"
        if order_path.exists():
            if not np.array_equal(np.load(order_path),order):raise RuntimeError("Saved epoch permutation changed")
        else:
            tmp=order_path.with_suffix(".tmp.npy")
            with tmp.open("wb") as f:
                np.save(f,order);f.flush();os.fsync(f.fileno())
            os.replace(tmp,order_path)
        state["order_path"]=order_path.name;state["order_sha256"]=sha(order_path)
        start_pos=state["next_pos"] if state["epoch"]==epoch else 0
        generation=uuid.uuid4().hex
        for position in range(start_pos,len(order)):
            record=records[int(order[position])]
            result,_=train_step(runtime,record,epoch,position,out/"physical_visits.jsonl")
            append_json(out/"visits.jsonl",dict(epoch=epoch,position=position,pair_id=record["pair_id"],updated=result["updated"],generation=generation))
            state["epoch"]=epoch;state["next_pos"]=position+1;state["attempted_pairs"]+=1
            if result["updated"]:
                state["global_step"]+=1;state["epoch_updates"]+=1
                state["epoch_loss_sum"]+=result["loss"];state["epoch_loss_count"]+=1
            else:
                reason=result["skip_reason"]
                state["skip_counts"][reason]=state["skip_counts"].get(reason,0)+1
                state["epoch_skips"][reason]=state["epoch_skips"].get(reason,0)+1
            resource_history.append(result["resources"])
            if len(resource_history)>200:resource_history.pop(0)
            resource_error=None
            if len(resource_history)>=50:
                allocated=[x["allocated"] for x in resource_history]
                if min(allocated[-5:])-max(allocated[:20])>128*2**20:
                    resource_error="Persistent GPU allocation growth after warmup"
            meminfo={line.split(":")[0]:int(line.split()[1])*1024 for line in Path("/proc/meminfo").read_text().splitlines() if len(line.split())>=2 and line.split()[1].isdigit()}
            if meminfo.get("MemAvailable",2**40)<2**30:resource_error="WSL available memory fell below 1 GiB"
            if __import__("shutil").disk_usage(run).free<cfg.get("minimum_free_gib",2)*2**30:resource_error="Disk reserve below configured 2 GiB"
            if resource_error:
                save_checkpoint(last,runtime.head,runtime.optimizer,state,run_identity)
                atomic_json(out/"resource_stop.json",dict(reason=resource_error,state=state,resource=result["resources"],time=now()))
                raise RuntimeError(resource_error)
            step_times.append(result["seconds"])
            if len(step_times)>200:step_times.pop(0)
            state["abnormal_gt_label_skips"]=state.get("abnormal_gt_label_skips",0)+int(result.get("abnormal_gt_label_skip",False))
            if state["attempted_pairs"]==100:
                atomic_json(out/"first100_label_gate.json",dict(attempted=100,
                    abnormal_gt_label_skips=state["abnormal_gt_label_skips"],
                    normal_skips=sum(state["skip_counts"].values())-state["abnormal_gt_label_skips"],
                    skip_reasons=state["skip_counts"],contract_errors="raise immediately rather than skip"))
                if state["abnormal_gt_label_skips"]>20:
                    save_checkpoint(last,runtime.head,runtime.optimizer,state,run_identity)
                    raise RuntimeError("More than 20% abnormal GT/label skips in first100; inspect contract before continuation")
            if result["updated"] and state["global_step"]%1000==0:
                save_checkpoint(last,runtime.head,runtime.optimizer,state,run_identity)
            if state["attempted_pairs"]==200:
                atomic_json(out/"eta_after_200.json",dict(estimated=True,mean_pair_seconds=float(np.mean(step_times)),
                    remaining_training_seconds=float(np.mean(step_times))*(total-state["attempted_pairs"]),
                    validation_not_included=True,load_seconds=runtime.load_seconds))
        state["validation_pending"]=True
        save_checkpoint(last,runtime.head,runtime.optimizer,state,run_identity)
        runtime.check_frozen()
        checkpoint_id="epoch_"+str(epoch)+":"+state_hash(runtime.head)
        summary=validate(runtime,val_records,out/f"validation_epoch_{epoch:02d}",checkpoint_id,run_identity)
        bce=summary.get("pair_bce")
        coverage=dict(epoch=epoch,expected_pairs=len(records),attempted_pairs=state["next_pos"],unique_pairs_seen=state["next_pos"],
            coverage=state["next_pos"]/len(records),order_sha256=state["order_sha256"],valid_updates=state["epoch_updates"],
            skips=state["epoch_skips"],train_pair_bce=state["epoch_loss_sum"]/state["epoch_loss_count"] if state["epoch_loss_count"] else None,
            val_pair_bce=bce,checkpoint_id=checkpoint_id,complete=state["next_pos"]==len(records))
        atomic_json(out/f"epoch_{epoch:02d}_coverage.json",coverage)
        if bce is not None and (state["best_bce"] is None or bce<state["best_bce"]):
            state["best_bce"]=bce;state["best_epoch"]=epoch
            save_checkpoint(best,runtime.head,runtime.optimizer,state,run_identity,extra={"checkpoint_id":checkpoint_id})
        state.update(epoch=epoch+1,next_pos=0,epoch_updates=0,epoch_skips={},epoch_loss_sum=0.0,epoch_loss_count=0,validation_pending=False)
        save_checkpoint(last,runtime.head,runtime.optimizer,state,run_identity,extra={"checkpoint_id":checkpoint_id})
        atomic_json(out/"status.json",dict(status="running" if state["epoch"]<cfg["epochs"] else "committed",state=state,expected_epochs=cfg["epochs"],
            expected_visits=total,time=now(),elapsed_seconds=time.perf_counter()-stage_start,calls=runtime.calls,
            eta_estimated_seconds=float(np.mean(step_times))*(total-state["attempted_pairs"]) if step_times else None))
    return state
