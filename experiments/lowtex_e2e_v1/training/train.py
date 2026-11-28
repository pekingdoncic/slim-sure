import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch
import yaml

from configs.cfg_outdoor_train import get_cfg_defaults
from src.loss import SLiM_Loss
from ..evaluation.development import validate
from ..evaluation.texturepoor_sfm.dataset import EXP,ROOT,DATA
from ..models.slim_wrapper import ControlledSLiM
from .data import load_pair
from .optimization import UpdatePolicy
from .sampling import Pool


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path,value):
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False,default=str)+"\n")
    tmp.replace(path)


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def parser():
    p=argparse.ArgumentParser()
    p.add_argument("stage",nargs="?",default="preflight",choices=["preflight","prepare_texturepoor","baseline","smoke","exp0","exp1","exp2","evaluate"])
    p.add_argument("--profile",choices=["r4080","large_vram"],default="r4080")
    p.add_argument("--mode",choices=["smoke","full"],default=None)
    p.add_argument("--data-root",type=Path,default=ROOT/"data/megadepth")
    p.add_argument("--texturepoor-root",type=Path,default=DATA)
    p.add_argument("--weights",type=Path,default=ROOT/"ckpt/megadepth_19epochs.ckpt")
    p.add_argument("--output-dir",type=Path,default=EXP/"outputs")
    p.add_argument("--checkpoint",type=Path)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--gpu",default=None)
    p.add_argument("--seed",type=int,default=66)
    p.add_argument("--exp",default="exp0")
    p.add_argument("--dataset",choices=["both","texturepoor","megadepth"],default="both")
    p.add_argument("--split",choices=["val","test"],default="val")
    p.add_argument("--resolution",type=int)
    p.add_argument("--until-update",type=int,help="engineering smoke checkpoint boundary")
    p.add_argument("--unfreeze",choices=["head_only","partial_unfreeze","end_to_end"],default="end_to_end")
    return p


def verify():
    st=json.loads((EXP/"reports/state.json").read_text())
    assert st["protocol_sha256"]==sha(EXP/"protocol_frozen.md")
    for name,h in json.loads((EXP/"manifests/dev_manifest_hashes.json").read_text()).items():
        assert sha(EXP/"manifests"/name)==h,name


def state_rng():
    return dict(py=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),gpu=torch.cuda.get_rng_state())


def restore_rng(r):
    random.setstate(r["py"]);np.random.set_state(r["numpy"]);torch.set_rng_state(r["cpu"]);torch.cuda.set_rng_state(r["gpu"])


def train(args,cfg,settings,model,d,manifest,pool):
    full=args.mode=="full"
    accum=settings["accumulate"]
    total=int(pool.ends[-1]//accum) if full else settings["smoke_updates"]
    if full and pool.ends[-1]%accum:raise ValueError("full budget not divisible by accumulation")
    loss_fn=SLiM_Loss(cfg.LOSS)
    policy=UpdatePolicy(model,None,settings,total,args.unfreeze)
    opt,scheduler=policy.optimizer,policy.scheduler
    last=d/"checkpoint_last.pt"
    identity=sha(d/"run_manifest.json")
    step=cursor=0
    observed=np.zeros(len(pool.bins),dtype=np.int64)
    unique=observed.copy()
    best={"lowtex":None,"general":None}
    physical=None
    physical_ids={}
    if full:
        index=json.loads((pool.cache/"unordered_identity.json").read_text())
        physical=np.zeros(index["total"],dtype=np.bool_)
    def physical_count():
        return int(physical.sum()) if physical is not None else None
    audit_bn=model.bn_snapshot()
    if args.resume:
        saved=torch.load(last,map_location="cpu")
        if saved["identity"]!=identity:raise ValueError("checkpoint run identity mismatch")
        model.load_state_dict(saved["model"]);opt.load_state_dict(saved["optimizer"]);scheduler.load_state_dict(saved["scheduler"])
        step,cursor=saved["step"],saved["cursor"]
        unique=saved["unique"];observed=saved["observed"];best=saved["best"];policy.apply(step)
        if not all(torch.equal(model.bn_snapshot()[k].cpu(),v) for k,v in saved["bn"].items()):raise ValueError("BN snapshot drift on resume")
        if full:
            physical[:]=np.unpackbits(saved["physical_bits"],count=len(physical)).astype(bool)
        restore_rng(saved["rng"])
        print("resumed",step,cursor,flush=True)

    def checkpoint():
        tmp=last.with_suffix(".tmp")
        torch.save(dict(identity=identity,model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
            scaler=None,physical_bits=np.packbits(physical) if full else None,rng=state_rng(),step=step,cursor=cursor,unique=unique,observed=observed,best=best,
            bn={k:v.cpu() for k,v in model.bn_snapshot().items()}),tmp)
        tmp.replace(last)
        atomic(d/"progress.json",dict(step=step,planned_updates=total,cursor=cursor,unique_rows=int(unique.sum()),
              unique_unordered=physical_count(),unordered_coverage=(physical_count()/len(physical) if full else None),coverage=float(unique.sum()/pool.sizes.sum()),per_bin={b["bin"]:[int(n),int(u),int(b["legal_rows"])] for b,n,u in zip(pool.bins,observed,unique)}))

    def evaluate():
        v=d/"validation"/f"update_{step:08d}"
        receipt=v/"accepted.json"
        if receipt.is_file():
            scores=json.loads(receipt.read_text())
        else:
            attempt=v/stamp()
            scores=validate(model,args.resolution,args.data_root,attempt,not full)
            atomic(receipt,scores)
        for tag,val in [("lowtex",scores["megadepth"]["scene_equal"]["lt_recall"]),("general",scores["megadepth"]["scene_equal"]["auc@10"])]:
            if val is not None and (best[tag] is None or val>best[tag]):
                best[tag]=val
                ck=d/("checkpoint_best_"+tag+".pt")
                tmp=ck.with_suffix(".tmp")
                torch.save(dict(identity=identity,step=step,model=model.state_dict(),score=val),tmp);tmp.replace(ck)
        checkpoint()

    if step==0 or step==total or step%settings["eval_interval_updates"]==0:
        checkpoint()
        evaluate()
    stop=args.until_update if args.until_update is not None else total
    if stop>total or (full and stop!=total):raise ValueError("partial stop only valid for smoke")
    segment=d/("train_metrics_"+stamp()+".jsonl")
    max_mem=[0,0]
    with segment.open("x") as log:
        while step<stop:
            policy.apply(step)
            opt.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            records=[]
            before=time.perf_counter()
            reference=model.coarse_temperature.detach().clone()
            for micro in range(accum):
                bi,row,frames,new=pool.at(cursor)
                name=pool.bins[bi]["bin"]
                key=f"{name}/{row}"
                print("visit",cursor,key,flush=True) if step==0 else None
                start=time.perf_counter()
                batch,_=load_pair(args.data_root,name,frames,args.resolution,"cuda",key)
                loaded=time.perf_counter()-start
                torch.cuda.synchronize()
                forward_start=time.perf_counter()
                loss,metrics=model.train_pair(batch,cfg,loss_fn,torch.bfloat16)
                if not torch.isfinite(loss):raise FloatingPointError(f"nonfinite loss at {key}: {batch['loss_scalars']}")
                torch.cuda.synchronize()
                forward_s=time.perf_counter()-forward_start
                backward_start=time.perf_counter()
                (loss/accum).backward()
                torch.cuda.synchronize()
                metrics.update(forward_s=forward_s,backward_s=time.perf_counter()-backward_start)
                metrics.update(loss=float(loss.detach()),pair=key,load_s=loaded)
                records.append(metrics);cursor+=1;unique[bi]+=int(new);observed[bi]+=1
                if full:
                    if bi not in physical_ids:
                        idxfile=pool.cache/(name+".unordered.npy")
                        if sha(idxfile)!=index["hashes"][idxfile.name]:raise ValueError(idxfile)
                        physical_ids[bi]=np.load(idxfile,mmap_mode="r")
                    physical[physical_ids[bi][row]]=True
                del loss,batch
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),settings["clip_norm"],error_if_nonfinite=True)
            if grad<=0:raise RuntimeError("zero target gradient")
            update_start=time.perf_counter()
            opt.step();scheduler.step();step+=1
            torch.cuda.synchronize()
            update_s=time.perf_counter()-update_start
            if not all(torch.equal(v,audit_bn[k]) for k,v in model.bn_snapshot().items()):raise RuntimeError("BN running statistics drifted")
            peak=[torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved()]
            max_mem=list(map(max,zip(max_mem,peak)))
            log.write(json.dumps(dict(step=step,seconds=time.perf_counter()-before,grad_norm=float(grad),
               update_s=update_s,parameter_delta=float((model.coarse_temperature.detach()-reference).abs()),peak_bytes=peak,
               micro=records,seen=int(unique.sum()),cursor=cursor))+"\n");log.flush()
            if step in (1,stop) or step%settings["checkpoint_interval_updates"]==0:checkpoint()
            if step%settings["eval_interval_updates"]==0 or step==total:evaluate()
    previous=json.loads((d/"memory_stats.json").read_text()) if (d/"memory_stats.json").exists() else {}
    for metrics_file in d.glob("train_metrics_*.jsonl"):
        if metrics_file==segment or previous:
            continue
        with metrics_file.open() as source:
            for line in source:
                peak=json.loads(line)["peak_bytes"]
                max_mem=list(map(max,zip(max_mem,peak)))
    atomic(d/"memory_stats.json",dict(max_allocated_bytes=max(max_mem[0],previous.get("max_allocated_bytes",previous.get("segment_max_allocated",0))),
                                    max_reserved_bytes=max(max_mem[1],previous.get("max_reserved_bytes",previous.get("segment_max_reserved",0)))))
    checkpoint()
    if full and step==total and (not np.array_equal(unique,pool.sizes) or not physical.all()):raise RuntimeError("coverage incomplete")
    atomic(d/"summary.json",dict(status="PASS" if step==total else "RUNNING",mode=args.mode,step=step,planned=total,seen=int(unique.sum()),cursor=cursor))
    print("updates",step,"of",total,"seen",int(unique.sum()),flush=True)


def main():
    args=parser().parse_args()
    verify()
    args.mode=args.mode or ("smoke" if args.stage=="smoke" else "full")
    if args.split=="test": raise SystemExit("Final testing requires Phase 08 frozen model receipt")
    if args.exp!="exp0" or args.stage in ("exp1","exp2"):raise SystemExit("requested experiment has not passed its implementation gate")
    if args.texturepoor_root.resolve()!=DATA.resolve():raise ValueError("alternate dataset must match frozen manifest root")
    if args.stage=="prepare_texturepoor":
        from ..evaluation.texturepoor_sfm.build_pairs import main as prepare
        return prepare()
    settings={}
    for name in ("base",args.profile,"exp0_baseline"):
        settings.update(yaml.safe_load((EXP/"configs"/(name+".yaml")).read_text()))
    args.resolution=args.resolution or settings["resolution"]
    if args.resolution%32 or args.resolution<=0:raise ValueError("resolution must be positive and divisible by 32")
    if args.stage=="preflight":
        print(dict(device=torch.cuda.get_device_name(),torch=torch.__version__,allocator=torch.cuda.memory.get_allocator_backend(),
              allocator_env={k:os.getenv(k) for k in ("PYTORCH_CUDA_ALLOC_CONF","PYTORCH_ALLOC_CONF")},data=str(args.data_root),weights=args.weights.is_file()))
        return
    args.output_dir=args.output_dir.resolve()
    if EXP not in args.output_dir.parents:raise ValueError("output must be inside experiment")
    if args.resume and not args.checkpoint:raise ValueError("--resume requires --checkpoint")
    if args.checkpoint and not args.resume and args.stage!="evaluate":raise ValueError("training checkpoint requires --resume")
    if args.resume:
        if args.checkpoint.name!="checkpoint_last.pt" or EXP not in args.checkpoint.resolve().parents:raise ValueError("expected experiment checkpoint_last.pt")
        d=args.checkpoint.resolve().parent
    else:
        d=args.output_dir/(args.stage+"_"+args.mode)/stamp()
        d.mkdir(parents=True,exist_ok=False)
    print("run",d,flush=True)
    lock=(d/"writer.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    class Tee:
        def __init__(self,stream,log):
            self.stream=stream
            self.log=log
        def write(self,text):
            self.stream.write(text);self.log.write(text);self.log.flush()
            return len(text)
        def flush(self):
            self.stream.flush();self.log.flush()
    logfile=(d/"stdout.log").open("a",buffering=1)
    sys.stdout=Tee(sys.stdout,logfile);sys.stderr=Tee(sys.stderr,logfile)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(2)
    cfg=get_cfg_defaults()
    manifest=dict(seed=args.seed,stage=args.stage,mode=args.mode,profile=args.profile,resolution=args.resolution,args_unfreeze=args.unfreeze,
                data_root=str(args.data_root.resolve()),weights_sha256=sha(args.weights),pool_sha256=sha(EXP/"manifests/train_pool.json"),
                protocol_sha256=sha(EXP/"protocol_frozen.md"),settings=settings,torch=str(torch.__version__),gpu=torch.cuda.get_device_name(),
                allocator_env={k:os.getenv(k) for k in ("PYTORCH_CUDA_ALLOC_CONF","PYTORCH_ALLOC_CONF")},allocator=torch.cuda.memory.get_allocator_backend(),
                git=__import__("subprocess").check_output(["git","rev-parse","HEAD"],text=True).strip())
    if args.mode=="full" and args.stage=="exp0":
        manifest.update(full_budget_sha256=sha(EXP/"manifests/full_budget.json"),
             code_sha256={str(p.relative_to(EXP)):sha(p) for folder in ("models","training","evaluation") for p in sorted((EXP/folder).rglob("*.py"))},
             source_identity_sha256=sha(EXP/"manifests/source_identity.json"))
    path=d/"run_manifest.json"
    if args.resume:
        if json.loads(path.read_text())!=manifest:raise ValueError("resolved training identity changed")
    else:
        atomic(path,manifest)
        (d/"resolved_config.yaml").write_text(yaml.safe_dump(manifest))
    if args.mode=="full" and args.stage=="exp0":
        atomic(EXP/"reports/full_run.json",dict(directory=str(d.relative_to(EXP)),identity=sha(path),seed=args.seed))
    model=ControlledSLiM(cfg.MODEL,freeze_bn=settings["freeze_backbone_bn_stats"],checkpointing=settings["gradient_checkpointing"],
                         fine_cap=settings["max_fine_train_matches"]).cuda()
    model.load_state_dict(torch.load(args.weights,map_location="cpu")["state_dict"],strict=True)
    if args.stage=="evaluate" and args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint,map_location="cpu")["model"])
    if args.stage in ("baseline","evaluate"):
        domains=("texturepoor","megadepth") if args.dataset=="both" else (args.dataset,)
        return validate(model,args.resolution,args.data_root,d/"development",args.mode=="smoke",domains)
    if args.dataset!="both":raise ValueError("training requires both validation routes")
    if args.mode=="full":
        receipt=EXP/"manifests/full_budget.json"
        if not receipt.is_file():raise SystemExit("full budget and real smoke acceptance pending")
        frozen=json.loads(receipt.read_text())
        if (frozen["resolution"]!=args.resolution or frozen["pool_sha256"]!=manifest["pool_sha256"]
            or frozen["weights_sha256"]!=manifest["weights_sha256"] or frozen["protocol_sha256"]!=manifest["protocol_sha256"]
            or args.unfreeze!="end_to_end" or args.seed not in frozen["seeds"] or args.profile!="r4080"):
            raise ValueError("full budget mismatch")
    pool=Pool(EXP,args.data_root,args.seed)
    train(args,cfg,settings,model,d,manifest,pool)


if __name__=="__main__":
    main()
