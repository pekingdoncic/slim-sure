import argparse
import math
import os
from pathlib import Path
from ..utils.run_io import EXP,ROOT,config


def parser():
    p = argparse.ArgumentParser(description="CCS-SLiM high-resolution training and frozen evaluation")
    p.add_argument("--mode",choices=("smoke","short","medium","full"),default="smoke")
    p.add_argument("--run-id",default="pro5000")
    p.add_argument("--gpu",default=os.environ.get("CUDA_VISIBLE_DEVICES","0"))
    p.add_argument("--python",default=os.environ.get("PYTHON","python"))
    p.add_argument("--data-root",type=Path,default=Path(os.environ.get("MEGADEPTH_ROOT",ROOT/"data/megadepth")))
    tp = ROOT/"data/Texture-PoorSfMDataset"
    p.add_argument("--texturepoor-root",type=Path,default=Path(os.environ.get("TEXTUREPOOR_ROOT",tp if tp.exists() else ROOT/"data/TexturePoorSfM_dataset")))
    p.add_argument("--scannet-root",type=Path,default=Path(os.environ.get("SCANNET_ROOT",ROOT/"data/scannet")))
    p.add_argument("--scannet-index-root",type=Path,default=Path(os.environ["SCANNET_INDEX_ROOT"]) if os.environ.get("SCANNET_INDEX_ROOT") else None)
    p.add_argument("--hpatches-root",type=Path,default=Path(os.environ.get("HPATCHES_ROOT",ROOT/"data/hpatches-sequences-release")))
    p.add_argument("--weights",type=Path,default=Path(os.environ.get("SLIM_WEIGHTS",ROOT/"ckpt/megadepth_19epochs.ckpt")))
    p.add_argument("--experiments",default="P0,P1,P2,P3,P4")
    p.add_argument("--resolution-plan",default="1024,1216")
    p.add_argument("--vram-cap-gib",type=float,default=30.)
    p.add_argument("--full-epochs",type=int)
    p.add_argument("--steps-1024",type=int)
    p.add_argument("--steps-1216",type=int)
    for s in ("resume-pipeline","monitor","dev-only"):
        p.add_argument("--"+s,action="store_true")
    return p


def plan(args):
    from ..utils.run_io import read
    sides = [int(s) for s in args.resolution_plan.replace("→",",").split(",")]
    models = args.experiments.upper().split(",")
    if sides not in ([1024],[1024,1216]) or len(models)!=len(set(models)) or not set(models)<=set("P0 P1 P2 P3 P4".split()):
        raise ValueError("use 1024 or 1024,1216 and distinct P0..P4")
    shared = config("P1")
    if args.mode!="full":
        if any(x is not None for x in (args.full_epochs,args.steps_1024,args.steps_1216)):
            raise ValueError("explicit epoch/step budgets are for full mode")
        return sides,models,[shared[args.mode][f"steps_{r}"] for r in sides]
    n = read(EXP/"manifests/train_pool.json")["legal_rows"]
    per_epoch = math.ceil(n/shared["effective_batch_size"])
    specified = [getattr(args,f"steps_{r}") for r in sides]
    if args.full_epochs is not None and not any(x is not None for x in specified):
        if args.full_epochs<1:
            raise ValueError("positive full epoch count required")
        total = args.full_epochs*per_epoch
        first = math.ceil(total*5/8)
        steps = [total] if len(sides)==1 else [first,total-first]
    elif args.full_epochs is None and all(x is not None and x>0 for x in specified):
        steps = specified
    else:
        raise ValueError("full requires --full-epochs N or explicit budgets for each planned resolution")
    if sum(steps)<per_epoch:
        raise ValueError(f"full budget must cover all {n} legal rows ({per_epoch} updates)")
    return sides,models,steps
