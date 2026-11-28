"""Single server operation. Each CUDA trial starts with an empty process allocator."""
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import time
import traceback
from argparse import Namespace
from pathlib import Path
from ..utils.run_io import EXP,ROOT,config,read,write,now,sha,lock
from .runtime import Guard,precision_name


def unpack(options):
    args = Namespace(**options)
    for k,v in vars(args).copy().items():
        if (k.endswith("_root") or k=="weights") and v is not None:
            setattr(args,k,Path(v))
    return args


def preflight(args,job,guard):
    import torch
    from ..data.dataset_adapter import split_check
    pool = split_check()
    versions = {}
    for lib in ("torch","torchvision","triton","einops","yacs","numpy","opencv-python","h5py","kornia",
                "timm","mamba-ssm","PyYAML","pytest","matplotlib"):
        try:
            versions[lib] = importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:
            versions[lib] = None
    # Native imports exercise loader/ABI availability; actual kernel behavior is tested next.
    from ..models.slim_wrapper import CCSMatcher
    from ..evaluation.pose import pair_metrics
    from ..data.texture_adapter import energy
    import matplotlib,pytest
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    free = shutil.disk_usage(EXP).free/1024**3
    if free<config("P1")["min_disk_gib"]:
        raise RuntimeError(f"insufficient experiment disk space {free:.2f} GiB")
    status = dict(platform=platform.platform(),python=sys.version,executable=sys.executable,dependencies=versions,
                  cuda=torch.version.cuda,device=guard.snapshot(),precision=precision_name(config("P1")["precision"]),
                  disk_free_gib=free,requested_cap_gib=args.vram_cap_gib,weight_sha256=sha(args.weights),
                  training_rows=pool["legal_rows"],physical_pairs=pool["unique_unordered"],time=now(),paths={})
    for k in ("data_root","texturepoor_root","scannet_root","scannet_index_root","hpatches_root","weights"):
        path = getattr(args,k)
        status["paths"][k] = dict(path=str(path) if path else None,exists=bool(path and path.exists()))
    from ..data.texturepoor_protocol import verify
    status["texturepoor_protocol"]={}
    for formal in (False,True):
        try:
            check=verify(args.texturepoor_root,formal)
        except FileNotFoundError as exc:
            check=dict(status="BLOCKED",reason=str(exc))
        status["texturepoor_protocol"]["test" if formal else "development"]=check
    status["precision_reason"]="BF16 supported by device; kernel trial pending" if status["precision"]=="bfloat16" else "FP16 selected by hardware capability"
    write(Path(job["report"])/"server_preflight.json",status)
    return status


def prepare(args,job):
    from ..data.dataset_adapter import prepare_pool,manifest
    from ..data.texture_adapter import calibrate
    cache = Path(job["cache"])
    report = Path(job["report"])
    if not (args.data_root/"index/trainvaltest_list/train_list.txt").is_file():
        write(report/"training_data.json",dict(status="BLOCKED",reason="missing official MegaDepth train_list"))
        return
    with lock(cache/"prepare.lock"):
        marker = cache/"identity.json"
        spec = {"manifest":sha(EXP/"manifests/train_pool.json"),"data_root":str(args.data_root),"version":job["source"]}
        # Re-check index identity on every new run; write isolated caches only.
        from ..data.dataset_adapter import index_path
        pool = manifest("train_pool")
        for b in pool["bins"]:
            index = index_path(args.data_root,b["bin"])
            if sha(index) not in (b["local_layout_sha256"],b["original_sha256"]):
                raise ValueError(f"training index differs: {index}")
        if (not marker.exists() or read(marker)!=spec or
            not all((cache/(b["bin"]+".npy")).is_file() for b in pool["bins"])):
            prepare_pool(args.data_root,cache)
            write(marker,spec)
    for side in job["calibration_sides"]:
        calibrate(args.data_root,cache,report/"calibration",side,config("P1"))
    write(report/"training_data.json",dict(status="PASS",**spec))


def run(path):
    import torch
    job = read(path)
    args = unpack(job["options"])
    cfg = config(job.get("model","P0"))
    action = job["action"]
    report = Path(job["report"])
    directory = Path(job["directory"])
    directory.mkdir(parents=True,exist_ok=True)
    guard = None
    try:
        if action=="prepare":
            try:
                prepare(args,job)
            except FileNotFoundError as exc:
                write(report/"training_data.json",dict(status="BLOCKED",reason=str(exc)))
            return
        if action=="probe":
            from .vram_probe import probe_configs
            probe_configs(args,cfg,job)
            return
        guard = Guard(args.vram_cap_gib)
        if action=="preflight":
            preflight(args,job,guard)
            return
        if action in ("train","trial"):
            from .trainer import train_stage
            result = train_stage(args,cfg,job,guard,probe=action=="trial")
            if action=="trial":
                guard.poll()
                mem = guard.snapshot()
                if mem["peak_process_gib"] is None:
                    raise RuntimeError("driver did not expose process memory; check nvidia-smi PID visibility")
                safe = mem["reserved_gib"]<=guard.target and mem["peak_process_gib"]<=guard.cap
                write(directory/"trial_result.json",dict(status="PASS" if safe else "OVER_BUDGET",oom=False,
                    micro=job["micro"],checkpointing=job["checkpointing"],precision=job["precision"],
                    pairs_per_s=result["pairs_per_s"],seconds_per_step=result["seconds_per_step"],memory=mem))
            return
        from ..models.slim_wrapper import CCSMatcher
        from ..evaluation.eval_hr import evaluate
        from ..evaluation.efficiency_benchmark import benchmark
        from .optimizer import seed_all
        seed_all(cfg["seed_model"])
        model = CCSMatcher(cfg,args.weights).cuda().eval()
        if job.get("checkpoint"):
            data = torch.load(job["checkpoint"],map_location="cpu",weights_only=False)
            if data["identity"]!=job["identity"]:
                raise ValueError("selected checkpoint and run identity differ")
            model.load_state_dict(data["model"],strict=True)
            del data
        if action=="benchmark":
            try:
                benchmark(model,args,job["side"],directory,report/"calibration",job["precision"],guard,args.mode=="smoke")
            except FileNotFoundError as exc:
                write(directory/"benchmark.json",dict(status="BLOCKED",reason=str(exc)))
        elif action=="eval":
            if job["split"]=="test":
                frozen = read(report/"model_freeze.json")
                selected = frozen["models"][cfg["model"]]
                checkpoint = job.get("checkpoint") or args.weights
                if args.mode!="full" or args.dev_only or selected["sha256"]!=sha(checkpoint) or frozen["identity"]!=job["identity"]:
                    raise ValueError("official test requires matching frozen weights and full protocol")
            evaluate(model,args,job["side"],job["domain"],job["split"],directory,report/"calibration",job["precision"],guard)
        else:
            raise ValueError(action)
    except Exception as exc:
        memory = guard.snapshot() if guard else {}
        info = dict(error=str(exc),traceback=traceback.format_exc(),memory=memory,job=job,time=now(),
                    command=[sys.executable,"-m",__spec__.name,str(path)])
        write(directory/"worker_error.json",info)
        if action=="trial":
            oom = isinstance(exc,torch.OutOfMemoryError) or any(k in str(exc).lower() for k in
                  ("out of memory","cublas_status_alloc_failed","cudnn_status_alloc_failed","cudaerrormemoryallocation"))
            from .runtime import MemoryPressure
            write(directory/"trial_result.json",dict(status="OOM" if oom else "OVER_BUDGET" if isinstance(exc,MemoryPressure) else "ERROR",
                  oom=oom,micro=job["micro"],checkpointing=job["checkpointing"],precision=job["precision"],
                  pairs_per_s=None,seconds_per_step=None,**info))
        raise
    finally:
        if guard:
            guard.close()


if __name__=="__main__":
    run(Path(sys.argv[1]))
