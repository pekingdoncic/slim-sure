"""One-pair GPU export/timing worker. Launched only by the evaluate stage."""
import argparse
import errno
import gc
import json
import os
from pathlib import Path
import time
import uuid

from .geometry_worker import atomic_json,file_sha
from .exports import MODES,save_prediction,worker_identity,json_value




def arm_parent_death_signal():
    """WSL/Linux kernel terminates this worker if the supervising parent vanishes."""
    import ctypes
    import signal
    expected=int(os.environ["SLIM_EXPORT_PARENT_PID"])
    if os.getppid()!=expected:
        raise RuntimeError("Export parent disappeared before worker initialization")
    library=ctypes.CDLL(None,use_errno=True)
    library.prctl.restype=ctypes.c_int
    # Linux PR_SET_PDEATHSIG. No driver/environment mutation.
    if library.prctl(1,int(signal.SIGTERM),0,0,0)!=0:
        raise OSError(ctypes.get_errno(),"PR_SET_PDEATHSIG failed")
    # Close the race between reading ppid and installing the kernel death signal.
    if os.getppid()!=expected:
        raise RuntimeError("Export parent disappeared while arming death signal")


def heartbeat(directory,phase,**fields):
    atomic_json(Path(directory)/"heartbeat.json",dict(phase=phase,token=uuid.uuid4().hex,
        time=time.time(),pid=os.getpid(),**fields))


def inference_timing(runtime,record,mode):
    """Time production image preprocessing/model/head; never generate GT diagnostics."""
    import torch
    from reliability_data import load_pair
    from .runtime import resource_sample
    runtime.head.eval();runtime.model.eval()
    runtime.model.texture_budget_enabled=mode=="A_texture"
    runtime.model.texture_fraction=.2 if mode=="A_texture" else 0.
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
    begin=time.perf_counter();io_begin=begin
    data=load_pair(record,device="cuda",resolution=runtime.config["resolution"],include_gt=False)
    torch.cuda.synchronize();io_seconds=time.perf_counter()-io_begin
    model_begin=time.perf_counter()
    with torch.no_grad(),torch.cuda.amp.autocast(enabled=runtime.config["amp"]):
        runtime.model(data,coarse_only=False)
    torch.cuda.synchronize();model_seconds=time.perf_counter()-model_begin
    runtime.calls["backbone"]+=1;runtime.calls["original_fine"]+=1
    head_seconds=0.
    if mode=="A_score":
        head_begin=time.perf_counter()
        with torch.no_grad():logits=runtime.logits(data)
        torch.cuda.synchronize();head_seconds=time.perf_counter()-head_begin
        del logits
    torch.cuda.synchronize()
    elapsed=time.perf_counter()-begin
    components={"image_io_preprocess_h2d_seconds":io_seconds,"model_full_seconds":model_seconds,
        "projection_head_seconds":head_seconds}
    for key in ("feat_extract_time","upsample_time","coarse_match_time","fine_match_time","refine_time",
                "texture_selection_seconds","texture_compute_seconds"):
        if key in data:components[key]=float(data[key])
    memory=resource_sample()
    row={"pair_id":record["pair_id"],"mode":mode,"seconds":elapsed,"components":components,
         "memory":memory,"candidate_counts":json_value(data.get("candidate_counts")),
         "correlation_shapes":json_value(data.get("correlation_shapes")),
         "GT_depth_io":False,"GT_diagnostics":False}
    del data;gc.collect()
    return row


def run(request_path,attempt_dir):
    attempt=Path(attempt_dir);attempt.mkdir(parents=True,exist_ok=True)
    heartbeat(attempt,"cold_load")
    request=json.loads(Path(request_path).read_text())
    cfg=json.loads(Path(request["config_path"]).read_text())
    record=request["record"];purpose=request["purpose"];out=Path(request["output"])
    if purpose not in ("export","timing_only"):raise ValueError("Unknown worker purpose")
    expected=worker_identity(request["config_path"],request["run_identity"],request["checkpoint"],
                             request["calibration"],record,purpose)
    if expected!=request["identity"]:raise ValueError("Export request identity changed")
    started=time.perf_counter()
    from .runtime import Runtime,load_checkpoint,identity
    if identity(cfg,request["run_identity"]["data"])!=request["run_identity"]:
        raise ValueError("Code/config/data/profile no longer match engineering acceptance")
    import torch
    runtime=Runtime(cfg,attempt/"runtime")
    load_checkpoint(request["checkpoint"],runtime.head,None,request["run_identity"],restore_rng=False)
    frozen=json.loads(Path(request["calibration"]).read_text())
    if frozen.get("status")!="committed":raise ValueError("Texture calibration is not committed")
    runtime.model.texture_threshold=float(frozen["threshold"])
    runtime.head.eval()
    torch.cuda.synchronize();cold=time.perf_counter()-started
    checksum_paths=[]
    if purpose=="export":
        from .trainer import Predictor
        predict=Predictor(runtime,full=True)
        for mode in MODES:
            heartbeat(attempt,"inference",mode=mode,event="start",pair_id=record["pair_id"])
            begin=time.perf_counter();prediction=predict(record,mode)
            prediction["diagnostic_export_wall_seconds"]=time.perf_counter()-begin
            prediction["executed_forward"]=True
            save_prediction(out/mode,prediction,dict(expected,mode=mode))
            checksum_paths.extend(mode+"/"+name for name in ("arrays.npz","metadata.json","commit.json"))
            del prediction;gc.collect()
            heartbeat(attempt,"inference",mode=mode,event="committed",pair_id=record["pair_id"])
    else:
        rows=[]
        for mode in MODES:
            for iteration in range(13):
                heartbeat(attempt,"inference",mode=mode,iteration=iteration,event="start",pair_id=record["pair_id"])
                row=inference_timing(runtime,record,mode)
                row.update(iteration=iteration,purpose="warmup" if iteration<3 else "steady_repeat")
                rows.append(row)
                heartbeat(attempt,"inference",mode=mode,iteration=iteration,event="committed",pair_id=record["pair_id"])
        atomic_json(out/"timing_records.json",rows);checksum_paths.append("timing_records.json")
    heartbeat(attempt,"finalizing",event="frozen_state_check")
    runtime.check_frozen()
    summary={"status":"committed","purpose":purpose,"pair_id":record["pair_id"],"unique_pairs":1,
        "cold_load_seconds":cold,"worker_wall_seconds":time.perf_counter()-started,
        "calls":runtime.calls,"gpu_concurrency":1,"checkpoint":request["checkpoint"],
        "no_training_or_optimizer_step":True,"teacher_calls":0,"sure_backbone_calls":0}
    atomic_json(out/"worker_summary.json",summary);checksum_paths.append("worker_summary.json")
    atomic_json(out/"commit.json",{"status":"committed","identity":expected,
        "sha256":{name:file_sha(out/name) for name in checksum_paths}})
    heartbeat(attempt,"committed",pair_id=record["pair_id"])
    del runtime;gc.collect();torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--request",required=True)
    parser.add_argument("--attempt-dir",required=True)
    args=parser.parse_args()
    try:
        arm_parent_death_signal()
        from .evaluation import writer_lock
        request=json.loads(Path(args.request).read_text())
        # Held until model teardown, including if the supervising parent disappears.
        with writer_lock(Path(request["controller_root"])/"gpu_worker_lock"):
            run(args.request,args.attempt_dir)
    except BaseException as error:
        text=str(error).lower()
        critical=any(word in text for word in ("out of memory","nonfinite","nan","illegal memory","device-side assert"))
        transient=isinstance(error,OSError) and error.errno in (errno.EAGAIN,errno.EINTR)
        atomic_json(Path(args.attempt_dir)/"failure.json",{"status":"failed","error":repr(error),
            "error_type":type(error).__name__,"retryable":bool(transient and not critical),
            "resource_or_numerical_failure":critical,"time":time.time()})
        raise


if __name__=="__main__":main()
