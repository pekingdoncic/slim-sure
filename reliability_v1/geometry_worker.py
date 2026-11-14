"""CPU-only worker for the already established canonical MegaDepth geometry."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ARRAY_KEYS=("points0_orig","points1_orig","K0","K1","T_0to1")


def file_sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1048576),b""):h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def array_sha(value):
    a=np.ascontiguousarray(value)
    h=hashlib.sha256(str(a.dtype).encode()+json.dumps(a.shape).encode()+a.tobytes())
    return h.hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp."+str(os.getpid()))
    with tmp.open("w") as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def validate_protocol(protocol):
    if protocol["geometry"]!={"backend":"LO-RANSAC","pixel_threshold":0.5,"confidence":0.99999}:
        raise ValueError("Canonical geometry changed")
    if protocol["seeds"]!=list(range(5)):raise ValueError("Expected external seeds 0 through 4")
    for path,expected in protocol["source_sha256"].items():
        if file_sha(path)!=expected:raise ValueError("Geometry source changed: "+path)


def prediction_identity(pred,protocol):
    return {"arrays":{k:array_sha(pred[k]) for k in ARRAY_KEYS},
            "protocol_sha256":digest(protocol),"point_order":"native_order_then_RandomState(seed).permutation"}


def solve_file(prediction,protocol_path,output):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise RuntimeError("CPU geometry worker must hide CUDA")
    protocol=json.loads(Path(protocol_path).read_text());validate_protocol(protocol)
    sys.path.insert(0,str(Path(protocol["hybrid_root"]).parent))
    from hybrid_match.evaluation.pose import estimate_pose,backend_information
    from hybrid_match.evaluation.metrics import pose_error
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with np.load(prediction,allow_pickle=False) as f:
        pred={k:f[k] for k in ARRAY_KEYS};pid=str(f["pair_id"].item())
    identity=prediction_identity(pred,protocol)
    source_hash=file_sha(prediction)
    atomic_json(output/"backend.json",backend_information("LO-RANSAC"))
    computed=backend_calls=0
    for seed in protocol["seeds"]:
        target=output/("seed_%d.json"%seed)
        one_identity=dict(identity,pair_id=pid,seed=seed)
        checksum_path=output/("seed_%d_commit.json"%seed)
        if target.exists() and checksum_path.exists():
            committed=json.loads(target.read_text())
            if not checksum_path.exists() or json.loads(checksum_path.read_text())["sha256"]!=file_sha(target):
                raise ValueError("Geometry cache checksum failed")
            if committed.get("identity")!=one_identity or committed.get("status")!="committed":
                raise ValueError("Geometry cache identity differs or incomplete")
            continue
        state=output/("seed_%d_state.json"%seed)
        atomic_json(state,{"status":"running","identity":one_identity,"started_unix":time.time()})
        with (output/"physical_attempts.jsonl").open("a") as log:
            log.write(json.dumps({"seed":seed,"status":"started","time":time.time()})+"\n")
            log.flush();os.fsync(log.fileno())
        try:
            result=estimate_pose(pred["points0_orig"],pred["points1_orig"],pred["K0"],pred["K1"],protocol["geometry"],seed)
            computed+=1;backend_calls+=int(result["backend_invoked"])
            if result["status"]=="success":
                rotation,translation=pose_error(pred["T_0to1"],result["R"],result["t"])
                # Degenerate GT translation is a data error, not solver failure.
                if not np.isfinite([rotation,translation]).all():raise ValueError("Undefined GT pose error")
                error=max(rotation,translation)
            else:rotation=translation=error=None
            record={"schema":"slim_reliability_geometry_v1","status":"committed","identity":one_identity,
                    "prediction_sha256":source_hash,"solver_status":result["status"],
                    "failure_reason":result["failure_reason"],"R_error_deg":rotation,"t_error_deg":translation,
                    "pose_error_deg":error,"num_inliers":int(result["num_inliers"]),
                    "geometry_seconds":result["geometry_seconds"],"backend_invoked":bool(result["backend_invoked"]),
                    "backend_seed":result["backend_seed"],"backend_seed_controlled":result["backend_seed_controlled"],
                    "seed_control":result["seed_control"],"resolved_geometry":result["resolved_geometry"],
                    "R":result["R"].tolist() if result["R"] is not None else None,
                    "t":result["t"].tolist() if result["t"] is not None else None}
            with (output/"physical_attempts.jsonl").open("a") as log:
                log.write(json.dumps({"seed":seed,"status":"returned","backend_invoked":bool(result["backend_invoked"]),"time":time.time()})+"\n")
            atomic_json(target,record)
            atomic_json(output/("seed_%d_commit.json"%seed),{"status":"committed","sha256":file_sha(target),"identity":one_identity})
            atomic_json(state,{"status":"committed","identity":one_identity,"finished_unix":time.time()})
        except BaseException as error:
            atomic_json(state,{"status":"failed","identity":one_identity,"error":repr(error),"finished_unix":time.time()})
            raise
    atomic_json(output/"physical_calls.json",{"new_solver_calls":computed,"new_backend_invocations":backend_calls,
        "warmup_calls":0,"retry_calls":0,"timing_repeat_calls":0})


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--prediction",required=True);p.add_argument("--protocol",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();solve_file(a.prediction,a.protocol,a.output)


if __name__=="__main__":main()
