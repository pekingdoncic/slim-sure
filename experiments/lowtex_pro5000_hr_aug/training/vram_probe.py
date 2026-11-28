"""Candidate trials are isolated child processes; only their measured configuration is inherited."""
import json
import os
import subprocess
import sys
from pathlib import Path
from ..utils.run_io import PKG,read,write,now


def dtype_failure(message):
    text = message.lower()
    return any(x in text for x in ("bfloat16","bf16","unsupported dtype","expected scalar type","not implemented for","input type"))


def probe_configs(args,cfg,job):
    directory = Path(job["directory"])
    directory.mkdir(parents=True,exist_ok=True)
    trials = []
    success = []
    candidates = cfg["micro_batch_candidates"]
    precision = job["precision"]
    strategies = ("full","selective","off") if cfg["gradient_checkpointing"]=="auto" else (cfg["gradient_checkpointing"],)
    for ck in strategies:
        corr,graph = cfg["correlation_chunk"],cfg["graph_chunk_size"]
        for micro in candidates:
            if cfg["effective_batch_size"]%micro:
                continue
            while True:
                root = directory/f"{ck}_b{micro}_{precision}_c{corr}_g{graph}"
                spec = dict(job,directory=str(root),start=0,end=3,total=3,micro=micro,
                            checkpointing=ck,precision=precision,action="trial",correlation_chunk=corr,graph_chunk_size=graph)
                path = root/"job.json"
                cached = root/"trial_result.json"
                if cached.is_file() and read(path)==spec:
                    row = read(cached)
                else:
                    write(path,spec)
                    cmd = [sys.executable,"-m",PKG+".training.worker",str(path)]
                    proc = subprocess.run(cmd)
                    if not cached.is_file():
                        raise RuntimeError(f"probe trial exited {proc.returncode} without result: {cmd}")
                    row = read(cached)
                row.update(correlation_chunk=corr,graph_chunk_size=graph)
                trials.append(row)
                write(directory/"trials.json",trials)
                if row["status"]=="ERROR" and precision=="bfloat16" and dtype_failure(row.get("error","")):
                    write(directory/"bf16_trials.json",trials)
                    write(directory/"precision_fallback.json",dict(from_precision=precision,to_precision="float16",reason=row["error"]))
                    print(f"BF16 kernel rejected; retry FP16 with GradScaler: {row['error']}",flush=True)
                    return probe_configs(args,cfg,dict(job,precision="float16"))
                if row["status"] in ("OOM","OVER_BUDGET") and cfg["bbr"] and micro==1 and (corr>2048 or graph>256):
                    corr,graph=max(2048,corr//2),max(256,graph//2)
                    continue
                break
            if row["status"]=="ERROR":
                raise RuntimeError(row.get("error","probe trial failed"))
            if row["status"] in ("OOM","OVER_BUDGET"):
                break
            if row["status"]!="PASS":
                raise RuntimeError(f"unexpected probe result {row}")
            success.append(row)
    # Compare like precision if an unsupported BF16 kernel forced fallback.
    success = [v for v in success if v["precision"]==precision]
    if not success:
        raise RuntimeError(f"no safe unfrozen training configuration at R{job['side']}; see {directory}")
    best = max(success,key=lambda r:r["pairs_per_s"])
    write(directory/"selected.json",dict(**best,selection="highest measured pairs/s within target",time=now()))
    return best
