"""Serial one-pair export processes and immutable CPU prediction transport."""
from pathlib import Path
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np

from .geometry_worker import atomic_json,digest,file_sha
from .evaluation import writer_lock,freeze_pilot,_atomic_npz


MODES=("A0","A_score","A_texture")

def terminate_worker_group(process):
    """Only called for a child started with start_new_session=True."""
    if process.poll() is not None:return
    if process.pid<=0 or process.pid==os.getpgrp():
        raise RuntimeError("Refusing to signal the controller process group")
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        process.wait(timeout=10)




def json_value(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):json_value(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_value(v) for v in value]
    return value


def save_prediction(directory,prediction,identity):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    arrays={};metadata={}
    for key,value in prediction.items():
        if isinstance(value,np.ndarray):
            if value.dtype.hasobject:raise TypeError("Object arrays are forbidden in prediction transport")
            arrays[key]=value
        else:metadata[key]=json_value(value)
    _atomic_npz(directory/"arrays.npz",arrays)
    atomic_json(directory/"metadata.json",metadata)
    atomic_json(directory/"commit.json",{"status":"committed","identity":identity,
        "sha256":{name:file_sha(directory/name) for name in ("arrays.npz","metadata.json")}})


def load_prediction(directory,identity):
    directory=Path(directory);commit=json.loads((directory/"commit.json").read_text())
    if commit["status"]!="committed" or commit["identity"]!=identity:
        raise ValueError("Export cache identity differs")
    for name,expected in commit["sha256"].items():
        if file_sha(directory/name)!=expected:raise ValueError("Export cache checksum failed")
    result=json.loads((directory/"metadata.json").read_text())
    with np.load(directory/"arrays.npz",allow_pickle=False) as z:
        result.update({key:z[key] for key in z.files})
    result["executed_forward"]=False
    return result


def worker_identity(config_path,run_identity,checkpoint,calibration,record,purpose):
    pointer=Path(checkpoint);meta=json.loads(pointer.read_text())
    payload=pointer.parent/meta["payload"]
    if payload.parent.resolve()!=pointer.parent.resolve() or file_sha(payload)!=meta["sha256"]:
        raise ValueError("Formal checkpoint checksum or path mismatch")
    if meta["identity_sha"]!=digest(run_identity):raise ValueError("Formal checkpoint has a different accepted identity")
    return {"schema":"slim_reliability_isolated_export_v1","run_identity":run_identity,
        "config_file_sha256":file_sha(config_path),"checkpoint_pointer_sha256":file_sha(pointer),
        "checkpoint_payload_sha256":meta["sha256"],"texture_calibration_sha256":file_sha(calibration),
        "record_sha256":digest(json_value(record)),"pair_id":str(record["pair_id"]),
        "purpose":purpose,"maximum_unique_pairs_per_worker":1}


class FilePredictor:
    """One unique pair per worker; A0/score/texture are serial and atomically cached."""
    def __init__(self,config_path,run_identity,out_dir,checkpoint,
                 python_executable="/home/bupt/miniconda3/envs/slim/bin/python"):
        self.config_path=Path(config_path).resolve()
        self.config=json.loads(self.config_path.read_text())
        self.run_identity=run_identity;self.out=Path(out_dir).resolve()
        self.checkpoint=Path(checkpoint).resolve();self.python=str(python_executable)
        self.calibration=Path(self.config["run_dir"])/"texture_calibration.json"
        watchdog=Path(self.config["run_dir"])/"watchdog.json"
        timing=json.loads(watchdog.read_text())
        self.pair_timeout=float(timing["pair_timeout_seconds"])
        self.cold_timeout=float(timing["cold_load_timeout_seconds"])
        if self.pair_timeout<=0 or self.cold_timeout<=0 or not timing.get("measured"):
            raise ValueError("Export watchdog needs measured engineering timing evidence")
        self.watchdog_evidence={"path":str(watchdog),"sha256":file_sha(watchdog),"settings":timing}
        self.out.mkdir(parents=True,exist_ok=True)
        base={"run_identity":run_identity,"config_file_sha256":file_sha(self.config_path),
            "checkpoint":str(self.checkpoint),"checkpoint_pointer_sha256":file_sha(self.checkpoint),
            "calibration_sha256":file_sha(self.calibration),"watchdog":self.watchdog_evidence,
            "maximum_unique_pairs_per_worker":1,"gpu_concurrency":1}
        target=self.out/"identity.json"
        if target.exists() and json.loads(target.read_text())!=base:raise ValueError("Export controller identity changed")
        atomic_json(target,base)
        self.successes=0;self.smoke_pairs=[]
        state=self.out/"controller_state.json"
        if state.exists():
            previous=json.loads(state.read_text())
            self.successes=int(previous.get("successful_export_workers",0))
            self.smoke_pairs=previous.get("smoke_pair_ids",[])[:3]

    def _launch(self,record,directory,identity,purpose):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        request={"identity":identity,"config_path":str(self.config_path),"checkpoint":str(self.checkpoint),
                 "calibration":str(self.calibration),"record":json_value(record),
                 "run_identity":self.run_identity,"purpose":purpose,"output":str(directory),"controller_root":str(self.out)}
        request_path=directory/"request.json";atomic_json(request_path,request)
        final=directory/"commit.json"
        if final.exists():
            saved=json.loads(final.read_text())
            if saved["identity"]!=identity:raise ValueError("Worker commit identity differs")
            for name,expected in saved["sha256"].items():
                if file_sha(directory/name)!=expected:raise ValueError("Worker artifact checksum mismatch")
            return saved
        attempts_path=directory/"attempts.json"
        attempts=json.loads(attempts_path.read_text()) if attempts_path.exists() else []
        if len(attempts)>=2:raise RuntimeError("At most one same-configuration transient retry is allowed")
        # Resume only the explicitly allowed retry class, never resource/numeric failures.
        if attempts and not attempts[-1].get("retryable",False):
            raise RuntimeError("Previous export failure requires diagnosis; automatic retry is disabled")
        for number in range(len(attempts),2):
            attempt_dir=directory/("attempt_%d"%number);attempt_dir.mkdir(parents=True,exist_ok=True)
            heartbeat=attempt_dir/"heartbeat.json"
            command=[self.python,"-m","reliability_v1.export_worker","--request",str(request_path),
                     "--attempt-dir",str(attempt_dir)]
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",
                     OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",
                     SLIM_EXPORT_PARENT_PID=str(os.getpid()))
            started=time.time();last_progress=started;token=None;phase="cold_load";timeout=False
            log_path=attempt_dir/"worker.log"
            process=None
            previous_sigterm=None
            signal_handler_installed=False
            def interrupted_by_sigterm(signum,frame):
                raise KeyboardInterrupt("Export controller received SIGTERM")
            try:
                # Export runs on the CLI main thread; retain the old handler on exit.
                try:
                    previous_sigterm=signal.getsignal(signal.SIGTERM)
                    signal.signal(signal.SIGTERM,interrupted_by_sigterm)
                    signal_handler_installed=True
                except ValueError:
                    # Non-main-thread callers still get finally cleanup on exceptions.
                    pass
                with log_path.open("a") as log:
                    process=subprocess.Popen(command,cwd=self.config["repo_root"],env=env,
                                             stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    atomic_json(attempt_dir/"process.json",{"pid":process.pid,"process_group":process.pid,
                        "command":command,"started":started,"parent_pid":os.getpid()})
                    while process.poll() is None:
                        if heartbeat.exists():
                            status=json.loads(heartbeat.read_text())
                            current=status.get("token")
                            if current!=token:
                                token=current;last_progress=time.time();phase=status.get("phase","inference")
                        limit=self.cold_timeout if phase in ("cold_load","initializing") else self.pair_timeout
                        if time.time()-last_progress>limit:
                            timeout=True;terminate_worker_group(process)
                            break
                        time.sleep(.25)
                    code=process.wait()
            except BaseException as interruption:
                # Propagate user/supervisor interruptions; never turn them into a retry.
                atomic_json(attempt_dir/"interruptions"/(str(time.time_ns())+".json"),
                    {"status":"interrupted","error":repr(interruption),"time":time.time(),
                     "child_pid":process.pid if process is not None else None,
                     "phase":phase,"last_progress_token":token,"automatic_retry":False})
                raise
            finally:
                try:
                    if process is not None and process.poll() is None:
                        terminate_worker_group(process)
                finally:
                    if signal_handler_installed:
                        signal.signal(signal.SIGTERM,previous_sigterm)
            failure_path=attempt_dir/"failure.json"
            failure=json.loads(failure_path.read_text()) if failure_path.exists() else None
            tail=log_path.read_text(errors="replace")[-12000:].lower()
            forbidden=any(word in tail for word in ("out of memory","nonfinite","nan loss","inf loss","illegal memory","device-side assert"))
            retryable=bool(not timeout and not forbidden and code not in (-9,-15)
                           and (failure.get("retryable",False) if failure is not None else code!=0))
            attempt={"attempt":number,"status":"committed" if code==0 and final.exists() else "failed",
                "returncode":code,"timeout":timeout,"retryable":retryable,"wall_seconds":time.time()-started,
                "log":str(log_path),"failure":failure,"unique_pairs":1,"purpose":purpose}
            attempts.append(attempt);atomic_json(attempts_path,attempts)
            if attempt["status"]=="committed":
                committed=json.loads(final.read_text())
                if committed["identity"]!=identity:raise ValueError("Worker committed wrong identity")
                for name,expected in committed["sha256"].items():
                    if file_sha(directory/name)!=expected:raise ValueError("Worker output checksum failed")
                if purpose=="export":
                    self.successes+=1
                    if len(self.smoke_pairs)<3 and str(record["pair_id"]) not in self.smoke_pairs:
                        self.smoke_pairs.append(str(record["pair_id"]))
                atomic_json(self.out/"controller_state.json",{"status":"running",
                    "successful_export_workers":self.successes,"three_pair_smoke":"pass" if len(self.smoke_pairs)>=3 else "running",
                    "smoke_pair_ids":self.smoke_pairs,
                    "maximum_unique_pairs_per_worker":1,"gpu_concurrency":1,
                    "last_pair_id":record["pair_id"],"last_purpose":purpose,"last_finished":time.time()})
                return committed
            if not retryable:break
        raise RuntimeError("Isolated export worker failed; see "+str(attempts_path))

    def __call__(self,record,mode="A_score"):
        if mode not in MODES:raise ValueError(mode)
        identity=worker_identity(self.config_path,self.run_identity,self.checkpoint,self.calibration,record,"export")
        directory=self.out/"pairs"/digest({"pair_id":str(record["pair_id"])})[:24]
        with writer_lock(self.out):
            self._launch(record,directory,identity,"export")
            return load_prediction(directory/mode,dict(identity,mode=mode))

    def measure_costs(self,records,out_dir):
        chosen=freeze_pilot(records,10);out=Path(out_dir);rows=[];cold=[]
        with writer_lock(self.out),writer_lock(out):
            atomic_json(out/"timing_manifest.json",json_value(chosen))
            for record in chosen:
                identity=worker_identity(self.config_path,self.run_identity,self.checkpoint,self.calibration,record,"timing_only")
                directory=self.out/"timing"/digest({"pair_id":str(record["pair_id"])})[:24]
                self._launch(record,directory,identity,"timing_only")
                timing=json.loads((directory/"timing_records.json").read_text())
                rows.extend(timing)
                cold.append(json.loads((directory/"worker_summary.json").read_text())["cold_load_seconds"])
            modes={}
            for mode in MODES:
                sample=[r["seconds"] for r in rows if r["mode"]==mode and r["purpose"]=="steady_repeat"]
                modes[mode]={"count":len(sample),"mean_seconds":float(np.mean(sample)),
                    "median_seconds":float(np.median(sample)),"p95_seconds":float(np.quantile(sample,.95))}
            result={"status":"committed","pair_count":len(chosen),"warmup_forwards":len(chosen)*9,
                "timing_forwards":len(chosen)*30,"geometry_calls":0,"modes":modes,
                "cold_load_seconds":cold,"unique_pairs_per_worker":1,"gpu_concurrency":1,
                "scope":"Image-only production forward plus file I/O; no GT depth loading, labels or projection diagnostics. Each pair has a fresh resident worker. Not a deployment p95.",
                "physical_retry_details":str(self.out/"timing")}
            atomic_json(out/"timing_records.json",rows);atomic_json(out/"timing_summary.json",result)
            atomic_json(out/"state.json",result)
            return result
