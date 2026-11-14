"""Resume core engineering only. This entry cannot dispatch formal training/evaluation."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


def write_json(path,value):
    tmp=path.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(value,f,sort_keys=True,indent=2);f.write("\n");f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def worker(cfg,run):
    # Install the parent-death guard before importing Torch or reading data.
    expected=int(os.environ["RELIABILITY_RECOVERY_PARENT"])
    if os.getppid()!=expected:raise RuntimeError("Supervisor disappeared")
    if ctypes.CDLL(None,use_errno=True).prctl(1,signal.SIGTERM,0,0,0)!=0:raise OSError(ctypes.get_errno(),"parent-death guard failed")
    if os.getppid()!=expected:raise RuntimeError("Supervisor disappeared")
    from reliability_v1.runtime import run_lock,atomic_json,identity,sha,Runtime
    from reliability_v1.smoke_manifest import build_smoke_manifest,SmokeDataset
    from reliability_v1.engineering import cpu_contracts,calibration,engineering
    with run_lock(run):
        atomic_json(run/"recovery_stage_tasks.json",dict(environment_probe="pass_with_bounded_scope",engineering_core="running",full_data_audit="incomplete",formal_training="not_started"))
        manifest=build_smoke_manifest(cfg,cfg["source_run"],run)
        path=run/"engineering_smoke_manifest.json"
        train=SmokeDataset(path,"train");val=SmokeDataset(path,"val")
        try:
            if len(train)!=8 or len(val)!=4:raise RuntimeError("Engineering requires exactly 8+4 fixed pairs")
            # The contract is current and hash-bound to the dedicated manifest. No old labels/statistics are loaded.
            atomic_json(run/"feature_label_contract.json",manifest["coordinate_contract"])
            atomic_json(run/"label_statistics_provenance.json",dict(
                coordinate_contract_sha256=manifest["coordinate_contract_sha256"],
                engineering_smoke_manifest_sha256=sha(path),
                scope="All labels and statistics are recomputed through corrected production coordinates",
                historical_label_statistics_reused=False))
            cpu_contracts(cfg,run)
            threshold=calibration(cfg,train,run)
            data_ident=dict(scope="engineering_smoke_only",formal_training_eligible=False,
                            engineering_smoke_manifest_sha256=sha(path),
                            texture_calibration_sha256=sha(run/"texture_calibration.json"))
            run_id=identity(cfg,data_ident)
            saved=run/"run_identity.json"
            if saved.exists() and json.loads(saved.read_text())!=run_id:
                raise RuntimeError("Recovery code/config/data identity changed; preserve prior evidence and use a new recovery directory")
            atomic_json(saved,run_id)
            runtime=Runtime(cfg,run);runtime.model.texture_threshold=threshold["threshold"]
            result=engineering(runtime,train,val,run_id,scope="engineering_core")
            atomic_json(run/"recovery_stage_tasks.json",dict(environment_probe="pass_with_bounded_scope",engineering_core="pass",full_data_audit="incomplete",formal_training="not_started"))
            print(json.dumps(dict(status=result["status"],scope=result["scope"],physical_updates=result["physical_optimizer_updates"],checkpoint=str(run/result["smoke_checkpoint"])),sort_keys=True),flush=True)
        finally:
            train.close();val.close()


def supervise(cfg,run,worker_args):
    from reliability_v1.environment_monitor import EnvironmentMonitor
    root=Path(cfg["repo_root"])
    logpath=run/"engineering_worker.log"
    monitor=None;process=None;status=None
    try:
        # A supervisor sees SIGBUS/kernel faults even when the worker cannot save evidence.
        monitor=EnvironmentMonitor(run,"engineering_guard",minimum_host_bytes=(cfg["minimum_free_gib"]+1)*2**30)
        with logpath.open("a") as log:
            env=dict(os.environ,RELIABILITY_RECOVERY_PARENT=str(os.getpid()),PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
            process=subprocess.Popen([sys.executable,str(root/"reliability_engineering_recovery.py"),*worker_args,"--worker"],stdout=log,stderr=subprocess.STDOUT,env=env)
            while process.poll() is None:
                monitor.check();time.sleep(2)
            status=monitor.check()
        if process.returncode==-signal.SIGBUS:
            status=monitor.record_failure("Engineering worker terminated by SIGBUS",dict(returncode=process.returncode,log=str(logpath)))
        if process.returncode:raise RuntimeError("Engineering worker failed; inspect "+str(logpath))
    except BaseException as exc:
        if process is not None and process.poll() is None:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        if monitor is not None:
            status=monitor.status
        else:
            status_file=run/"environment_probe"/"engineering_guard"/"status.json"
            status=json.loads(status_file.read_text()) if status_file.exists() else dict(status="blocked",reason=str(exc))
        returncode=process.returncode if process is not None else None
        signal_name=signal.Signals(-returncode).name if returncode is not None and returncode<0 else None
        failure_status="blocked" if status is None or status.get("status")!="pass" else "failed"
        value=dict(returncode=returncode,status=failure_status,log=str(logpath),
                   environment=status,formal_training="not_started",signal=signal_name,
                   exception_type=type(exc).__name__,exception=str(exc),traceback=traceback.format_exc())
        write_json(run/("engineering_execution_failure_"+str(time.time_ns())+".json"),value)
        write_json(run/"engineering_execution.json",value)
        write_json(run/"recovery_stage_tasks.json",dict(
            environment_probe="pass_with_bounded_scope" if status and status.get("status")=="pass" else "blocked",
            engineering_core=failure_status,full_data_audit="incomplete",formal_training="not_started"))
        raise
    value=dict(returncode=process.returncode,status="pass",log=str(logpath),environment=status,
               formal_training="not_started",signal=None)
    write_json(run/"engineering_execution.json",value)
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/reliability_v1.json")
    parser.add_argument("--source-run",required=True)
    parser.add_argument("--run-dir",required=True)
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    root=Path(cfg["repo_root"]).resolve();source=Path(args.source_run).resolve();run=Path(args.run_dir).resolve()
    if not source.is_relative_to(root/"runs") or not run.is_relative_to(root/"runs") or run==source:raise ValueError("Distinct recovery run required inside repository runs/")
    if (cfg["resolution"],cfg["batch_size"],cfg["num_workers"],cfg["engineering_pairs"],cfg["engineering_val_pairs"])!=(832,1,0,8,4):raise ValueError("Original registered numerical and engineering configuration required")
    cfg.update(source_run=str(source),run_dir=str(run),run_stage="engineering_core",
               training_protocol=dict(status="pending_review",note="Native 100 replacement draws per index vs full-manifest visits remains unresolved"),
               acceptance_scope="engineering_smoke_only")
    os.chdir(root);sys.dont_write_bytecode=True;run.mkdir(parents=True,exist_ok=True)
    if args.worker:
        worker(cfg,run)
        return
    write_json(run/"recovery_resolved_config.json",cfg)
    write_json(run/"recovery_command.json",dict(argv=[sys.executable,*sys.argv],cwd=str(root),source_run=str(source),stages_allowed=["engineering_core"],formal_training="not_started"))
    value=supervise(cfg,run,sys.argv[1:])
    print(json.dumps(value,sort_keys=True),flush=True)


if __name__=="__main__":
    try:main()
    except BaseException:
        traceback.print_exc()
        sys.exit(1)
