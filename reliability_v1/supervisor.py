"""Observable subprocess watchdog; no total training-duration cutoff."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

def supervise(argv,config):
    run=Path(config["run_dir"]);run.mkdir(parents=True,exist_ok=True)
    launch=run/"launches";launch.mkdir(exist_ok=True)
    attempt=0
    active={"process":None}
    def mark_failure(reason,returncode=None):
        tasks_path=run/"stage_tasks.json"
        if tasks_path.exists():
            tasks=json.loads(tasks_path.read_text())
            for key in tasks:
                if tasks[key]=="running":tasks[key]="failed"
            tmp=tasks_path.with_suffix(".supervisor.tmp")
            tmp.write_text(json.dumps(tasks,indent=2));os.replace(tmp,tasks_path)
        path=launch/("failure_"+uuid.uuid4().hex[:8]+".json")
        path.write_text(json.dumps(dict(status="failed",reason=reason,returncode=returncode,time_unix=time.time()),indent=2))
    def interrupt(signum,frame):
        proc=active["process"]
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        (launch/("interrupted_"+uuid.uuid4().hex[:8]+".json")).write_text(json.dumps(dict(status="failed",reason="supervisor_interrupted",signal=signum,pid=proc.pid if proc else None)))
        mark_failure("supervisor_interrupted",128+signum)
        raise SystemExit(128+signum)
    signal.signal(signal.SIGINT,interrupt);signal.signal(signal.SIGTERM,interrupt)
    while True:
        tag=time.strftime("%Y%m%dT%H%M%S")+"_"+uuid.uuid4().hex[:6]
        logfile=launch/(tag+".log");started=time.time()
        command=[sys.executable,str(Path(config["repo_root"])/"reliability_stage.py"),*argv,"--worker"]
        with logfile.open("x") as log:
            proc=subprocess.Popen(command,cwd=config["repo_root"],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,RELIABILITY_SUPERVISOR_PID=str(os.getpid())))
            active["process"]=proc
            print(json.dumps(dict(event="launched",pid=proc.pid,log=str(logfile),command=command)),flush=True)
            (launch/(tag+".json")).write_text(json.dumps(dict(pid=proc.pid,command=command,start_unix=started,status="running"),indent=2))
            terminated=False
            while proc.poll() is None:
                paths=[run/"heartbeat.json",run/"data_audit_progress.json",run/"stage_tasks.json",
                       run/"evaluation/pilot/state.json",run/"evaluation/costs/state.json"]
                paths+=list((run/"full_train").glob("validation_*/state.json"))
                paths+=list((run/"engineering").glob("validation_*/state.json"))
                paths+=list((run/"evaluation/exports").glob("*/attempt_*/heartbeat.json"))
                paths+=list((run/"evaluation/exports").glob("*/runtime/heartbeat.json"))
                activity=max([started]+[p.stat().st_mtime for p in paths if p.exists()])
                limits=json.loads((run/"watchdog.json").read_text()) if (run/"watchdog.json").exists() else {}
                timeout=max(float(limits.get("pair_timeout_seconds",600)),float(limits.get("cold_load_timeout_seconds",600)))
                # No whole-task time cap: only absence of recorded forward/audit/validation progress.
                if time.time()-activity>timeout:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                    terminated=True
                    (launch/(tag+".watchdog.json")).write_text(json.dumps(dict(status="failed",reason="no_progress",timeout_seconds=timeout,last_activity_unix=activity),indent=2))
                    print("Stopped stalled worker; committed checkpoints remain resumable.",flush=True)
                    break
                time.sleep(1)
        code=proc.returncode
        (launch/(tag+".json")).write_text(json.dumps(dict(pid=proc.pid,command=command,start_unix=started,end_unix=time.time(),status="committed" if code==0 else "failed",returncode=code),indent=2))
        print(logfile.read_text()[-12000:],flush=True)
        if code==0:return 0
        # Never replay engineering automatically: preserve its physical update budget.
        formal=("--run-stage" in argv and argv[argv.index("--run-stage")+1]=="full_train") or ("--run_stage" in argv and argv[argv.index("--run_stage")+1]=="full_train")
        text=logfile.read_text().lower()
        unsafe=terminated or any(x in text for x in ("out of memory","nonfinite","nan","resource","disk","allocation growth")) or code in (-9,137)
        transient=code in (-11,139) or any(x in text for x in ("connection reset","temporarily unavailable","interrupted system call"))
        if formal and not unsafe and transient and attempt==0:
            attempt+=1
            if (run/"full_train/last.json").exists() and "--resume" not in argv:argv=[*argv,"--resume"]
            print("One recorded transient-error recovery using the same configuration.",flush=True)
            continue
        mark_failure("worker_failed",code)
        return code or 1
