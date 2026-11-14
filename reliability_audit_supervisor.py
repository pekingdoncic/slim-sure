"""Supervise bounded audit phases in an independent run; historical data stays read-only."""
import argparse,json,os,signal,subprocess,sys,time,fcntl
from pathlib import Path
from reliability_v1.environment_monitor import EnvironmentMonitor
def save(path,value):
    tmp=path.with_suffix(".tmp")
    with tmp.open("w") as f:json.dump(value,f,indent=2,sort_keys=True);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--pilot",action="store_true")
    a=p.parse_args();cfg=json.loads(Path(a.config).read_text());run=Path(cfg["run_dir"]);phase="large_index_pilot" if a.pilot else "complete_audit"
    out=run/phase;out.mkdir(exist_ok=True)
    guard=EnvironmentMonitor(run,phase,3*2**30)
    argv=[sys.executable,"-u","reliability_data.py","--config",str(Path(a.config).resolve()),"--run-dir",str(run),"--resume-audit","--audit-evidence-dir",str(out)]
    if a.pilot:argv+=["--max-new-indices","1"]
    with (run/"writer.lock").open("a+") as lock,(out/"worker.log").open("a") as log:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        child=subprocess.Popen(argv,cwd=cfg["repo_root"],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1"))
        started=time.monotonic();last_progress=started;last_mtime=0
        save(out/"process.json",dict(parent_pid=os.getpid(),worker_pid=child.pid,argv=argv,phase=phase,formal_optimizer_updates=0))
        try:
            while child.poll() is None:
                state=guard.check()
                progress=out/"data_audit_progress.json"
                if progress.exists() and progress.stat().st_mtime_ns!=last_mtime:
                    last_mtime=progress.stat().st_mtime_ns;last_progress=time.monotonic()
                save(out/"heartbeat.json",dict(time=time.time(),pid=child.pid,phase=phase,seconds=time.monotonic()-started,last_progress_seconds_ago=time.monotonic()-last_progress,environment=state["latest"]))
                # Source identity verification and a slow index may take minutes.
                if time.monotonic()-last_progress>900:raise RuntimeError("Audit progress watchdog exceeded 900 seconds")
                time.sleep(2)
            state=guard.check()
            save(out/"execution.json",dict(status="pass" if child.returncode==0 else "blocked",returncode=child.returncode,seconds=time.monotonic()-started,environment=state,formal_optimizer_updates=0))
            if child.returncode:raise RuntimeError("Audit worker failed; inspect "+str(out/"worker.log"))
        except BaseException as e:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            save(out/"failure.json",dict(status="blocked",reason=repr(e),returncode=child.returncode,phase=phase,progress=str(out/"data_audit_progress.json"),resume_argv=argv))
            raise
if __name__=="__main__":main()
