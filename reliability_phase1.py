"""Forward-only Phase 1 preparation, supervised and independently restartable."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
from reliability_v1.runtime import atomic_json,run_lock
from reliability_v1.environment_monitor import EnvironmentMonitor

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--phase",choices=["freeze","statistics"],required=True);p.add_argument("--worker",action="store_true")
    a=p.parse_args();cfg=json.loads(Path(a.config).read_text());run=Path(cfg["run_dir"]);root=Path(cfg["repo_root"])
    if a.worker:
        from reliability_v1.phase1_prepare import verify_unique_files,freeze_sampling,real_statistics
        if a.phase=="freeze":
            verify_unique_files(run);result=freeze_sampling(cfg,run)
        else:result=real_statistics(cfg,run)
        atomic_json(run/(a.phase+"_result.json"),result);return
    out=run/("prepare_"+a.phase);out.mkdir(exist_ok=True)
    if not (run/"complete_audit/execution.json").exists() or json.loads((run/"complete_audit/execution.json").read_text())["status"]!="pass":
        raise RuntimeError("Audit supervisor must finish successfully before preparation")
    with run_lock(run),(out/"worker.log").open("a") as log:
        guard=EnvironmentMonitor(run,"prepare_"+a.phase,3*2**30)
        argv=[sys.executable,"-u",str(root/"reliability_phase1.py"),"--config",str(Path(a.config).resolve()),"--phase",a.phase,"--worker"]
        child=subprocess.Popen(argv,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1"))
        atomic_json(out/"process.json",dict(parent_pid=os.getpid(),worker_pid=child.pid,argv=argv,formal_optimizer_updates=0))
        start=time.time()
        try:
            while child.poll() is None:
                env=guard.check()
                latest=max([start]+[q.stat().st_mtime for q in [run/"prepare_heartbeat.json",run/"heartbeat.json"] if q.exists()])
                if time.time()-latest>900:raise RuntimeError("Preparation progress watchdog exceeded 900 seconds")
                atomic_json(out/"heartbeat.json",dict(time=time.time(),seconds=time.time()-start,resources=env["latest"]))
                time.sleep(2)
            env=guard.check()
            if child.returncode:raise RuntimeError("Preparation worker returncode "+str(child.returncode))
            atomic_json(out/"execution.json",dict(status="pass",seconds=time.time()-start,environment=env,formal_optimizer_updates=0))
        except BaseException as e:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            atomic_json(out/"failure.json",dict(status="blocked",reason=repr(e),returncode=child.returncode,argv=argv))
            raise
if __name__=="__main__":main()
