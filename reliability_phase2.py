"""Explicit second-task stages. --dry-run has no output-file or process side effects."""
import argparse,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
def dry_plan(config,config_path,stage,resume):
    root=Path(config["repo_root"]).resolve();run=Path(config["run_dir"]).resolve()
    stages=["full_train","final_epe","report"] if stage=="pipeline" else [stage]
    commands=[]
    for s in stages:
        command=[sys.executable,str(root/"reliability_phase2.py"),s,"--config",str(config_path)]
        if resume and s!="report":command.append("--resume")
        commands.append(command)
    return dict(dry_run=True,repo_root=str(root),run_dir=str(run),stages=stages,commands=commands,
                output_paths={s:str(run/{"full_train":"formal_train","final_epe":"final_epe","report":"phase2_report.json"}[s]) for s in stages},
                resume_commands={s:[sys.executable,str(root/"reliability_phase2.py"),s,"--config",str(config_path),"--resume"] for s in stages if s!="report"},
                formal_optimizer_updates=0,real_final_epe_executed=False)
def main():
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["full_train","final_epe","report","pipeline"]);p.add_argument("--config",required=True)
    p.add_argument("--resume",action="store_true");p.add_argument("--dry-run",action="store_true");p.add_argument("--worker",action="store_true")
    a=p.parse_args();path=Path(a.config).resolve();cfg=json.loads(path.read_text());run=Path(cfg["run_dir"]).resolve();root=Path(cfg["repo_root"]).resolve()
    if a.dry_run:print(json.dumps(dry_plan(cfg,path,a.stage,a.resume),indent=2));return
    os.chdir(root)
    if a.worker:
        if a.stage=="pipeline":raise ValueError("Pipeline cannot be a stage worker")
        from reliability_v1.phase2_runtime import train,final_epe,report
        result={"full_train":train,"final_epe":final_epe,"report":lambda c,r,resume:report(c,r)}[a.stage](cfg,run,resume=a.resume)
        print(json.dumps(dict(stage=a.stage,status="committed"),sort_keys=True));return
    from reliability_v1.environment_monitor import EnvironmentMonitor
    from reliability_v1.runtime import atomic_json,run_lock
    stages=["full_train","final_epe","report"] if a.stage=="pipeline" else [a.stage]
    with run_lock(run):
        for stage in stages:
            completed=run/({"full_train":"formal_train/status.json","final_epe":"final_epe/state.json","report":"phase2_report.json"}[stage])
            if a.resume and completed.exists() and json.loads(completed.read_text()).get("status")=="committed":continue
            guard=EnvironmentMonitor(run,"formal_"+stage,3*2**30)
            command=[sys.executable,"-u",str(root/"reliability_phase2.py"),stage,"--config",str(path),"--worker"]
            if a.resume and stage!="report":command.append("--resume")
            with (run/(stage+".log")).open("a") as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                atomic_json(run/"launch_state.json",dict(status="running",stage=stage,pid=child.pid,supervisor_pid=os.getpid(),command=command,time=time.time()))
                started=time.monotonic()
                try:
                    while child.poll() is None:
                        env=guard.check()
                        progress_files=[run/"formal_train/heartbeat.json",run/"final_epe/state.json"]
                        progress_files+=list((run/"formal_train").glob("lr_epoch_*.progress.json"))
                        progress_files+=list((run/"formal_train").glob("validation_*/state.json"))
                        latest=max([p.stat().st_mtime for p in progress_files if p.exists()]+[time.time()-(time.monotonic()-started)])
                        if time.time()-latest>cfg.get("progress_watchdog_seconds",900):
                            raise RuntimeError("Stage progress watchdog exceeded configured limit")
                        atomic_json(run/"launch_heartbeat.json",dict(stage=stage,pid=child.pid,time=time.time(),elapsed_seconds=time.monotonic()-started,resources=env["latest"]))
                        time.sleep(2)
                    guard.check()
                    if child.returncode:raise RuntimeError("Stage worker returncode "+str(child.returncode))
                except BaseException as e:
                    if child.poll() is None:
                        os.killpg(child.pid,signal.SIGTERM)
                        try:child.wait(10)
                        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    atomic_json(run/"launch_state.json",dict(status="blocked",stage=stage,reason=repr(e),last_valid_checkpoint=str(run/"formal_train/last.json"),
                        resume_command=[sys.executable,str(root/"reliability_phase2.py"),stage,"--config",str(path),"--resume"],
                        pipeline_resume=[sys.executable,str(root/"reliability_phase2.py"),"pipeline","--config",str(path),"--resume"]))
                    raise
                atomic_json(run/"launch_state.json",dict(status="committed",stage=stage,returncode=child.returncode))
if __name__=="__main__":
    try:main()
    except BaseException:traceback.print_exc();sys.exit(1)
