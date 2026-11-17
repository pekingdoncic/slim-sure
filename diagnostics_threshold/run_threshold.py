"""Background serial threshold supervisor with kernel-cursor watchdog."""
from shared import *
import argparse,signal,traceback
from reliability_v1.runtime import run_lock
def guard(run,stage):
    sys.path.insert(0,str(ROOT/'diagnostics_phase3'))
    from runner import guard_for
    return guard_for(run,stage)
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--background',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--worker',choices=['S0','S1','S2','S3','S4','S5']);p.add_argument('--limit',type=int);a=p.parse_args();run=Path(a.run)
    if a.background:
        f=run/'supervisor.pid'
        if f.exists() and (Path('/proc')/f.read_text().strip()/'cmdline').exists():raise RuntimeError('Existing supervisor PID; inspect before launching')
        cmd=['nohup',sys.executable,'-B','-u',str(Path(__file__).resolve()),'--run',str(run)]+(['--resume'] if a.resume else [])
        with (run/'supervisor.log').open('ab') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        text(f,str(child.pid)+'\n');save(run/'launch.json',dict(command=cmd,pid=child.pid,started=now(),log=str(run/'supervisor.log')))
        print(json.dumps(read(run/'launch.json')));return
    if a.worker:
        from extract import run_setting
        try:run_setting(run,a.worker,a.limit)
        except BaseException as e:
            import torch
            status='oom' if isinstance(e,torch.cuda.OutOfMemoryError) else 's0_not_reproduced' if isinstance(e,StopRoute) and a.worker=='S0' else 'failed'
            save(run/a.worker/'failure.json',dict(status=status,reason=repr(e),traceback=traceback.format_exc(),time_beijing=now(),pid=os.getpid(),resources=resource_sample()))
            raise
        return
    identity_gate()
    with run_lock(run):
        g=guard(run,'T_supervisor');started=time.monotonic()
        for setting in ['S0','S1','S2','S3','S4','S5']:
            out=run/setting;out.mkdir(exist_ok=True)
            if (out/'status.json').exists() and read(out/'status.json')['status'] in ['completed','oom','skipped']:continue
            if setting=='S1' and (run/'degradation.json').exists() and 'T S1' in read(run/'degradation.json')['skipped']:
                save(out/'status.json',dict(status='skipped',reason='Preregistered >16h degradation',time_beijing=now()));continue
            cmd=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--run',str(run),'--worker',setting]
            with (out/'worker.log').open('ab') as log:
                child=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
                save(out/'launch.json',dict(command=cmd,pid=child.pid,started=now()))
                save(run/'stage_status.json',dict(status='running',stage=setting,supervisor_pid=os.getpid(),worker_pid=child.pid,time_beijing=now()))
                try:
                    while child.poll() is None:
                        g.check();save(run/'heartbeat.json',dict(status='running',setting=setting,pid=os.getpid(),worker_pid=child.pid,unix=time.time(),time_beijing=now(),resources=g.last_sample))
                        h=out/'heartbeat.json'
                        if h.exists() and time.time()-h.stat().st_mtime>900:raise RuntimeError('Worker no progress for 900s')
                        time.sleep(5)
                    g.check()
                except BaseException as e:
                    if child.poll() is None:
                        os.killpg(child.pid,signal.SIGTERM)
                        try:child.wait(10)
                        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    save(run/'stage_status.json',dict(status='blocked',stage=setting,reason=repr(e),stop_entire_chain=True,time_beijing=now()));raise
            fail=read(out/'failure.json') if (out/'failure.json').exists() else {}
            if child.returncode:
                save(out/'status.json',dict(status=fail.get('status','failed'),returncode=child.returncode,failure=fail,time_beijing=now()))
                if fail.get('status')=='oom':continue
                if fail.get('status')=='s0_not_reproduced':
                    save(run/'stage_status.json',dict(status='partial',stage='S0',reason='S0 not reproduced; T stopped; Phase5 independent E0 required',time_beijing=now()));return
                save(run/'stage_status.json',dict(status='failed',stage=setting,reason=fail,time_beijing=now()));return
            save(out/'status.json',dict(status='completed',worker=read(out/'worker_status.json'),time_beijing=now()))
        save(run/'stage_status.json',dict(status='scan_completed',report='pending',seconds=time.monotonic()-started,time_beijing=now()))
if __name__=='__main__':main()
