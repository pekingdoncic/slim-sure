"""Background task supervisor; current-boot checks continue a fully reviewed cursor."""
from pathlib import Path
import os,sys,argparse,subprocess,time,signal,json,traceback,importlib,shutil
sys.dont_write_bytecode=True
from common import *
from reliability_v1.environment_monitor import EnvironmentMonitor
from reliability_v1.phase2_runtime import make_identity
from reliability_v1.runtime import run_lock,resource_sample
def guard_for(run,stage):
    candidates=[Path(run)/'initial_environment.json']
    candidates += list((Path(run)/'environment_probe').glob('*/status.json')) if (Path(run)/'environment_probe').exists() else []
    entries=[(p,read(p)) for p in candidates]
    entries=[(p,x) for p,x in entries if x.get('status')=='pass' and x.get('boot_id')==Path('/proc/sys/kernel/random/boot_id').read_text().strip()]
    p,seed=max(entries,key=lambda z:z[0].stat().st_mtime_ns)
    cursor=seed.get('journal_cursor',seed.get('cursor'))
    if not cursor:raise RuntimeError('No reviewed kernel cursor')
    g=EnvironmentMonitor.__new__(EnvironmentMonitor);g.run=Path(run);g.out=Path(run)/'environment_probe'/stage;g.out.mkdir(parents=True,exist_ok=True)
    g.minimum_host_bytes=2*2**30;g.boot=seed['boot_id'];g.cursor=cursor;g.started=time.monotonic();g.checks=0;g.last_sample=None;g.blocked_reason=None;g.status=None
    save(g.out/'seed.json',dict(path=str(p),sha256=sha(p),cursor=cursor,reason='Continue from streamed initial current-boot review / prior passing task; unchanged fault checks'))
    g.check(False);return g
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--task',required=True,choices=['A','B','C','A4','report']);p.add_argument('--background',action='store_true');p.add_argument('--worker',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args();run=Path(a.run);stage=a.task;out=run/stage;out.mkdir(exist_ok=True)
    if a.background:
        pidfile=out/'supervisor.pid'
        if pidfile.exists():
            old=int(pidfile.read_text());proc=Path('/proc')/str(old)/'cmdline'
            if proc.exists() and str(run).encode() in proc.read_bytes():raise RuntimeError('Task already running')
        if (out/'status.json').exists() and read(out/'status.json').get('status')=='completed':raise RuntimeError('Task already completed')
        cmd=[sys.executable,'-B','-u',str(Path(__file__)), '--run',str(run),'--task',stage]+(['--resume'] if a.resume else [])
        with (out/'supervisor.log').open('ab') as log:
            job=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        text(pidfile,str(job.pid)+'\n');save(out/'launch.json',dict(pid=job.pid,command=cmd,started=now()));print(json.dumps(dict(pid=job.pid,command=cmd,log=str(out/'worker.log'))));return
    if a.worker:
        begin=time.monotonic()
        try:
            module=importlib.import_module({'A':'task_a','B':'task_b','C':'task_c','A4':'task_a4','report':'report_phase3'}[stage])
            value=module.main(run)
            save(out/'worker_status.json',dict(status='completed',time_beijing=now(),seconds=time.monotonic()-begin,result=value,resources=resource_sample()))
        except BaseException as error:
            save(out/'worker_status.json',dict(status='blocked' if isinstance(error,GateFailure) else 'failed',time_beijing=now(),reason=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-begin))
            raise
        return
    with run_lock(out):
        if digest(make_identity(read(CONFIG),SOURCE))!=EXPECTED:raise GateFailure('Global frozen identity mismatch')
        guard=guard_for(run,stage);cmd=[sys.executable,'-B','-u',str(Path(__file__)),'--run',str(run),'--task',stage,'--worker']+(['--resume'] if a.resume else [])
        with (out/'worker.log').open('ab') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
            started=now();save(out/'status.json',dict(status='running',supervisor_pid=os.getpid(),worker_pid=child.pid,command=cmd,started=started))
            try:
                while child.poll() is None:
                    guard.check();save(out/'supervisor_heartbeat.json',dict(time_beijing=now(),unix=time.time(),pid=child.pid,resources=guard.last_sample));time.sleep(2)
                guard.check()
                worker=read(out/'worker_status.json') if (out/'worker_status.json').exists() else {}
                save(out/'status.json',dict(status=worker.get('status','failed') if child.returncode else 'completed',started=started,finished=now(),supervisor_pid=os.getpid(),worker_pid=child.pid,returncode=child.returncode,worker=worker))
            except BaseException as error:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(10)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                save(out/'status.json',dict(status='blocked',started=started,finished=now(),reason=repr(error),storage_dependent_tasks_must_stop=True))
                raise
if __name__=='__main__':main()
