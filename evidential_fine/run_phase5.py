"""Phase5 independent background stage supervisor; serial GPU and cursor guard."""
import os,sys
from pathlib import Path
ROOT=Path('/home/bupt/projects/SLiM-master');sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from evidential_fine.common import *
import argparse,traceback,signal
from reliability_v1.runtime import run_lock
def monitor(run,stage):
    sys.path.insert(0,str(ROOT/'diagnostics_phase3'))
    from runner import guard_for
    return guard_for(run,'phase5_'+stage)
def stop_owned_geometry(run):
    """Geometry workers use independent sessions; explicitly stop only verified owners."""
    stopped=[]
    for path in sorted((Path(run)/'geometry').glob('launch_*.json')):
        record=read(path);pid=int(record['pid']);proc=Path('/proc')/str(pid)/'cmdline'
        expected=str(Path(run)/'geometry/plan.json').encode()
        try:
            command=proc.read_bytes()
            if b'pose_units.py' not in command or expected not in command:continue
            os.killpg(pid,signal.SIGTERM);stopped.append(pid)
        except ProcessLookupError:pass
        except FileNotFoundError:pass
    deadline=time.time()+5
    while time.time()<deadline and any((Path('/proc')/str(pid)/'cmdline').exists() for pid in stopped):time.sleep(.2)
    for pid in stopped:
        proc=Path('/proc')/str(pid)/'cmdline'
        try:
            command=proc.read_bytes()
            if b'pose_units.py' in command and expected in command:os.killpg(pid,signal.SIGKILL)
        except (ProcessLookupError,FileNotFoundError):pass
    return stopped
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--stage',required=True,choices=['E0','A','B','evaluate','report']);p.add_argument('--background',action='store_true');p.add_argument('--worker',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args();run=Path(a.run);out=run/a.stage;out.mkdir(exist_ok=True)
    if a.background:
        if (out/'supervisor.pid').exists():
            old=int((out/'supervisor.pid').read_text());proc=Path('/proc')/str(old)/'cmdline'
            if proc.exists() and str(run).encode() in proc.read_bytes():raise RuntimeError('Same stage already running')
        cmd=['nohup',sys.executable,'-B','-u',str(Path(__file__).resolve()),'--run',str(run),'--stage',a.stage]+(['--resume'] if a.resume else [])
        with (out/'supervisor.log').open('ab') as log:
            c=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
        text(out/'supervisor.pid',str(c.pid)+'\n');save(out/'launch.json',dict(command=cmd,pid=c.pid,started=now()));print(json.dumps(read(out/'launch.json')));return
    if a.worker:
        try:
            if a.stage=='E0':
                from evidential_fine.e0 import run as execute
                result=execute(run)
            elif a.stage in ['A','B']:
                from evidential_fine.training import train
                result=train(run,a.stage,a.resume)
            elif a.stage=='evaluate':
                from evidential_fine.final_eval import execute
                result=execute(run)
            else:
                from evidential_fine.report import execute
                result=execute(run)
            save(out/'worker_status.json',dict(status='completed',time_beijing=now(),pid=os.getpid(),result=result))
        except BaseException as e:
            save(out/('failure_'+str(time.time_ns())+'.json'),dict(status='failed',reason=repr(e),traceback=traceback.format_exc(),time_beijing=now(),pid=os.getpid()))
            raise
        return
    identity_gate();context=read(run/'context.json');tr=Path(context['T_run'])
    if not (tr/'handoff.json').exists():raise RuntimeError('T report/process handoff not committed')
    if read(tr/'stage_status.json').get('stop_entire_chain'):raise RuntimeError('T globally stopped entire chain')
    if a.stage in ['A','B'] and read(run/'E0/summary.json')['routes'][a.stage]['status']!='pass':raise RuntimeError('Route failed E0; training forbidden')
    if a.stage=='B':
        review=run/'E0/native_final_reference_review.json'
        if not review.exists() or read(review).get('status')!='pass':
            raise RuntimeError('B exact Phase4 final-output reference gate pending; run evidential_fine/e0_final_reference.py')
    with run_lock(run):
        g=monitor(run,a.stage);cmd=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--run',str(run),'--stage',a.stage,'--worker']+(['--resume'] if a.resume else [])
        with (out/'worker.log').open('ab') as log:
            c=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
            save(out/'status.json',dict(status='running',supervisor_pid=os.getpid(),worker_pid=c.pid,command=cmd,started=now()))
            save(run/'stage_status.json',dict(status='running',stage=a.stage,supervisor_pid=os.getpid(),worker_pid=c.pid,time_beijing=now()))
            try:
                while c.poll() is None:
                    g.check();save(out/'supervisor_heartbeat.json',dict(unix=time.time(),time_beijing=now(),worker_pid=c.pid,resources=g.last_sample))
                    heartbeat=out/'heartbeat.json'
                    if heartbeat.exists() and time.time()-heartbeat.stat().st_mtime>900:raise RuntimeError('Worker no progress for 900s')
                    time.sleep(5)
                g.check()
            except BaseException as e:
                if a.stage=='report':stop_owned_geometry(run)
                storage=bool(g.blocked_reason and ('Storage fault' in g.blocked_reason or 'storage fault' in g.blocked_reason))
                if c.poll() is None:
                    if not storage and a.stage in ['A','B']:
                        text(out/'pause_requested',repr(e));deadline=time.time()+30
                        while c.poll() is None and time.time()<deadline:time.sleep(1)
                    if c.poll() is None:
                        os.killpg(c.pid,signal.SIGTERM)
                        try:c.wait(10)
                        except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait()
                save(out/'status.json',dict(status='blocked',reason=repr(e),stop_entire_chain=storage,time_beijing=now()));save(run/'stage_status.json',dict(status='blocked',stage=a.stage,reason=repr(e),stop_entire_chain=storage));raise
        status='completed' if c.returncode==0 else 'failed'
        save(out/'status.json',dict(status=status,returncode=c.returncode,worker_pid=c.pid,supervisor_pid=os.getpid(),time_beijing=now()))
        save(run/'stage_status.json',dict(status=status if a.stage=='report' else 'stage_'+status,stage=a.stage,time_beijing=now()))
if __name__=='__main__':main()
