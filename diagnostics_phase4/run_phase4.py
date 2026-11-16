"""Resumable background CPU task launcher; no Runtime/model construction."""
from common4 import *
import argparse,subprocess,signal,traceback,importlib
from reliability_v1.phase2_runtime import make_identity
from reliability_v1.runtime import run_lock,resource_sample
from runner import guard_for
STAGES=['P0','P1','P2','P2b','P3','P4','report']
def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--stage',choices=STAGES+['remaining'],required=True);p.add_argument('--background',action='store_true');p.add_argument('--worker',action='store_true')
    a=p.parse_args();run=ROOT/'runs'/('phase4_diag_'+a.run_id);out=run/a.stage;out.mkdir(exist_ok=True)
    context=read(run/'context.json')
    if Path(context['results'])!=ROOT/'experiment_results'/('phase4_'+a.run_id):raise GateFailure('Result/run ID mismatch')
    if a.background:
        if (out/'status.json').exists() and read(out/'status.json').get('status')=='completed':raise GateFailure('Already completed')
        for f in run.glob('*/supervisor.pid'):
            old=int(f.read_text());cmd=Path('/proc')/str(old)/'cmdline'
            if cmd.exists() and str(run.name.removeprefix('phase4_diag_')).encode() in cmd.read_bytes():raise GateFailure('Existing active Phase 4 writer')
        cmd=[sys.executable,'-B','-u',str(Path(__file__)),'--run-id',a.run_id,'--stage',a.stage]
        with (out/'supervisor.log').open('ab') as log:
            job=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'))
        text(out/'supervisor.pid',str(job.pid));save(out/'launch.json',dict(pid=job.pid,command=cmd,time_beijing=now()));print(json.dumps(read(out/'launch.json')));return
    if a.stage=='remaining':
        for stage in STAGES:
            state=run/stage/'status.json'
            if state.exists() and read(state).get('status')=='completed':continue
            code=subprocess.call([sys.executable,'-B','-u',str(Path(__file__)),'--run-id',a.run_id,'--stage',stage],cwd=ROOT)
            if code:raise RuntimeError('Phase stopped: '+stage)
        save(out/'status.json',dict(status='completed',time_beijing=now()));return
    if a.worker:
        begin=time.monotonic()
        try:
            value=importlib.import_module('phase4_'+a.stage.lower()).main(run)
            save(out/'worker_status.json',dict(status='completed',time_beijing=now(),seconds=time.monotonic()-begin,result=value,resources=resource_sample()))
        except BaseException as exc:
            save(out/'worker_status.json',dict(status='blocked' if isinstance(exc,GateFailure) else 'failed',time_beijing=now(),reason=repr(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-begin));raise
        return
    with run_lock(run):
        actual=digest(make_identity(read(CONFIG),SOURCE))
        if actual!=EXPECTED:
            save(run/'global_identity_failure.json',dict(expected=EXPECTED,actual=actual,time_beijing=now()));raise GateFailure('GLOBAL IDENTITY MISMATCH')
        guard=guard_for(run,a.stage)
        command=[sys.executable,'-B','-u',str(Path(__file__)),'--run-id',a.run_id,'--stage',a.stage,'--worker']
        with (out/'worker.log').open('ab') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
            started=now();save(out/'status.json',dict(status='running',supervisor_pid=os.getpid(),worker_pid=child.pid,command=command,started=started))
            try:
                while child.poll() is None:
                    guard.check();save(out/'supervisor_heartbeat.json',dict(time_beijing=now(),worker_pid=child.pid,resources=guard.last_sample));time.sleep(2)
                guard.check();worker=read(out/'worker_status.json')
                state=dict(status=worker['status'] if child.returncode else 'completed',returncode=child.returncode,started=started,finished=now(),supervisor_pid=os.getpid(),worker_pid=child.pid,command=command,worker=worker)
                save(out/'status.json',state)
                globalstate=read(run/'stage_status.json');globalstate['stages'][a.stage]=state['status'];globalstate.update(status='running' if child.returncode==0 and a.stage!='report' else state['status'],last_verified_beijing=now());save(run/'stage_status.json',globalstate)
                if child.returncode:raise RuntimeError('Worker failed; see '+str(out/'worker_status.json'))
            except BaseException as exc:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(10)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                if not (out/'status.json').exists() or read(out/'status.json').get('status')=='running':
                    save(out/'status.json',dict(status='blocked',reason=repr(exc),started=started,finished=now()))
                raise
if __name__=='__main__':main()
