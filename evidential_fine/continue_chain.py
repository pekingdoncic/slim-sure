"""Serial stage handoff; active stages keep their existing guarded supervisor."""
import sys,os,json,time,subprocess,datetime,fcntl
from pathlib import Path
ROOT=Path('/home/bupt/projects/SLiM-master');PY='/home/bupt/miniconda3/envs/slim/bin/python'
def now():return datetime.datetime.now().astimezone().isoformat()
def read(p):return json.loads(p.read_text())
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def active(pid,run):
    p=Path('/proc')/str(pid)
    try:
        return b'run_phase5.py' in (p/'cmdline').read_bytes() and str(run).encode() in (p/'cmdline').read_bytes() and (p/'stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False
def main(run):
    run=Path(run);control=run/'chain';control.mkdir(exist_ok=True)
    with (control/'coordinator.lock').open('w') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(control/'launch.json',dict(pid=os.getpid(),command=sys.argv,started=now(),stages=['A','B','evaluate','report'],automatic_retry=False))
        save(control/'status.json',dict(status='running',time_beijing=now(),pid=os.getpid()))
        try:
            for stage in ['A','B','evaluate','report']:
                d=run/stage;d.mkdir(exist_ok=True);st=d/'status.json'
                while True:
                    state=read(st) if st.exists() else {}
                    root=read(run/'stage_status.json')
                    if root.get('stop_entire_chain'):raise RuntimeError('Global identity/storage block '+str(root))
                    if state.get('status')=='completed':
                        if active(state.get('supervisor_pid',0),run) or active(state.get('worker_pid',0),run):time.sleep(2);continue
                        if not (d/'worker_status.json').exists() or read(d/'worker_status.json').get('status')!='completed':raise RuntimeError('Completed supervisor without valid worker result')
                        break
                    if state.get('status') in ['failed','blocked','paused']:raise RuntimeError('Stage needs explicit cause audit before recovery: '+stage+' '+str(state))
                    launch=d/'launch.json'
                    if state.get('status')=='running':
                        if not active(state.get('supervisor_pid',0),run):raise RuntimeError('Running stage lost supervisor: '+stage)
                    elif launch.exists() and active(read(launch).get('pid',0),run):
                        pass
                    elif not launch.exists():
                        if stage in ['A','B'] and (d/'last.json').exists():raise RuntimeError('Existing checkpoint requires audited resume')
                        cmd=[PY,'-B',str(ROOT/'evidential_fine/run_phase5.py'),'--run',str(run),'--stage',stage,'--background']
                        x=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,timeout=60)
                        save(control/('launch_'+stage+'.json'),dict(command=cmd,time_beijing=now(),returncode=x.returncode,stdout=x.stdout,stderr=x.stderr))
                        if x.returncode:raise RuntimeError('Stage launch failed '+stage+': '+x.stderr)
                    else:raise RuntimeError('Existing launch exited before status; audit required: '+stage)
                    heartbeat=read(d/'heartbeat.json') if (d/'heartbeat.json').exists() else None
                    save(control/'heartbeat.json',dict(pid=os.getpid(),time_beijing=now(),stage=stage,supervisor=state,latest_progress=heartbeat))
                    time.sleep(10)
                save(control/('handoff_'+stage+'.json'),dict(time_beijing=now(),stage=stage,verified_supervisor_exit=True,worker=read(d/'worker_status.json')))
            save(control/'status.json',dict(status='completed',time_beijing=now(),pid=os.getpid()))
        except BaseException as e:
            save(control/'status.json',dict(status='stopped_for_review',reason=repr(e),time_beijing=now(),pid=os.getpid()));raise
if __name__=='__main__':main(sys.argv[1])
