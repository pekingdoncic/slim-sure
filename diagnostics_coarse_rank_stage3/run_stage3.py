"""Single GPU writer, then three bounded persistent CPU pose workers, then report."""
from s3_common import *
import argparse,fcntl,signal
def run_child(cmd,label,env=None):
 out=RUN/'logs';out.mkdir(exist_ok=True);g=monitor('supervisor_'+label)
 with (out/(label+'.log')).open('ab') as f:
  p=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=env)
  save(out/(label+'_launch.json'),dict(time=now(),pid=p.pid,command=cmd));start=time.time()
  try:
   while p.poll() is None:
    g.check(False);hb=RUN/'heartbeats'/(label+'.json')
    if time.time()-max(start,read(hb)['unix'] if hb.exists() else start)>900:raise RuntimeError('Worker heartbeat stale >900s: '+label)
    save(RUN/'supervisor_heartbeat.json',dict(time=now(),unix=time.time(),stage=label,worker_pid=p.pid,resources=g.last_sample));time.sleep(2)
   g.check(False)
   if p.returncode:raise RuntimeError(label+' exited '+str(p.returncode))
  except BaseException:
   if p.poll() is None:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=30)
   raise
def supervise():
 with (RUN/'writer.lock').open('a+') as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);gate();assert read(RUN/'validation/status.json')['status']=='pass'
  for ds in APPROVED:
   for m in ['A','B0','B','C']:
    if m=='C' and not read(RUN/ds/'coverage.json')['gate_pass']:
     save(RUN/ds/m/'status.json',dict(status='skipped',planned=len(records(ds)),completed=0,reasons=read(RUN/ds/'coverage.json')['reasons']));continue
    if all(valid(ds,m,r) for r in records(ds)):
     if m=='A' and not (RUN/ds/'coverage.json').exists():
      from s3_forward import coverage
      coverage(ds)
     continue
    save(RUN/'stage_status.json',dict(status='running',stage='forward',dataset=ds,method=m,time=now()))
    run_child([sys.executable,'-B','-u',str(CODE/'run_stage3.py'),'--worker','--dataset',ds,'--method',m],ds+'_'+m)
  # CPU geometry is serial to GPU inference/timing; max three resident workers.
  save(RUN/'stage_status.json',dict(status='running',stage='pose',time=now()));g=monitor('pose_supervisor');children=[];pose_start=time.time()
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
  try:
   for wid in range(3):
    cmd=['/home/bupt/miniconda3/envs/hybrid-eval/bin/python','-B','-u',str(CODE/'s3_pose.py'),'--run',str(RUN),'--worker',str(wid)]
    log=(RUN/'logs'/('pose_'+str(wid)+'.log')).open('ab');p=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env);children.append((wid,p,log))
    save(RUN/'logs'/('pose_'+str(wid)+'_launch.json'),dict(pid=p.pid,time=now(),command=cmd))
   while any(p.poll() is None for _,p,_ in children):
    g.check(False)
    for wid,p,log in children:
     if p.poll() not in [None,0]:raise RuntimeError('Pose worker failed '+str(wid))
     hb=RUN/'heartbeats'/('pose_'+str(wid)+'.json')
     if p.poll() is None and time.time()-max(pose_start,read(hb)['unix'] if hb.exists() else pose_start)>900:raise RuntimeError('Pose heartbeat stale')
    save(RUN/'supervisor_heartbeat.json',dict(time=now(),unix=time.time(),stage='pose',workers=[dict(id=i,pid=p.pid,returncode=p.poll()) for i,p,_ in children],resources=g.last_sample));time.sleep(2)
   assert all(p.returncode==0 for _,p,_ in children)
  finally:
   for _,p,log in children:
    if p.poll() is None:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=30)
    log.close()
  run_child([sys.executable,'-B','-u',str(CODE/'s3_report.py')],'report')
def main():
 p=argparse.ArgumentParser();p.add_argument('--background',action='store_true');p.add_argument('--worker',action='store_true');p.add_argument('--dataset',choices=list(APPROVED));p.add_argument('--method',choices=['A','B0','B','C']);p.add_argument('--limit',type=int);a=p.parse_args()
 if a.background:
  previous=RUN/'launch.json'
  if previous.exists():
   q=read(previous);proc=Path('/proc')/str(q['pid'])/'cmdline'
   if proc.exists() and str(CODE/'run_stage3.py').encode() in proc.read_bytes():
    print(json.dumps(dict(status='already_running',**q)));return
  cmd=[sys.executable,'-B','-u',str(Path(__file__).resolve())]
  with (RUN/'supervisor.log').open('ab') as f:q=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  save(RUN/'launch.json',dict(time=now(),pid=q.pid,command=cmd));print(json.dumps(dict(pid=q.pid,run=str(RUN),command=cmd)));return
 if a.worker:
  from s3_forward import run
  run(a.dataset,a.method,a.limit)
 else:supervise()
if __name__=='__main__':
 try:main()
 except BaseException as e:
  save(RUN/('worker_failure_'+str(os.getpid())+'.json'),dict(time=now(),reason=repr(e),traceback=traceback.format_exc()));save(RUN/'stage_status.json',dict(status='failed',time=now(),reason=repr(e)));raise
