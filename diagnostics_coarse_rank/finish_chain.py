"""Finish authorized diagnostic-only timing after optional workers have exited."""
from common import *
from worker import monitor
import signal
def main(background=False):
 if background:
  cmd=[sys.executable,'-B','-u',str(Path(__file__).resolve())]
  with (RUN/'finish_chain.log').open('ab') as f:proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  save(RUN/'finish_chain_launch.json',dict(time=now(),pid=proc.pid,command=cmd));print(json.dumps(dict(pid=proc.pid,command=cmd)));return
 g=monitor('finish_chain_watch')
 while True:
  states=[read(RUN/m/'status.json').get('status') if (RUN/m/'status.json').exists() else 'pending' for m in ['C8','C12','Caffine']]
  if 'failed' in states:raise RuntimeError('Optional worker failed; no follow-on work')
  if states==['completed']*3:break
  g.check(False);time.sleep(2)
 subprocess.run([sys.executable,'-B','diagnostics_coarse_rank/finalize.py','--stage','optional'],cwd=ROOT,check=True)
 for m in ['A1','B0','B','C4']:
  save(RUN/'stage_status.json',dict(status='running',stage='timing_'+m,independent_model_runs=0,time=now()))
  cmd=[sys.executable,'-B','-u','diagnostics_coarse_rank/timing.py','--method',m]
  out=RUN/'timing'/m;out.mkdir(parents=True,exist_ok=True)
  with (out/'worker.log').open('ab') as f:
   child=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   save(out/'launch.json',dict(pid=child.pid,time=now(),command=cmd))
   try:
    while child.poll() is None:
     g.check(False);save(RUN/'finish_chain_heartbeat.json',dict(time=now(),unix=time.time(),stage='timing_'+m,worker_pid=child.pid));time.sleep(2)
    if child.returncode:raise RuntimeError('Timing '+m+' failed; see worker log')
   except BaseException:
    if child.poll() is None:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=30)
    raise
 assert read(OUT/'real_empty_forward_contract.json')['status']=='pass'
 subprocess.run([sys.executable,'-B','diagnostics_coarse_rank/finalize.py','--stage','final'],cwd=ROOT,check=True)
if __name__=='__main__':
 try:main('--background' in sys.argv)
 except BaseException as e:
  save(RUN/'finish_chain_failure.json',dict(time=now(),reason=repr(e),traceback=traceback.format_exc()));raise
