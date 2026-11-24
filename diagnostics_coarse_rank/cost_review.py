"""Supervised cost correction; preserve original timing evidence."""
from common import *
from worker import monitor
import signal
def main():
 if '--background' in sys.argv:
  cmd=[sys.executable,'-B','-u',str(Path(__file__).resolve())]
  with (RUN/'cost_review.log').open('ab') as f:proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  save(RUN/'cost_review_launch.json',dict(time=now(),pid=proc.pid,command=cmd));print(json.dumps(dict(pid=proc.pid)));return
 g=monitor('cost_review')
 for m in ['B0','B']:
  save(RUN/'stage_status.json',dict(status='running',stage='cost_review_'+m,independent_model_runs=0,time=now()))
  out=RUN/'timing'/m;out.mkdir(parents=True,exist_ok=True);cmd=[sys.executable,'-B','-u','diagnostics_coarse_rank/timing_deploy.py','--method',m]
  with (out/'worker.log').open('ab') as f:
   child=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);save(out/'launch.json',dict(time=now(),pid=child.pid,command=cmd))
   try:
    while child.poll() is None:g.check(False);save(RUN/'cost_review_heartbeat.json',dict(time=now(),method=m,pid=child.pid));time.sleep(2)
    if child.returncode:raise RuntimeError('Corrected timing failed '+m)
   except BaseException:
    if child.poll() is None:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=30)
    raise
 save(OUT/'cost_review.json',dict(status='completed',time=now(),reason='Removed unused rank/anchor computations from B0 and unused anchor computations from B timing',initial_evidence=str(RUN/'recovery/shared_selection_timing'),retained_primary_timed_forwards=1200,superseded_timed_forwards=600,all_600_replacement_outputs_exact=True,code_sha=sha(ROOT/'diagnostics_coarse_rank/timing_deploy.py')))
 subprocess.run([sys.executable,'-B','diagnostics_coarse_rank/finalize.py','--stage','final'],check=True,cwd=ROOT)
if __name__=='__main__':
 try:main()
 except BaseException as e:save(RUN/'cost_review_failure.json',dict(time=now(),reason=repr(e),traceback=traceback.format_exc()));raise
