"""Requested serial runs only; each method receives a fresh process."""
from common import *
import argparse
def main():
 p=argparse.ArgumentParser();p.add_argument('--methods',nargs='+',required=True);p.add_argument('--background',action='store_true');a=p.parse_args()
 if a.background:
  cmd=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--methods',*a.methods]
  with (RUN/'chain.log').open('ab') as f:proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  save(RUN/'chain_launch.json',dict(time=now(),pid=proc.pid,command=cmd));print(json.dumps(dict(pid=proc.pid,methods=a.methods)));return
 for method in a.methods:
  if method not in ['A1','A2']:
   assert read(OUT/'stage0_acceptance.json')['status']=='pass'
   assert (OUT/'Stage1_Report.md').exists()
  if method in ['C4','C8','C12','Caffine']:
   assert read(OUT/('coverage_'+method+'.json'))['gate_pass']
  save(RUN/'stage_status.json',dict(status='running',stage=method,independent_model_runs=0,time=now()))
  cmd=[sys.executable,'-B','-u',str(Path(__file__).parent/'worker.py'),'--method',method]
  rc=subprocess.run(cmd,cwd=ROOT).returncode
  if rc:save(RUN/'stage_status.json',dict(status='failed',stage=method,exit_code=rc,independent_model_runs=0,time=now()));raise SystemExit(rc)
 save(RUN/'stage_status.json',dict(status='awaiting_analysis',completed_methods=a.methods,independent_model_runs=0,time=now()))
if __name__=='__main__':main()
