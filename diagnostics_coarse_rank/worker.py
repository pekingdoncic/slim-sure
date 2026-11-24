"""Serial supervised real forwards, immutable per-pair receipts and resumable order."""
from common import *
from capture import Capture,install
from policies import tests
import argparse,fcntl,signal
def monitor(label):
 from reliability_v1.environment_monitor import EnvironmentMonitor
 files=[RUN/'initial_environment.json',*list((RUN/'environment_probe').glob('*/status.json'))]
 entries=[(p,read(p)) for p in files];boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 entries=[(p,x) for p,x in entries if x.get('status')=='pass' and x.get('boot_id')==boot]
 p,s=max(entries,key=lambda z:z[0].stat().st_mtime_ns)
 g=EnvironmentMonitor.__new__(EnvironmentMonitor);g.run=RUN;g.out=RUN/'environment_probe'/label;g.out.mkdir(parents=True,exist_ok=True)
 g.minimum_host_bytes=2*2**30;g.boot=boot;g.cursor=s['journal_cursor'];g.started=time.monotonic();g.checks=0;g.last_sample=None;g.blocked_reason=None;g.status=None
 save(g.out/'seed.json',dict(path=str(p),sha256=sha(p),cursor=g.cursor));g.check(False);return g
def run(method,limit):
 import torch
 gate();out=RUN/method;out.mkdir(exist_ok=True);cb=Capture(method);rt=NativeRuntime(out/'runtime')
 derived=install(rt.model,cb);(out/'derived_native.txt').write_text(derived)
 save(out/'invocation.json',dict(time=now(),pid=os.getpid(),method=method,planned=100,limit=limit,code=code_identity(),command=sys.argv))
 save(out/'selector_tests.json',tests());rs=recs();begin=time.monotonic();done=sum(committed(method,r) for r in rs)
 # Fresh process warmup, never counted as a diagnostic sample.
 warm=dict(rt.load(rs[0]));rt.forward(warm);torch.cuda.synchronize();del warm;gc.collect();cb.active=True
 for index,r in enumerate(rs):
  if limit and index>=limit:break
  if committed(method,r):continue
  cb.current=r;beat(method,state='forward',index=index,pair_id=r['pair_id'],committed=done)
  t=time.perf_counter();inputs=rt.load(r);d=dict(inputs);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t0=time.perf_counter()
  rt.forward(d);torch.cuda.synchronize();forward=time.perf_counter()-t0
  arrays,metrics=extract(d,rt.model.texture_threshold);arrays.update(d['_rank_capture'])
  assert len(arrays['raw_id'])==len(arrays['capture_native_i'])
  histfrom=__import__('diagnostics_threshold.shared',fromlist=['get_unit'])
  hist=histfrom.get_unit(HISTORY,'S0',r,True)
  checks={k:np.array_equal(arrays[k],hist[k]) for k in ['grid_denominator','grid_correct_target','grid_low0','grid_low1','grid_legal0','grid_legal1']}
  assert all(checks.values()),('fixed masks differ',checks)
  if method=='dev':
   cb.active=False;plain=dict(inputs);rt.forward(plain);torch.cuda.synchronize();cb.active=True
   same={k:torch.equal(d[k],plain[k]) for k in ['i_idx_c','j_idx_c','coarse_match_id','fine_parent_id','fine_coord_0','fine_coord_1']}
   assert all(same.values()),same
  else:same=None
  result=dict(**base(r),method=method,planned=100,status='completed',metrics=metrics,counts=dict(parents=len(arrays['raw_id']),children=len(arrays['valid']),valid_children=int(arrays['valid'].sum())),
   structure=dict(native_replay_exact=True,masks_exact_history=checks,parent_ID_mapping_exact=True,unique_parents=len(np.unique(arrays['raw_id'])),dev_default_off_vs_hook=same),
   timing=dict(forward_with_audit_seconds=forward,unit_seconds=time.perf_counter()-t,deploy_cost=False,note='Includes rank/hash audit; deployment timing requires separate audit-free 3x run after approval'),
   resources=dict(**resource_sample(),gpu_allocated=torch.cuda.memory_allocated(),gpu_reserved=torch.cuda.memory_reserved(),gpu_peak_allocated=torch.cuda.max_memory_allocated(),gpu_peak_reserved=torch.cuda.max_memory_reserved()),elapsed=time.monotonic()-begin)
  commit(method,r,arrays,result);done+=1
  print(json.dumps(dict(event='committed',method=method,index=index+1,completed=done,seconds=result['timing']['unit_seconds'],counts=result['counts'])),flush=True)
  beat(method,state='committed',index=index,committed=done,planned=100,resources=result['resources'])
  del d,inputs,arrays,hist;gc.collect()
 rt.check_frozen()
 save(out/'worker_status.json',dict(status='completed' if done==100 else 'partial',completed=done,planned=100,failed=0,seconds=time.monotonic()-begin,frozen=True,code=code_identity(),time=now()))
def supervise(method,limit):
 out=RUN/method;out.mkdir(exist_ok=True)
 with (RUN/'writer.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  guard=monitor(method)
  cmd=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--method',method,'--worker']+(['--limit',str(limit)] if limit else [])
  with (out/'worker.log').open('ab') as log:
   child=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   save(out/'status.json',dict(status='running',supervisor_pid=os.getpid(),worker_pid=child.pid,command=cmd,started=now()))
   try:
    while child.poll() is None:
     guard.check(False);save(out/'supervisor_heartbeat.json',dict(time=now(),unix=time.time(),child_pid=child.pid,resources=guard.last_sample))
     hb=out/'heartbeat.json'
     if hb.exists() and time.time()-read(hb)['unix']>600:raise RuntimeError('Worker heartbeat stale >600s')
     time.sleep(2)
    guard.check(False)
    s=read(out/'worker_status.json') if (out/'worker_status.json').exists() else {}
    save(out/'status.json',dict(status=s.get('status','failed'),worker_pid=child.pid,exit_code=child.returncode,finished=now(),worker=s))
    if child.returncode:raise RuntimeError('Worker exited '+str(child.returncode))
   except BaseException as e:
    if child.poll() is None:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=30)
    save(out/'status.json',dict(status='failed',reason=repr(e),worker_pid=child.pid,time=now(),last_committed=sum(committed(method,r) for r in recs())))
    raise
def main():
 p=argparse.ArgumentParser();p.add_argument('--method',required=True,choices=['dev','A1','A2','B0','B','oracle','C4','C8','C12','Caffine']);p.add_argument('--limit',type=int);p.add_argument('--worker',action='store_true');p.add_argument('--background',action='store_true');a=p.parse_args()
 out=RUN/a.method;out.mkdir(exist_ok=True)
 if a.background:
  cmd=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--method',a.method]+(['--limit',str(a.limit)] if a.limit else [])
  with (out/'supervisor.log').open('ab') as log:proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  save(out/'launch.json',dict(time=now(),pid=proc.pid,command=cmd));print(json.dumps(dict(pid=proc.pid,command=cmd,log=str(out/'worker.log'))));return
 if a.worker:
  try:run(a.method,a.limit)
  except BaseException as e:
   save(out/'worker_status.json',dict(status='failed',time=now(),reason=repr(e),traceback=traceback.format_exc()));raise
 else:supervise(a.method,a.limit)
if __name__=='__main__':main()
