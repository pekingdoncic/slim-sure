"""Persistent CPU-only canonical LO-RANSAC, one immutable receipt per seed."""
import os,sys,json,time,hashlib,argparse,traceback
from pathlib import Path
import numpy as np
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp.'+str(os.getpid()))
 with t.open('w') as f:json.dump(x,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--worker',type=int,required=True);a=p.parse_args()
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 run=Path(a.run);protocol=read(run/'geometry_protocol.json');assert protocol['geometry']==dict(backend='LO-RANSAC',pixel_threshold=.5,confidence=.99999) and protocol['seeds']==list(range(5))
 for path,h in protocol['source_sha256'].items():assert sha(path)==h
 code=read(run/'code_identity.json')['execution_files'];assert sha(__file__)==code[Path(__file__).name]
 sys.path.insert(0,str(Path(protocol['hybrid_root']).parent))
 from hybrid_match.evaluation.pose import estimate_pose,backend_information
 from hybrid_match.evaluation.metrics import pose_error
 backend=backend_information('LO-RANSAC');reference=read('/home/bupt/projects/hybrid_match/outputs/m2b1_cpu_20261112T133600/solver_contract.json')['backend_information']['LO-RANSAC']
 for key in ['poselib_version','poselib_default_ransac_options','poselib_default_bundle_options','poselib_seed_supported']:assert backend[key]==reference[key],key
 save(run/'pose'/('backend_'+str(a.worker)+'.json'),backend);start=time.monotonic();done=new=0
 for ds in ['megadepth1500','scannet1500']:
  ident=digest(dict(context=sha(run/'context.json'),code=code,dataset=ds))
  rs=[json.loads(x) for x in (run/(ds+'_records.jsonl')).open()];methods=['A','B0','B']+(['C'] if read(run/ds/'coverage.json')['gate_pass'] else [])
  for index,r in enumerate(rs):
   if index%3!=a.worker:continue
   for m in methods:
    base=run/ds/m/'pairs'/r['pair_id'];pred=base.with_suffix('.npz');rc=read(base.with_suffix('.receipt.json'));assert rc['identity']==ident and rc['arrays_sha']==sha(pred)
    with np.load(pred,allow_pickle=False) as z:pt0=z['source'].astype(np.float64)*z['scale0'];pt1=z['target'].astype(np.float64)*z['scale1'];K0=z['K0'];K1=z['K1'];T=z['T']
    for seed in range(5):
     out=run/'pose'/ds/m/r['pair_id']/('seed_'+str(seed)+'.json');receipt=out.with_suffix('.receipt.json')
     key=dict(pair_id=r['pair_id'],dataset=ds,method=m,seed=seed,prediction_sha=rc['arrays_sha'],protocol_sha=sha(run/'geometry_protocol.json'),identity=ident)
     if out.exists() and receipt.exists():
      prev=read(receipt);assert prev['identity']==key and prev['sha256']==sha(out);done+=1;continue
     save(run/'heartbeats'/('pose_'+str(a.worker)+'.json'),dict(unix=time.time(),pid=os.getpid(),dataset=ds,method=m,pair_id=r['pair_id'],seed=seed,committed=done,state='solver'))
     result=estimate_pose(pt0,pt1,K0,K1,protocol['geometry'],seed)
     if result['status']=='success':
      Rerr,terr=pose_error(T,result['R'],result['t']);assert np.isfinite([Rerr,terr]).all(),'Undefined GT pose error';error=max(Rerr,terr)
     else:Rerr=terr=error=None
     value=dict(status='committed',identity=key,scene=r['scene'],num_matches=len(pt0),num_inliers=int(result['num_inliers']),solver_status=result['status'],failure_reason=result['failure_reason'],rotation_error=Rerr,translation_error=terr,pose_error_deg=error,geometry_seconds=float(result['geometry_seconds']),backend_invoked=bool(result['backend_invoked']),backend_seed=result['backend_seed'],backend_seed_controlled=bool(result['backend_seed_controlled']),resolved_geometry=result['resolved_geometry'],seed_control=result['seed_control'])
     save(out,value);save(receipt,dict(identity=key,sha256=sha(out)));done+=1;new+=1
     if done%100==0:
      progress=dict(unix=time.time(),pid=os.getpid(),worker=a.worker,completed=done,new=new,seconds=time.monotonic()-start,seconds_per_new_unit=(time.monotonic()-start)/max(new,1))
      save(run/'pose'/('progress_'+str(a.worker)+'.json'),progress);print(json.dumps(progress),flush=True)
 save(run/'pose'/('status_'+str(a.worker)+'.json'),dict(status='completed',completed=done,new=new,seconds=time.monotonic()-start))
if __name__=='__main__':
 try:main()
 except BaseException:
  print(traceback.format_exc(),flush=True);raise
