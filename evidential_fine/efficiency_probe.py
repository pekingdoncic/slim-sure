"""Bounded one-pair diagnosis; no optimizer or training."""
import os,sys
from pathlib import Path
ROOT=Path('/home/bupt/projects/SLiM-master');sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from evidential_fine.common import *
from evidential_fine.native import prefix,inputs,predict
from evidential_fine.efficiency import path
from evidential_fine.heads import EvidentialHead
from reliability_v1.runtime import load_checkpoint,run_lock
from evidential_fine.train_core import head_identity
from evidential_fine.run_phase5 import monitor
import torch
run=Path(sys.argv[1]);out=run/'efficiency_mismatch_probe';out.mkdir(exist_ok=True)
with run_lock(run):
 identity_gate();guard=monitor(run,'efficiency_mismatch_probe');guard.check()
 rt=NativeRuntime(out/'runtime');heads={}
 try:
  for route in ['A','B']:
   h=EvidentialHead(route).cuda().float().eval()
   load_checkpoint(run/route/'epoch_01.json',h,None,head_identity(run,route),restore_rng=False);heads[route]=h
  r=records('epe')[0];d=rt.load(r);prefix(rt,d,'A');rows={}
  for route in ['A','B']:
   guard.check();data=dict(d)
   if route=='B':
    with torch.no_grad(),torch.cuda.amp.autocast(enabled=rt.config['amp']):
     rt.model._fine_correlation(data);rt.model._get_fine_coord_test(data)
     data['fine_parent_id']=data['coarse_match_id'][data['m_idx_it']]
   features=inputs(rt,data,route);ref=get_unit(run,'final_'+route,r,True)
   with torch.no_grad(),torch.cuda.amp.autocast(enabled=False):
    whole=heads[route](features)['mu']
    parts=[heads[route](tuple(x[i:i+4096] for x in features))['mu'] for i in range(0,len(features[0]),4096)]
    chunked=torch.cat(parts) if parts else whole
   initial=data['coarse_coord_1'].float()+4 if route=='A' else data['intermediate_coord_1'].float()
   whole_target=(initial+whole).cpu().numpy();chunked_target=(initial+chunked).cpu().numpy()
   mu_whole=whole.cpu().numpy();mu_chunk=chunked.cpu().numpy()
   source,target,cp=path(rt,heads,d,route)
   row=dict(points=len(ref['cp']),initial_dtype=str(initial.dtype),cached_target_dtype=str(ref['target'].dtype),
    source_equal=np.array_equal(source.cpu().numpy(),ref['source']),parent_equal=np.array_equal(cp.cpu().numpy(),ref['cp']),
    whole_mu_max=float(np.max(np.abs(mu_whole-ref['mu']),initial=0)),chunk_mu_max=float(np.max(np.abs(mu_chunk-ref['mu']),initial=0)),
    whole_target_max=float(np.max(np.abs(whole_target-ref['target']),initial=0)),chunk_target_max=float(np.max(np.abs(chunked_target-ref['target']),initial=0)),
    original_path_target_max=float(np.max(np.abs(target.cpu().numpy()-ref['target']),initial=0)),
    whole_target_different_scalars=int((whole_target!=ref['target']).sum()),chunk_target_different_scalars=int((chunked_target!=ref['target']).sum()),
    decode_gpu_vs_cpu_max=float(np.max(np.abs(chunked_target-(initial.cpu().numpy()+mu_chunk)),initial=0)))
   rows[route]=row
   del features,whole,chunked,parts,source,target,cp,data
  guard.check();result=dict(status='diagnosed',time_beijing=now(),pair=base(r),routes=rows,checkpoint_fixed_epoch=2,optimizer_updates=0,frozen=rt.check_frozen(),old_efficiency_sha=sha(ROOT/'evidential_fine/efficiency.py'))
  save(out/'result.json',result);print(json.dumps(result))
 finally:
  save(out/'exit.json',dict(time_beijing=now(),frozen=rt.check_frozen(),pid=os.getpid()));del rt,heads;gc.collect();torch.cuda.empty_cache()
