"""Fresh process audit-free synchronized deployment timing on the diagnostic manifest."""
from common import *
from capture import install,ranks_at
from policies import select
from worker import monitor
import torch,types,argparse,fcntl,signal
class Online:
 def __init__(self,method):self.method=method
 def __call__(self,model,d,sim,P,ci,cj,topidx,topvals,vm,native):
  n0,n1=P.shape;ids0=ci.cpu().numpy();ids1=cj.cpu().numpy();idxnative=native.cpu().numpy()
  c=dict(native_i=ids0[idxnative//n1],native_j=ids1[idxnative%n1],w0=np.array(d['hw0_c'][1]),w1=np.array(d['hw1_c'][1]))
  low=low_grids(d,0,model.texture_threshold)['low'];li=np.flatnonzero(low[ids0]);c['low_i']=ids0[li]
  pj=np.full((len(li),5),-1,np.int64);pp=np.zeros((len(li),5),np.float32);pc=np.zeros((len(li),5),np.int64)
  for start in range(0,len(li),128):
   rows=torch.as_tensor(li[start:start+128],device=P.device);a=P[rows];idx=torch.argsort(a,dim=1,descending=True,stable=True)[:,:min(5,n1)];v=a.gather(1,idx)
   _,rank=ranks_at(P,rows[:,None].expand_as(idx).reshape(-1),idx.reshape(-1),ci,cj)
   k=idx.shape[1];sl=slice(start,start+len(rows));vv=v.float().cpu().numpy();ok=np.isfinite(vv)&(vv>0)
   pj[sl,:k]=np.where(ok,ids1[idx.cpu().numpy()],-1);pp[sl,:k]=np.where(ok,vv,0);pc[sl,:k]=rank.reshape(-1,k)
  c.update(pool_j=pj,pool_p=pp,pool_col_rank=pc)
  nr=torch.as_tensor(idxnative//n1,device=P.device);nc=torch.as_tensor(idxnative%n1,device=P.device)
  rr,cr=ranks_at(P,nr,nc,ci,cj);c['native_mutual']=(rr==1)&(cr==1)
  p=select(c);target=p[self.method if self.method in ['B0','B'] else 'C']
  pos0=np.searchsorted(ids0,target[:,0]);pos1=np.searchsorted(ids1,target[:,1]);selected=torch.as_tensor(pos0*n1+pos1,device=P.device)
  return selected,P.view(-1)[selected]
def run(method):
 gate();out=RUN/'timing'/method;out.mkdir(parents=True,exist_ok=True);g=monitor('timing_'+method)
 rt=NativeRuntime(out/'runtime')
 from empty_guard import install_empty_guard
 save(out/'empty_guard.json',install_empty_guard(rt.model))
 if method!='A1':install(rt.model,Online(method))
 original=rt.model._coarse_match_norm_clue
 def timed(*args,**kw):
  torch.cuda.synchronize();start=time.perf_counter();v=original(*args,**kw);torch.cuda.synchronize()
  kw['data']['coarse_wall_seconds']=time.perf_counter()-start;return v
 rt.model._coarse_match_norm_clue=timed
 rows=[];begin=time.monotonic()
 # Actual empty-output contract on the same model, with all coarse masks invalid.
 if method=='A1':
  empty=rt.load(recs()[0]);empty['mask0'].zero_();empty['mask1'].zero_()
  rt.forward(empty);torch.cuda.synchronize()
  result=dict(status='pass',parents=int(empty['coarse_match_id'].numel()),children=int(empty['fine_parent_id'].numel()),source_shape=list(empty['fine_coord_0'].shape),target_shape=list(empty['fine_coord_1'].shape),finite=bool(torch.isfinite(empty['fine_coord_0']).all() and torch.isfinite(empty['fine_coord_1']).all()))
  assert result['parents']==result['children']==0 and result['finite']
  save(OUT/'real_empty_forward_contract.json',result);del empty;gc.collect()
 for idx,r in enumerate(recs()):
  p=out/(pairname(r['pair_id'])+'.json')
  if p.exists():rows.append(read(p));continue
  g.check(False);beat('timing_'+method,index=idx,committed=len(rows),state='forward')
  inputs=rt.load(r);inputs.pop('_gt',None);warm=dict(inputs);rt.forward(warm);torch.cuda.synchronize();del warm
  times=[];coarse=[];mem=[]
  for repeat in range(3):
   d=dict(inputs);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();rt.forward(d);torch.cuda.synchronize()
   times.append(time.perf_counter()-start);coarse.append(d['coarse_wall_seconds']);mem.append([torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved()])
   # Audit output equivalence AFTER measured window; no cached choice used in forward.
   ref=get(method,r,True)
   assert np.array_equal(d['coarse_match_id'].cpu().numpy(),ref['raw_id'])
   assert np.array_equal(d['fine_coord_0'].cpu().numpy(),ref['source']) and np.array_equal(d['fine_coord_1'].cpu().numpy(),ref['target'])
   del d,ref
  row=dict(**base(r),method=method,repeats=times,coarse_repeats=coarse,median_seconds=float(np.median(times)),coarse_median_seconds=float(np.median(coarse)),peak_allocated=max(x[0] for x in mem),peak_reserved=max(x[1] for x in mem),output_exact=True)
  rows.append(row);save(p,row);del inputs;gc.collect()
  if (idx+1)%20==0:print(json.dumps(dict(method=method,timing_completed=idx+1)),flush=True)
 rt.check_frozen();g.check(False)
 res=dict(status='completed',method=method,planned=100,completed=len(rows),time=now(),seconds=time.monotonic()-begin,mean_pair_median_seconds=float(np.mean([r['median_seconds'] for r in rows])),mean_coarse_median_seconds=float(np.mean([r['coarse_median_seconds'] for r in rows])),peak_allocated=max(r['peak_allocated'] for r in rows),peak_reserved=max(r['peak_reserved'] for r in rows),rows=rows,scope='R832 batch1 AMP, fresh process, each pair warmup + 3 CUDA synchronized full fresh forwards; no GT/hash/receipt/cached-selection/RANSAC in measured windows; equivalent outputs verified afterwards',code_sha=sha(__file__))
 save(out/'status.json',res);print(json.dumps({k:v for k,v in res.items() if k!='rows'}))
def main():
 p=argparse.ArgumentParser();p.add_argument('--method',required=True,choices=['A1','B0','B','C4']);a=p.parse_args()
 with (RUN/'writer.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(a.method)
if __name__=='__main__':main()
