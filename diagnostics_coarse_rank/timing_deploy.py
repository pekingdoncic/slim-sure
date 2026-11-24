"""Corrected deploy timing: B0 has no rank/anchor audit; B has no anchor audit."""
from common import *
import timing as runner
from capture import install,ranks_at
from policies import allocate
import torch
class MinimalOnline:
 def __init__(self,method):self.method=method
 def __call__(self,model,d,sim,P,ci,cj,topidx,topvals,vm,native):
  n0,n1=P.shape;ids0=ci.cpu().numpy();ids1=cj.cpu().numpy();native_ids=native.cpu().numpy()
  ni=ids0[native_ids//n1];nj=ids1[native_ids%n1]
  low=low_grids(d,0,model.texture_threshold)['low'];li=np.flatnonzero(low[ids0]);low_i=ids0[li]
  pj=np.full((len(li),5),-1,np.int64);pp=np.zeros((len(li),5),np.float32);compressed=np.full((len(li),5),-1,np.int64)
  for start in range(0,len(li),128):
   rows=torch.as_tensor(li[start:start+128],device=P.device);a=P[rows];idx=torch.argsort(a,dim=1,descending=True,stable=True)[:,:min(5,n1)];v=a.gather(1,idx)
   k=idx.shape[1];sl=slice(start,start+len(rows));vv=v.float().cpu().numpy();ok=np.isfinite(vv)&(vv>0)
   pj[sl,:k]=np.where(ok,ids1[idx.cpu().numpy()],-1);pp[sl,:k]=np.where(ok,vv,0);compressed[sl,:k]=idx.cpu().numpy()
  q,r,rows=allocate(ni,low_i,pj,pp);sources=low_i[rows];target=pj[rows,0].copy()
  if self.method=='B' and r:
   rankcols=np.zeros((r,5),np.int64)
   rid=np.repeat(li[rows],5);cid=compressed[rows].reshape(-1);legal=cid>=0
   lr,lc=rid[legal],cid[legal];colr=[]
   for start in range(0,len(lr),128):
    ri=torch.as_tensor(lr[start:start+128],device=P.device);cj0=torch.as_tensor(lc[start:start+128],device=P.device);v=P[ri,cj0];col=P[:,cj0].T
    colr.append((1+((col>v[:,None])|((col==v[:,None])&(ci[None,:]<ci[ri,None]))).sum(1)).cpu().numpy())
   rankcols.reshape(-1)[legal]=np.concatenate(colr) if colr else np.empty(0,int)
   score=2/(np.arange(1,6)[None,:]+rankcols);score=np.where(pp[rows]>0,score,-np.inf)
   for k,z in enumerate(rows):target[k]=pj[z,np.lexsort((pj[z],-pp[z],-score[k]))[0]]
  finali=np.r_[ni[:len(ni)-r],sources];finalj=np.r_[nj[:len(nj)-r],target]
  pi=np.searchsorted(ids0,finali);pj0=np.searchsorted(ids1,finalj);selected=torch.as_tensor(pi*n1+pj0,device=P.device)
  return selected,P.view(-1)[selected]
if __name__=='__main__':
 assert sys.argv[-1] in ['B0','B']
 runner.Online=MinimalOnline;runner.__file__=__file__;runner.main()
