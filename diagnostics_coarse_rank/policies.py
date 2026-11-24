"""Deterministic post-P policies. No model, GT, filesystem or scores from evaluation."""
import numpy as np
def xy(ids,w):return np.stack([np.asarray(ids)%w,np.asarray(ids)//w],axis=-1).astype(np.float64)
def allocate(native_i, low_i, pool_j, pool_p, fraction=.1):
 M=len(native_i);q=int(np.floor(fraction*M));nonempty=(np.isfinite(pool_p)&(pool_p>0)).any(1)
 order=np.lexsort((low_i,-np.where(nonempty,pool_p[:,0],-np.inf)))
 for r in range(q,-1,-1):
  eligible=order[nonempty[order]&~np.isin(low_i[order],native_i[:M-r])]
  if len(eligible)>=r:return q,r,eligible[:r]
 raise AssertionError('r=0 is always feasible')
def select(c,radius=4,affine=False,neighbor=True,fraction=.1):
 ni,nj=c['native_i'],c['native_j'];M=len(ni)
 q,r,rows=allocate(ni,c['low_i'],c['pool_j'],c['pool_p'],fraction)
 targets=c['pool_j'][rows];p=c['pool_p'][rows];cr=c['pool_col_rank'][rows]
 b=2/(np.arange(1,6)[None,:]+cr) if r else np.empty((0,5))
 b=np.where(np.isfinite(p)&(p>0),b,-np.inf)
 sources=c['low_i'][rows];bidx=np.zeros(r,np.int64)
 for k in range(r):bidx[k]=np.lexsort((targets[k],-p[k],-b[k]))[0]
 b0idx=np.zeros(r,np.int64);cidx=bidx.copy();U=(np.isfinite(p)&(p>0)).sum(1)>=2
 anchor=np.flatnonzero(c['native_mutual'][:M-r]);ai=ni[anchor];aj=nj[anchor]
 apos=xy(ai,int(c['w0']));bpos=xy(aj,int(c['w1']));srcxy=xy(sources,int(c['w0']))
 H=np.zeros(r,bool);V=H.copy();nanchor=np.zeros(r,np.int64);residual=np.full(r,np.nan);condition=np.full(r,np.nan)
 reasons=np.full(r,'no_alternative',dtype='<U32');pred=np.full((r,2),np.nan);anchor_ids=np.full((r,8),-1,np.int64)
 for k in np.flatnonzero(U):
  reasons[k]='insufficient_anchors'
  dist=((apos-srcxy[k])**2).sum(1);legal=np.flatnonzero((dist<=radius**2)&(ai!=sources[k]))
  order=legal[np.lexsort((aj[legal],ai[legal],dist[legal]))];seen=set();chosen=[]
  for a in order:
   if int(ai[a]) in seen:continue
   seen.add(int(ai[a]));chosen.append(a)
   if len(chosen)==8:break
  nanchor[k]=len(chosen);anchor_ids[k,:len(chosen)]=anchor[chosen]
  if len(chosen)<3:continue
  H[k]=True;x=apos[chosen];y=bpos[chosen]
  if affine:
   center=x.mean(0);scale=max(float(np.sqrt(np.mean(np.sum((x-center)**2,1)))),1e-12)
   design=np.c_[(x-center)/scale,np.ones(len(x))]
   condition[k]=np.linalg.cond(design)
   if np.linalg.matrix_rank(design)<3 or not np.isfinite(condition[k]) or condition[k]>100:
    reasons[k]='collinear_or_ill_conditioned';continue
   weights=np.linalg.lstsq(design,y,rcond=None)[0]
   predicted=np.r_[(srcxy[k]-center)/scale,1]@weights;fitted=design@weights
  else:
   shift=np.median(y-x,axis=0);predicted=srcxy[k]+shift;fitted=x+shift
  residual[k]=np.median(np.linalg.norm(y-fitted,axis=1))
  if not np.isfinite(predicted).all() or not np.isfinite(residual[k]):reasons[k]='nonfinite_fit';continue
  if residual[k]>2:reasons[k]='residual_too_large';continue
  V[k]=True;reasons[k]='valid';pred[k]=predicted
  if not neighbor:continue
  d2=((xy(targets[k],int(c['w1']))-predicted)**2).sum(1);score=b[k]+.5*np.exp(-d2/8)
  # Score ties retain B before P/raw-ID ordering.
  prefer=np.arange(5)!=bidx[k];cidx[k]=np.lexsort((targets[k],-p[k],prefer,-score))[0]
 def packed(idx):
  outi=np.r_[ni[:M-r],sources];outj=np.r_[nj[:M-r],targets[np.arange(r),idx]]
  assert len(outi)==M and len(set(zip(outi.tolist(),outj.tolist())))==M
  return np.stack([outi,outj],axis=1)
 out=dict(q=np.array(q),r=np.array(r),slots=np.arange(M-r,M),rows=rows,sources=sources,targets=targets,p=p,b=b,bidx=bidx,cidx=cidx,
  B0=packed(b0idx),B=packed(bidx),C=packed(cidx),U=U,H=H,V=V,nanchor=nanchor,residual=residual,condition=condition,
  reasons=reasons,pred=pred,anchor_ids=anchor_ids,radius=np.array(radius),affine=np.array(affine),tail_source_reused=np.isin(sources,ni[M-r:]) if r else np.empty(0,bool))
 if not neighbor:assert np.array_equal(out['C'],out['B'])
 return out
def oracle(selection,gt_target,gt_valid):
 result=selection['B'].copy();s=selection['sources'];t=gt_target[s];v=gt_valid[s];pool=selection['targets']
 inside=v&(pool==t[:,None]).any(1)
 result[selection['slots'][inside],1]=t[inside]
 return result,inside,v
def tests():
 c=dict(native_i=np.array([0,1,2,3]),native_j=np.array([0,1,2,3]),low_i=np.array([2,3,4]),pool_j=np.array([[0,1,2,-1,-1],[3,4,0,-1,-1],[2,4,1,-1,-1]]),pool_p=np.array([[.8,.7,.6,0,0],[.9,.8,.7,0,0],[.9,.8,.7,0,0]]),pool_col_rank=np.ones((3,5),int),native_mutual=np.ones(4,bool),w0=np.array(7),w1=np.array(9))
 a=select(c,fraction=0);assert np.array_equal(a['B'],np.stack([c['native_i'],c['native_j']],1))
 a=select(c,fraction=.5,neighbor=False);assert a['r']==2 and a['sources'].tolist()==[3,4] and np.array_equal(a['B'],a['C'])
 assert xy([8,9],9).tolist()==[[8.,0.],[0.,1.]] and xy([8,9],7).tolist()==[[1.,1.],[2.,1.]]
 empty={**c,'native_i':np.empty(0,int),'native_j':np.empty(0,int),'native_mutual':np.empty(0,bool)}
 assert select(empty)['B'].shape==(0,2)
 no={**c,'pool_p':np.zeros((3,5))};assert select(no,fraction=1)['r']==0
 assert not np.any(np.isin(a['sources'],c['native_i'][:len(c['native_i'])-int(a['r'])]))
 return dict(status='pass',cases=['zero_quota_exact_order','C_disabled_equals_B','largest_feasible_allocator','ties_sourceID','empty_native','empty_positive_pool','different_grid_widths','protected_source_exclusion','padded_targets_excluded'])
