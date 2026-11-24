"""Derived native callback only after native TopK/threshold; never alters upstream P."""
import ast,types,inspect,textwrap,hashlib
from common import *
from policies import select,oracle
import torch
def matrix_hash(t,block=128):
 h=hashlib.sha256();h.update(str((tuple(t.shape),str(t.dtype))).encode())
 for s in range(0,len(t),block):h.update(t[s:s+block].contiguous().cpu().numpy().tobytes())
 return h.hexdigest()
def ranks_at(t,rows,cols,raw_i,raw_j,block=128):
 rr=[];cc=[]
 for start in range(0,len(rows),block):
  ri=rows[start:start+block];cj=cols[start:start+block];v=t[ri,cj]
  row=t[ri];col=t[:,cj].T
  rr.append((1+((row>v[:,None])|((row==v[:,None])&(raw_j[None,:]<raw_j[cj,None]))).sum(1)).cpu().numpy())
  cc.append((1+((col>v[:,None])|((col==v[:,None])&(raw_i[None,:]<raw_i[ri,None]))).sum(1)).cpu().numpy())
 return np.concatenate(rr) if rr else np.empty(0,int),np.concatenate(cc) if cc else np.empty(0,int)
def boundary_counts(t,rows,cols,block=128):
 out={k:[] for k in ['row_tie1','row_tie5','col_tie1','col_tie5']}
 for start in range(0,len(rows),block):
  for side,a in [('row',t[rows[start:start+block]]),('col',t[:,cols[start:start+block]].T)]:
   v=torch.topk(a,k=min(6,a.shape[1]),dim=1).values
   for k in [1,5]:
    if a.shape[1]<=k:x=torch.zeros(len(a),device=a.device,dtype=torch.long)
    else:x=torch.where(v[:,k-1]==v[:,k],(a==v[:,k-1,None]).sum(1),0)
    out[side+'_tie'+str(k)].append(x.cpu().numpy())
 return {k:np.concatenate(v) if v else np.empty(0,int) for k,v in out.items()}
class Capture:
 def __init__(self,method):self.method=method;self.active=False;self.current=None
 def __call__(self,model,d,sim,P,ci,cj,topidx,topvals,vm,native):
  if not self.active:return native,topvals[vm]
  assert d['image0'].shape[0]==1 and not model.texture_budget_enabled
  n0,n1=P.shape;ids0=ci.cpu().numpy();ids1=cj.cpu().numpy();native_cpu=native.cpu().numpy()
  # Replay native selection from the SAME P, including original torch.topk tie order.
  rv,ri=torch.topk(P.view(-1),k=len(topidx));replay=ri[rv>P.max()*model.coarse_match_thres]
  assert torch.equal(replay,native) and torch.equal(ri,topidx) and torch.equal(rv,topvals),'native replay structure'
  c=dict(candidate0=ids0,candidate1=ids1,native_i=ids0[native_cpu//n1],native_j=ids1[native_cpu%n1],native_flat=native_cpu,
   native_p=P.view(-1)[native].cpu().numpy(),topk_flat=topidx.cpu().numpy(),topk_values=topvals.cpu().numpy(),threshold=np.array(float((P.max()*model.coarse_match_thres).float())),K=np.array(len(topidx)),w0=np.array(d['hw0_c'][1]),w1=np.array(d['hw1_c'][1]))
  shaP=matrix_hash(P);shaS=matrix_hash(sim)
  c['P_sha']=np.array(shaP);c['sim_sha']=np.array(shaS);c['P_dtype']=np.array(str(P.dtype))
  low=low_grids(d,0,model.texture_threshold)['low'];li=np.flatnonzero(low[ids0]);c['low_i']=ids0[li]
  poolj=np.full((len(li),5),-1,np.int64);poolp=np.zeros((len(li),5),np.float32);poolcol=np.zeros((len(li),5),np.int64)
  for start in range(0,len(li),128):
   rows=torch.as_tensor(li[start:start+128],device=P.device)
   a=P[rows];idx=torch.argsort(a,dim=1,descending=True,stable=True)[:,:min(5,n1)];val=a.gather(1,idx)
   rp=rows[:,None].expand_as(idx).reshape(-1);cp=idx.reshape(-1)
   _,rank=ranks_at(P,rp,cp,ci,cj)
   k=idx.shape[1];sl=slice(start,start+len(rows));v=val.float().cpu().numpy();ok=np.isfinite(v)&(v>0)
   poolj[sl,:k]=np.where(ok,ids1[idx.cpu().numpy()],-1);poolp[sl,:k]=np.where(ok,v,0);poolcol[sl,:k]=rank.reshape(-1,k)
  c.update(pool_j=poolj,pool_p=poolp,pool_col_rank=poolcol)
  nr=torch.as_tensor(native_cpu//n1,device=P.device);nc=torch.as_tensor(native_cpu%n1,device=P.device)
  rr,cr=ranks_at(P,nr,nc,ci,cj);c.update(native_row_rank=rr,native_col_rank=cr,native_mutual=(rr==1)&(cr==1))
  plan=select(c);target=plan['B0'] if self.method=='B0' else plan['B']
  if self.method.startswith('C'):
   radius=int(self.method[1:]) if self.method[1:].isdigit() else 4
   plan=select(c,radius=radius,affine=self.method=='Caffine');target=plan['C']
  # Ground truth is isolated below; online policies above have no access to it.
  g=grid_contract(d,model.texture_threshold)
  if self.method=='oracle':target,inside,v=oracle(plan,g['correct'],g['den']);plan.update(oracle_inside=inside,oracle_GT_valid=v)
  if self.method in ['A1','A2','dev']:selected=native
  else:
   ref=get('A1',self.current,True)
   assert shaP==str(ref['capture_P_sha']) and shaS==str(ref['capture_sim_sha']),'Upstream scores differ from paired A1'
   assert np.array_equal(ids0,ref['candidate0']) and np.array_equal(ids1,ref['candidate1'])
   pos0=np.searchsorted(ids0,target[:,0]);pos1=np.searchsorted(ids1,target[:,1])
   assert np.array_equal(ids0[pos0],target[:,0]) and np.array_equal(ids1[pos1],target[:,1])
   selected=torch.as_tensor(pos0*n1+pos1,device=P.device)
  d['_rank_capture']={**{'capture_'+k:v for k,v in c.items()},**{'plan_'+k:v for k,v in plan.items()}}
  if self.method in ['A1','A2','dev']:
   gt=g['correct'];lookup0=np.full(len(g['den']),-1,np.int64);lookup0[ids0]=np.arange(n0)
   lookup1=np.full(len(g['grid'][1]['low']),-1,np.int64);lookup1[ids1]=np.arange(n1)
   present=g['den']&(lookup0>=0);present&=(gt>=0);ix=np.flatnonzero(present);ix=ix[lookup1[gt[ix]]>=0]
   r=torch.as_tensor(lookup0[ix],device=P.device);k=torch.as_tensor(lookup1[gt[ix]],device=P.device)
   arrays={'rank_source':ix,'rank_low':g['grid'][0]['low'][ix],'rank_gt_target':gt[ix]}
   for name,t in [('P',P),('sim',sim)]:
    row,col=ranks_at(t,r,k,ci,cj);arrays.update({f'rank_{name}_row':row,f'rank_{name}_col':col,f'rank_{name}_value':t[r,k].float().cpu().numpy()})
    arrays.update({f'rank_{name}_{key}':value for key,value in boundary_counts(t,r,k,128).items()})
   rowtop=torch.topk(P[r],k=min(2,n1),dim=1).values
   arrays['rank_P_margin']= (rowtop[:,0]-rowtop[:,1]).float().cpu().numpy() if n1>1 else np.full(len(ix),np.nan)
   gtflat=r*n1+k;arrays['rank_global_topk']=np.isin(gtflat.cpu().numpy(),c['topk_flat'])
   arrays['rank_above_threshold']=(P[r,k]>P.max()*model.coarse_match_thres).cpu().numpy()
   d['_rank_capture'].update(arrays)
  return selected,P.view(-1)[selected]
def install(model,callback):
 import src.slim as native
 src=textwrap.dedent(inspect.getsource(native.SLiM._coarse_match_norm_clue));tree=ast.parse(src)
 class Edit(ast.NodeTransformer):
  inserted=0;conf=0
  def visit_Assign(self,node):
   node=self.generic_visit(node)
   if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='coarse_match':
    self.inserted+=1
    return [node,ast.parse('coarse_match, selected_conf = _callback(self,data,sim_matrix,conf_matrix,candidate_i,candidate_j,top_k_indices,top_k_values,valid_matches,coarse_match)').body[0]]
   return node
  def visit_Call(self,node):
   if ast.unparse(node)=='coarse_confidence.append(top_k_values[valid_matches].detach())':
    self.conf+=1;return ast.parse('coarse_confidence.append(selected_conf.detach())').body[0].value
   return self.generic_visit(node)
 e=Edit();tree=e.visit(tree);assert e.inserted==1 and e.conf==1
 ast.fix_missing_locations(tree);ns=dict(vars(native));ns['_callback']=callback
 exec(compile(tree,'derived_postP_callback.py','exec'),ns)
 model._coarse_match_norm_clue=types.MethodType(ns['_coarse_match_norm_clue'],model)
 return ast.unparse(tree)
