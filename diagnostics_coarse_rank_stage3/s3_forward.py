"""Real per-method complete forwards; audits outside synchronized timing."""
from s3_common import *
from s3_data import load_pair
from timing_deploy import MinimalOnline
from timing import Online
import torch
def reconstruct(d,model):
 ci,cj=d['candidate_ids0'][0],d['candidate_ids1'][0];n0,n1=len(ci),len(cj);ids0=ci.cpu().numpy();ids1=cj.cpu().numpy()
 c=dict(candidate0=ids0,candidate1=ids1,w0=np.array(d['hw0_c'][1]),w1=np.array(d['hw1_c'][1]))
 low=low_grids(d,0,model.texture_threshold)['low'];li=np.flatnonzero(low[ids0]);c['low_i']=ids0[li]
 pj=np.full((len(li),5),-1,np.int64);pp=np.zeros((len(li),5),np.float32);pc=np.zeros((len(li),5),np.int64)
 if not n0 or not n1:
  c.update(native_i=np.empty(0,np.int64),native_j=np.empty(0,np.int64),native_mutual=np.empty(0,bool),pool_j=pj,pool_p=pp,pool_col_rank=pc,P_sha=np.array('no_matrix:'+str((n0,n1))),sim_sha=np.array('no_matrix:'+str((n0,n1))),K=np.array(0))
  return c,select(c)
 with torch.no_grad(),torch.cuda.amp.autocast(enabled=True):
  f0=d['feat0_c'];f1=d['feat1_c'];C=f0.shape[1]
  f0=f0.permute(0,2,3,1).reshape(1,-1,C)/C**.5;f1=f1.permute(0,2,3,1).reshape(1,-1,C)/C**.5
  sim=torch.einsum('lc,sc->ls',f0[0,ci].contiguous(),f1[0,cj].contiguous())/(model.coarse_temperature+1e-8)
  P=torch.softmax(sim,0)*torch.softmax(sim,1)
  assert torch.isfinite(sim).all() and torch.isfinite(P).all()
  values,idx=torch.topk(P.view(-1),k=min(model.max_coarse_matches,n0,n1));native=idx[values>P.max()*model.coarse_match_thres];native_np=native.cpu().numpy()
  c.update(P_sha=np.array(matrix_hash(P)),sim_sha=np.array(matrix_hash(sim)),native_i=ids0[native_np//n1],native_j=ids1[native_np%n1],K=np.array(len(idx)))
  for start in range(0,len(li),128):
   rows=torch.as_tensor(li[start:start+128],device=P.device);a=P[rows];ix=torch.argsort(a,dim=1,descending=True,stable=True)[:,:min(5,n1)];v=a.gather(1,ix)
   _,cr=ranks_at(P,rows[:,None].expand_as(ix).reshape(-1),ix.reshape(-1),ci,cj);k=ix.shape[1];sl=slice(start,start+len(rows));vv=v.float().cpu().numpy();ok=np.isfinite(vv)&(vv>0)
   pj[sl,:k]=np.where(ok,ids1[ix.cpu().numpy()],-1);pp[sl,:k]=np.where(ok,vv,0);pc[sl,:k]=cr.reshape(-1,k)
  rr,cr=ranks_at(P,native//n1,native%n1,ci,cj);c.update(native_mutual=(rr==1)&(cr==1),pool_j=pj,pool_p=pp,pool_col_rank=pc)
 return c,select(c)
def output_hash(d):
 return {k:array_sha(d[k].cpu().numpy()) for k in ['coarse_match_id','fine_parent_id','fine_coord_0','fine_coord_1']}
def runtime(m,out):
 rt=NativeRuntime(out/'runtime');save(out/'empty_guard.json',install_empty_guard(rt.model))
 if m in ['B0','B']:install(rt.model,MinimalOnline(m))
 elif m=='C':install(rt.model,Online('C4'))
 original=rt.model._coarse_match_norm_clue
 def timed(*args,**kw):
  torch.cuda.synchronize();t=time.perf_counter();v=original(*args,**kw);torch.cuda.synchronize();kw['data']['coarse_wall_seconds']=time.perf_counter()-t;return v
 rt.model._coarse_match_norm_clue=timed
 return rt
def coverage(ds):
 rows=[]
 for r in records(ds):
  x=get(ds,'A',r);rows.append(dict(pair_id=r['pair_id'],scene=r['scene'],**x['coverage']))
 total={k:sum(x[k] for x in rows) for k in ['S','U','H','V']};sc=sorted({x['scene'] for x in rows if x['V']});reasons=[]
 if total['U']==0:reasons.append('U=0')
 else:
  if total['H']/total['U']<.05:reasons.append('H/U<.05')
  if total['V']/total['U']<.05:reasons.append('V/U<.05')
 if total['V']<30:reasons.append('V<30')
 if len(sc)<3:reasons.append('V scenes<3')
 res=dict(dataset=ds,status='pass' if not reasons else 'skipped',gate_pass=not reasons,counts=total,ratios=dict(H_over_U=ratio(total['H'],total['U']),V_over_U=ratio(total['V'],total['U']),V_over_S=ratio(total['V'],total['S'])),valid_scenes=sc,reasons=reasons,per_pair=rows,uses_GT=False,time=now())
 save(RUN/ds/'coverage.json',res);save(OUT/(ds+'_coverage.json'),res);return res
def run(ds,m,limit=None):
 gate();rs=records(ds);out=RUN/ds/m;out.mkdir(parents=True,exist_ok=True)
 if m!='A':assert all(valid(ds,'A',r) for r in rs)
 if m=='C':assert read(RUN/ds/'coverage.json')['gate_pass']
 done=sum(valid(ds,m,r) for r in rs)
 if done==len(rs):save(out/'status.json',dict(status='completed',completed=done,planned=len(rs),resumed_no_extra_forward=True,time=now()));return
 rt=runtime(m,out);begin=time.monotonic();startdone=done;g=monitor(ds+'_'+m);sizes=[];samples=[]
 save(out/'invocation.json',dict(time=now(),pid=os.getpid(),method=m,dataset=ds,command=sys.argv,identity=identity(ds),planned=len(rs)))
 for idx,r in enumerate(rs):
  if limit and idx>=limit:break
  if valid(ds,m,r):continue
  g.check(False);stage=ds+'_'+m;beat(stage,state='load',index=idx,committed=done,pair_id=r['pair_id'])
  beginpair=time.perf_counter();inputs=load_pair(r);gt=inputs.pop('_gt')
  # GT matrices are also removed from the measured network dictionary.
  for key in ['K0','K1','T_0to1']:inputs.pop(key,None)
  warm=dict(inputs);rt.forward(warm);torch.cuda.synchronize();expected=output_hash(warm);del warm
  times=[];coarse=[];memory=[]
  for rep in range(3):
   d=dict(inputs);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter();rt.forward(d);torch.cuda.synchronize()
   times.append(time.perf_counter()-t);coarse.append(d['coarse_wall_seconds']);memory.append([torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved()])
   assert output_hash(d)==expected,'Nondeterministic repeated output'
   if rep<2:del d
  # Reconstruct identical native logits/P from saved backbone features, outside timing.
  c,plan=reconstruct(d,rt.model);expected_pairs=np.stack([c['native_i'],c['native_j']],1) if m=='A' else plan[m]
  actual=np.stack([d['i_idx_c'].cpu().numpy(),d['j_idx_c'].cpu().numpy()],1);assert np.array_equal(actual,expected_pairs),'Post-P selection mismatch'
  d['_gt']=gt;a,metrics=prior.extract(d,rt.model.texture_threshold)
  # Fine-child absence is distinct from coarse-parent absence.
  any_child=np.zeros(len(a['grid_denominator']),bool);any_child[a['source_grid'][a['parent_pos']]]=True;a['grid_any_child']=any_child
  for region,mask in [('low',a['grid_low0']),('nonlow',~a['grid_low0'])]:
   den=mask&a['grid_denominator'];metrics[region]['no_coarse_parent']=metrics[region]['no_child']
   metrics[region]['no_child']=int((den&~any_child).sum())
   metrics[region]['no_valid_child_GT']=int((den&any_child&~a['grid_has_evaluable_child']).sum())
   metrics[region]['all_valid_EPE_ge2']=int((den&a['grid_has_evaluable_child']&~a['grid_any_good']).sum())
   assert sum(metrics[region][k] for k in ['no_child','no_valid_child_GT','all_valid_EPE_ge2','success'])==metrics[region]['denominator']
  for name in ['source','target','mconf']:assert np.isfinite(a[name]).all()
  proof=dict(P_sha=str(c['P_sha']),sim_sha=str(c['sim_sha']),candidates=[array_sha(c['candidate0']),array_sha(c['candidate1'])],M=len(c['native_i']),q=int(plan['q']),r=int(plan['r']),native_selection=array_sha(np.stack([c['native_i'],c['native_j']],1)),slot_sources=array_sha(plan['sources']),target_pool=array_sha(plan['targets']))
  assert len(a['raw_id'])==proof['M'] and len(np.unique(a['raw_id']))==proof['M']
  if m!='A':
   ref=get(ds,'A',r,True);refres=get(ds,'A',r)
   assert proof==refres['selection_identity'],'Upstream/shared allocation differs from paired A'
   for key in ['grid_denominator','grid_correct_target','grid_low0','grid_low1','grid_legal0','grid_legal1']:assert np.array_equal(a[key],ref[key]),key
   transitions={}
   for reg,mask in [('low',a['grid_low0']),('nonlow',~a['grid_low0'])]:
    mask=mask&a['grid_denominator'];x=ref['grid_any_good'];y=a['grid_any_good']
    transitions[reg]=dict(denominator=int(mask.sum()),common_success=int((mask&x&y).sum()),rescued=int((mask&~x&y).sum()),lost=int((mask&x&~y).sum()),common_failure=int((mask&~x&~y).sum()))
   del ref,refres
  else:
   transitions={};a.update({'capture_'+k:v for k,v in c.items()});a.update({'plan_'+k:v for k,v in plan.items()})
  a.update(scale0=np.asarray(gt['plan0']['scale_input_to_orig_xy']),scale1=np.asarray(gt['plan1']['scale_input_to_orig_xy']),K0=gt['K0'],K1=gt['K1'],T=gt['T_0to1'])
  counts=dict(parents=len(a['raw_id']),children=len(a['valid']),valid_children=int(a['valid'].sum()),low_legal_grids=int(a['grid_low0'].sum()),legal_grids=int(a['grid_legal0'].sum()),low_GT_unavailable=int((a['grid_low0']&~a['grid_denominator']).sum()))
  result=dict(status='completed',pair_id=r['pair_id'],scene=r['scene'],dataset=ds,method=m,counts=counts,metrics=metrics,transitions_vs_A=transitions,selection_identity=proof,
   coverage=dict(S=len(plan['sources']),U=int(plan['U'].sum()),H=int(plan['H'].sum()),V=int(plan['V'].sum())),output_hashes=expected,
   timing=dict(repeats=times,coarse_repeats=coarse,median_seconds=float(np.median(times)),coarse_median_seconds=float(np.median(coarse)),peak_allocated=max(z[0] for z in memory),peak_reserved=max(z[1] for z in memory),repeats_exact=True,unit_seconds=time.perf_counter()-beginpair,scope='warmup +3 fresh synchronized forwards; GT/audit/hash/I/O/RANSAC outside measured windows'),resources=resource_sample(),time=now())
  commit(ds,m,r,a,result);done+=1;sizes.append(paths(ds,m,r)[0].stat().st_size+paths(ds,m,r)[1].stat().st_size);samples.append(result['timing']['unit_seconds'])
  beat(stage,state='committed',index=idx,committed=done,planned=len(rs),pair_id=r['pair_id'],resources=result['resources'])
  if done%25==0:
   eta=dict(time=now(),dataset=ds,method=m,completed=done,planned=len(rs),measured_seconds_per_pair=float(np.mean(samples)),remaining_method_seconds=(len(rs)-done)*float(np.mean(samples)),mean_pair_bytes=float(np.mean(sizes)),host_free_gib=shutil.disk_usage('/mnt/c').free/2**30)
   save(out/'progress.json',eta);print(json.dumps(eta),flush=True)
  del d,inputs,gt,a,c,plan,result;gc.collect()
 rt.check_frozen();g.check(False)
 save(out/'status.json',dict(status='completed' if done==len(rs) else 'partial',completed=done,planned=len(rs),new_completed=done-startdone,seconds=time.monotonic()-begin,frozen=True,time=now(),identity=identity(ds)))
 if m=='A' and done==len(rs):coverage(ds)
