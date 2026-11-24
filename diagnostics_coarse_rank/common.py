import os,sys,json,time,hashlib,subprocess,gc,math,traceback
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
sys.dont_write_bytecode=True
ROOT=Path('/home/bupt/projects/SLiM-master');sys.path.insert(0,str(ROOT))
from diagnostics_threshold.shared import NativeRuntime,low_grids,identity_gate,array_sha,ratio,records,base,pairname
from reliability_v1.runtime import sha,atomic_json as save,now,resource_sample,state_hash
from diagnostics_phase3.common import npzout,npzload,csvout,read
from reliability_data import project_visible
from reliability_v1.comparison import final_source_errors
import numpy as np
RUN=ROOT/'runs/coarse_rank_20261123T224428';OUT=ROOT/'experiment_results'/RUN.name
HISTORY=ROOT/'runs/phaseT_20261117T115519'
def code_identity():
 return {p.name:sha(p) for p in sorted((ROOT/'diagnostics_coarse_rank').glob('*.py')) if p.name in ['common.py','capture.py','policies.py','worker.py']}
def gate():
 c=read(RUN/'context.json')
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==c['HEAD']
 for p,h in c['frozen_files'].items():assert sha(p)==h,p
 assert sha(RUN/'protocol.json')==c['protocol_sha'] and sha(RUN/'diagnostic_manifest.json')==c['manifest_sha']
 identity_gate()
 assert (RUN/'inventory/route_frozen.json').exists(),'Inventory route must be frozen before results'
 return c
def recs():return read(RUN/'diagnostic_manifest.json')['records']
def paths(method,r):
 p=RUN/method/'pairs'/pairname(r['pair_id'])
 return p.with_suffix('.npz'),p.with_suffix('.json'),p.with_suffix('.receipt.json')
def committed(method,r):
 a,j,p=paths(method,r)
 if not p.exists():return False
 x=read(p);ctx=read(RUN/'context.json')
 assert x['pair_id']==r['pair_id'] and x['protocol_sha']==ctx['protocol_sha']
 assert x['code']==code_identity(),'Experiment selection/extraction code changed after commit'
 assert sha(a)==x['arrays_sha'] and sha(j)==x['result_sha']
 return True
def get(method,r,arrays=False):
 assert committed(method,r),(method,r['pair_id'])
 return npzload(paths(method,r)[0]) if arrays else read(paths(method,r)[1])
def commit(method,r,arrays,result):
 assert not committed(method,r)
 a,j,p=paths(method,r);a.parent.mkdir(parents=True,exist_ok=True)
 npzout(a,arrays);save(j,result)
 save(p,dict(time=now(),pair_id=r['pair_id'],scene=r['scene'],protocol_sha=read(RUN/'context.json')['protocol_sha'],code=code_identity(),arrays_sha=sha(a),result_sha=sha(j)))
def grid_contract(d,threshold):
 grid=[low_grids(d,s,threshold) for s in [0,1]]
 h0,w0=map(int,d['hw0_c']);h1,w1=map(int,d['hw1_c']);ids=np.arange(h0*w0)
 xy=np.stack([ids%w0,ids//w0],1)*8+4
 pr=project_visible(xy,d);gt=pr['points1_gt_input'];vis=pr['valid_gt'].astype(bool)
 finite=np.isfinite(gt).all(1);cell=np.zeros((len(ids),2),np.int64);cell[finite]=np.floor(gt[finite]/8).astype(np.int64)
 bounds=finite&(cell[:,0]>=0)&(cell[:,0]<w1)&(cell[:,1]>=0)&(cell[:,1]<h1)
 correct=np.full(len(ids),-1,np.int64);correct[bounds]=cell[bounds,1]*w1+cell[bounds,0]
 legal=np.zeros(len(ids),bool);legal[bounds]=grid[1]['legal'][correct[bounds]]
 return dict(grid=grid,correct=correct,den=grid[0]['legal']&vis&legal,gt=gt,reasons=pr['invalid_reason'],w0=w0,w1=w1)
def extract(d,threshold):
 g=grid_contract(d,threshold);i=d['i_idx_c'].cpu().numpy();j=d['j_idx_c'].cpu().numpy();raw=d['coarse_match_id'].cpu().numpy();m=d['m_idx_it'].cpu().numpy()
 e,valid,pr=final_source_errors(d);valid=valid.astype(bool);good=valid&(e<2)
 assert np.isfinite(e[valid]).all()
 assert np.array_equal(d['fine_parent_id'].cpu().numpy(),raw[m])
 assert np.array_equal(raw,i*len(g['grid'][1]['low'])+j)
 pc=np.bincount(m,minlength=len(i));vc=np.bincount(m[valid],minlength=len(i));gc0=np.bincount(m[good],minlength=len(i))
 stages={}
 for k,ids in [('any_parent',i),('has_evaluable_child',i[m[valid]]),('any_good',i[m[good]]),('correct_parent',i[g['den'][i]&(j==g['correct'][i])])]:
  arr=np.zeros(len(g['den']),bool);arr[ids]=True;stages[k]=arr
 ci=[d['candidate_ids'+str(s)][0].cpu().numpy() for s in [0,1]]
 stages['both_pass']=np.isin(np.arange(len(g['den'])),ci[0])&np.isin(g['correct'],ci[1])
 arrays=dict(raw_id=raw,source_grid=i,target_grid=j,parent_pos=m,child_parent_id=raw[m],source=d['fine_coord_0'].cpu().numpy(),target=d['fine_coord_1'].cpu().numpy(),
  epe=e,valid=valid,gt=pr['points1_gt_input'],mconf=d['coarse_confidence'].float().cpu().numpy(),child_counts=pc,valid_child_counts=vc,good_child_counts=gc0,
  grid_denominator=g['den'],grid_correct_target=g['correct'],grid_center_gt=g['gt'],grid_gt_reasons=g['reasons'],w0=np.array(g['w0']),w1=np.array(g['w1']),
  **{'grid_'+k:v for k,v in stages.items()})
 for s in [0,1]:arrays.update({f'candidate{s}':ci[s],f'grid_low{s}':g['grid'][s]['low'],f'grid_legal{s}':g['grid'][s]['legal'],f'grid_texture{s}':g['grid'][s]['texture']})
 assert len(set(raw.tolist()))==len(raw)
 metrics={}
 for name,mask in [('low',g['grid'][0]['low']),('nonlow',~g['grid'][0]['low'])]:
  den=g['den']&mask;succ=den&stages['any_good'];ids=np.flatnonzero(den);sids=np.flatnonzero(succ)
  # Each coarse cell lies in exactly one input 32x32 block.
  block=lambda x:(x//g['w0']//4)*math.ceil(g['w0']/4)+(x%g['w0']//4)
  blocks=len(np.unique(block(ids)));goodblocks=len(np.unique(block(sids)))
  childmask=valid&mask[i[m]]
  metrics[name]=dict(denominator=int(den.sum()),success=int(succ.sum()),recall=ratio(succ.sum(),den.sum()),
   both_pass=int((den&stages['both_pass']).sum()),correct_parent=int((den&stages['correct_parent']).sum()),
   no_child=int((den&~stages['any_parent']).sum()),no_valid_child_GT=int((den&stages['any_parent']&~stages['has_evaluable_child']).sum()),
   block_denominator=blocks,block_success=goodblocks,coverage=ratio(goodblocks,blocks),valid_children=int(childmask.sum()),good_children=int((childmask&good).sum()),precision=ratio((childmask&good).sum(),childmask.sum()))
 return arrays,metrics
def beat(method,**kw):
 save(RUN/method/'heartbeat.json',dict(time=now(),unix=time.time(),pid=os.getpid(),method=method,**kw))
