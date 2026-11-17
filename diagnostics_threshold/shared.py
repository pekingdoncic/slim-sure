"""New experiment utilities. Frozen files and historical runs are read-only."""
import os,sys,json,time,math,gc,hashlib,datetime,subprocess,shutil,contextlib,types,ast,inspect,textwrap
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
sys.dont_write_bytecode=True
ROOT=Path('/home/bupt/projects/SLiM-master')
sys.path.insert(0,str(ROOT))
from diagnostics_phase3.common import (read,save,text,csvout,npzout,npzload,records,pairname,base,positions,finite,clean,summary,distributions)
from reliability_v1.runtime import sha,digest,now,state_hash,prepare,resource_sample,atomic_json,append_json,seed_all
from reliability_v1.phase2_runtime import make_identity
SOURCE=ROOT/'runs/phase1_20261115T004137'
P3=ROOT/'runs/phase3_diag_20261116T100427'
P4=ROOT/'runs/phase4_diag_20261116T193724'
CONFIG=ROOT/'configs/reliability_phase2_20261115.json'
EXPECTED='1dc74dff202052885202e401ecf8fe44c40714b5877f0f547320aad86661808a'
class IdentityFailure(RuntimeError):pass
class StopRoute(RuntimeError):pass
def identity_gate():
    value=make_identity(read(CONFIG),SOURCE)
    if digest(value)!=EXPECTED:raise IdentityFailure('Global frozen identity mismatch: '+digest(value))
    return value
def array_sha(a):
    from reliability_v1.geometry_worker import array_sha
    return array_sha(a)
def receipt(run,stage,r):
    p=Path(run)/stage/'pairs'/(pairname(r['pair_id'])+'.json')
    if not p.exists():return None
    x=read(p)
    if x['identity']!=EXPECTED or x['pair_id']!=r['pair_id']:raise IdentityFailure('Committed unit identity mismatch')
    for v in x['files'].values():
        q=p.parent/v['name']
        if q.parent.resolve()!=p.parent.resolve() or sha(q)!=v['sha256']:raise IdentityFailure('Committed unit checksum mismatch: '+str(q))
    return x
def commit(run,stage,r,result,arrays):
    if receipt(run,stage,r):raise RuntimeError('Refusing to overwrite committed unit')
    p=Path(run)/stage/'pairs';p.mkdir(parents=True,exist_ok=True);name=pairname(r['pair_id']);files={}
    for kind,suffix,value in [('result','.result.json',result),('arrays','.npz',arrays)]:
        q=p/(name+suffix)
        (save if kind=='result' else npzout)(q,value)
        files[kind]={'name':q.name,'sha256':sha(q),'size':q.stat().st_size}
    x=dict(status='committed',identity=EXPECTED,**base(r),files=files,time_beijing=now())
    save(p/(name+'.json'),x);return x
def get_unit(run,stage,r,arrays=False):
    x=receipt(run,stage,r)
    if not x:return None
    p=Path(run)/stage/'pairs'/x['files']['arrays' if arrays else 'result']['name']
    return npzload(p) if arrays else read(p)
def historical(stage,r):
    from diagnostics_phase3.common import committed,arrays_of
    c=committed(P3,stage,r)
    if not c:raise IdentityFailure('Missing historical '+stage+' '+r['pair_id'])
    return arrays_of(P3,stage,c)
def beat(run,stage,**kw):
    save(Path(run)/stage/'heartbeat.json',dict(time_beijing=now(),unix=time.time(),pid=os.getpid(),stage=stage,**kw))
def ratio(a,b):return float(a/b) if b else None
def error_metrics(e):
    e=np.asarray(e,float)
    finite(e,'valid EPE')
    return dict(n=len(e),sum_EPE=float(e.sum()),mean_EPE=float(e.mean()) if len(e) else None,median_EPE=float(np.median(e)) if len(e) else None,
        **{f'lt{t}':ratio(int((e<t).sum()),len(e)) for t in [1,2,3]},gt5=ratio(int((e>5).sum()),len(e)),
        **{f'n_lt{t}':int((e<t).sum()) for t in [1,2,3]},n_gt5=int((e>5).sum()))
import numpy as np
def scene_summary(rows,metrics,methods,comparisons=()):
    return summary(rows,metrics,methods=methods,comparisons=list(comparisons))
def low_grids(d,side,threshold):
    from src.texture_budget import local_texture
    tex,cnt=local_texture(d['image'+str(side)],d['valid_mask'+str(side)],d['hw'+str(side)+'_c'])
    legal=d['mask'+str(side)].flatten().cpu().numpy().astype(bool)
    t=tex[0].cpu().numpy();count=cnt[0].cpu().numpy()
    return dict(texture=t,count=count,legal=legal,low=legal&(count>0)&np.isfinite(t)&(t<=threshold))
class NativeRuntime:
    """Strict original weights only; no reliability head or optimizer is instantiated."""
    def __init__(self,run):
        import torch
        from default_config import get_config
        from src.slim import SLiM
        self.config=read(CONFIG);self.run=Path(run);prepare(self.config,self.run)
        self.model=SLiM(get_config(self.config['native_config']).MODEL)
        obj=torch.load(self.config['checkpoint'],map_location='cpu',weights_only=False)
        loaded=self.model.load_state_dict(obj['state_dict'],strict=True);del obj
        self.model.cuda().eval().requires_grad_(False);self.model.capture_reliability=True
        self.model.sparse_direct=self.config['sparse_direct'];self.model.texture_budget_enabled=False
        self.model.texture_threshold=read(SOURCE/'texture_calibration.json')['threshold']
        assert self.model.reliability_head is None
        self.frozen_hash=state_hash(self.model);self.calls=0
        save(self.run/'native_contract.json',dict(strict_load=str(loaded),frozen_sha=self.frozen_hash,
            d_coarse=self.model.d_coarse,d_fine=self.model.d_fine,coarse_max=self.model.max_coarse_matches,
            fine_max=self.model.max_fine_matches,coarse_threshold=self.model.coarse_match_thres,
            fine_threshold=self.model.fine_match_thres,iterations=self.model.refine_iters,
            lookup=self.model.lookup_window_size,texture_fraction=self.model.texture_fraction,
            original_parameters=sum(p.numel() for p in self.model.parameters()),new_head_loaded=False))
    def check_frozen(self):
        from reliability_v1.runtime import Runtime
        return Runtime.check_frozen(self)
    def load(self,r):
        from reliability_data import load_pair
        return load_pair(r,device='cuda',resolution=832)
    def forward(self,d):
        import torch
        with torch.no_grad(),torch.cuda.amp.autocast(enabled=self.config['amp']):self.model(d)
        self.calls+=1
        return d
def setup_threshold(model,setting,out):
    """Clone the exact native function, change only its two norm entry decisions."""
    import torch,src.slim as native
    original=model._coarse_match_norm_clue
    source=textwrap.dedent(inspect.getsource(native.SLiM._coarse_match_norm_clue))
    tree=ast.parse(source);fn=tree.body[0]
    edits=0;threshold={'S0':.65,'S1':.55,'S2':.45,'S3':.30,'S4':None,'S5':.65}[setting]
    for node in ast.walk(fn):
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ['candidate_i','candidate_j']:
            if any(isinstance(q,ast.Constant) and q.value==.65 for q in ast.walk(node.value)):
                side=0 if node.targets[0].id=='candidate_i' else 1
                if setting=='S4':
                    node.value=ast.parse(f"torch.where(mask{side}[b].bool() if mask{side} is not None else torch.ones_like(norm{side}[b],dtype=torch.bool))[0]",mode='eval').body
                else:
                    for q in ast.walk(node.value):
                        if isinstance(q,ast.Constant) and q.value==.65:q.value=threshold
                edits+=1
    if edits!=2:raise IdentityFailure('Native norm source differs from audited two comparisons')
    ast.fix_missing_locations(tree);derived=ast.unparse(tree);namespace=dict(vars(native))
    exec(compile(tree,str(Path(out)/'norm_entry_wrapper.py.txt'),'exec'),namespace)
    model._coarse_match_norm_clue=types.MethodType(namespace[fn.name],model)
    model.texture_budget_enabled=setting=='S5'
    text(Path(out)/'norm_entry_wrapper.py.txt',derived+'\n')
    save(Path(out)/'effective_settings.json',dict(setting=setting,norm_threshold=threshold,norm_comparison='>' if setting!='S4' else 'legal mask only',native_source_sha=sha(ROOT/'src/slim.py'),
        entry_edits=edits,texture_budget=model.texture_budget_enabled,fraction=model.texture_fraction,valid_region_mask_preserved=True,
        coarse_match_threshold=model.coarse_match_thres,fine_match_threshold=model.fine_match_thres,max_coarse=model.max_coarse_matches,max_fine=model.max_fine_matches,refine_iters=model.refine_iters))
    return original
def geometry_protocol():
    from reliability_v1.evaluation import canonical_protocol
    return canonical_protocol()
def epipolar(d,p0,p1):
    protocol=geometry_protocol();sys.path.insert(0,str(Path(protocol['hybrid_root']).parent))
    from hybrid_match.evaluation.metrics import symmetric_epipolar_distance
    gt=d['_gt']
    return symmetric_epipolar_distance(p0*np.asarray(gt['plan0']['scale_input_to_orig_xy']),p1*np.asarray(gt['plan1']['scale_input_to_orig_xy']),gt['K0'],gt['K1'],gt['T_0to1'])
