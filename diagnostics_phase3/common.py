"""Shared Phase 3 contracts; no training or checkpoint writes."""
from pathlib import Path
import os,sys,json,time,hashlib,datetime,collections,math,csv,io
import numpy as np
ROOT=Path('/home/bupt/projects/SLiM-master');SOURCE=ROOT/'runs/phase1_20261115T004137'
CONFIG=ROOT/'configs/reliability_phase2_20261115.json'
EXPECTED='1dc74dff202052885202e401ecf8fe44c40714b5877f0f547320aad86661808a'
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT))
from reliability_v1.runtime import atomic_json,append_json,sha,digest,now,state_hash
from reliability_v1.comparison import ranks,spearman,sparsification,parent_errors,final_source_errors,summarize_rows,scene_bootstrap
METHODS=('head','lr','mconf')
FRACTIONS=np.arange(1,11)/10
class GateFailure(RuntimeError):pass
def read(p):return json.loads(Path(p).read_text())
def stamp():return now()
def pairname(pid):return hashlib.sha256(pid.encode()).hexdigest()[:24]
def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,float) and not math.isfinite(x):return None
    return x
def save(p,x):atomic_json(p,clean(x))
def text(p,s):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp')
    with q.open('w',encoding='utf-8') as f:f.write(s);f.flush();os.fsync(f.fileno())
    os.replace(q,p)
def csvout(p,rows):
    if not rows:text(p,'');return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    f=io.StringIO();w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
    w.writerows({k:json.dumps(clean(v),ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else v for k,v in r.items()} for r in rows)
    text(p,f.getvalue())
def npzout(p,arrays):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.npz.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def npzload(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def records(kind='validation'):
    rows=read(SOURCE/('fixed_final_epe.json' if kind=='epe' else 'fixed_validation.json'))['records']
    if len(rows)!=(100 if kind=='epe' else 750) or len({r['pair_id'] for r in rows})!=len(rows):raise GateFailure('Fixed manifest count/identity mismatch')
    if any(str(r['scene']) in ('0015','0022') for r in rows):raise GateFailure('Forbidden test scene')
    return rows
def selected():
    pointer=read(SOURCE/'formal_train/best.json')
    if pointer['identity_sha']!=EXPECTED or sha(SOURCE/'formal_train'/pointer['payload'])!=pointer['sha256']:raise GateFailure('Best checkpoint identity/checksum mismatch')
    import torch
    obj=torch.load(SOURCE/'formal_train'/pointer['payload'],map_location='cpu',weights_only=False)
    if digest(obj['identity'])!=EXPECTED:raise GateFailure('Embedded checkpoint identity mismatch')
    lrpath=Path(obj['extra']['lr_path'])
    if sha(lrpath)!=obj['extra']['lr_sha']:raise GateFailure('Best LR checksum mismatch')
    return pointer,obj,read(lrpath),lrpath
def cache(record,kind='epe'):
    folder=SOURCE/'final_epe' if kind=='epe' else SOURCE/'formal_train'/('validation_%02d'%read(SOURCE/'formal_train/status.json')['state']['best_epoch'])
    src=read(folder/'identity.json');meta=read(folder/(pairname(record['pair_id'])+'.json'))
    p=folder/meta['payload']
    if p.parent.resolve()!=folder.resolve() or meta['pair_id']!=record['pair_id'] or meta['identity_sha256']!=digest(src) or sha(p)!=meta['sha256']:raise GateFailure('Phase 2 cache identity/checksum mismatch')
    pred=npzload(p);raw=pred['raw_id']
    if len(raw)!=len(np.unique(raw)):raise GateFailure('Duplicate raw parent IDs')
    for m in METHODS:
        finite(pred[m],m)
        if len(pred[m])!=len(raw):raise GateFailure('Score parent alignment')
    if kind=='epe':
        finite(pred['final_source_input'],'source coordinates');finite(pred['final_target_input'],'target coordinates')
        finite(pred['child_epe'][pred['child_valid'].astype(bool)],'valid EPE')
    return pred,dict(payload=str(p),sha256=meta['sha256'],receipt=str(folder/(pairname(record['pair_id'])+'.json')))
def finite(a,label):
    if not np.isfinite(a).all():raise FloatingPointError('Nonfinite '+label)
def base(r):return dict(pair_id=r['pair_id'],scene=str(r['scene']),direction=r.get('direction','0to1'))
def positions(raw,ids):
    look={int(p):i for i,p in enumerate(raw)}
    if len(look)!=len(raw) or any(int(i) not in look for i in ids):raise GateFailure('Parent identity association failed')
    return np.asarray([look[int(i)] for i in ids],np.int64)
def classify(record,y,scores,mask=None):
    y=np.asarray(y,np.int8);mask=(y>=0) if mask is None else np.asarray(mask,bool)&(y>=0)
    return dict(**base(record),methods={m:ranks(y[mask],np.asarray(s)[mask]) for m,s in scores.items()},
        total=len(y),valid=int(mask.sum()),positive=int((y[mask]==1).sum()),ignore=int((~mask).sum()))
def errorrow(record,error,scores,ids):
    finite(error,'valid metric errors')
    return dict(**base(record),valid=len(error),methods={m:dict(Spearman=spearman(s,error),**sparsification(s,error,ids)) for m,s in scores.items()})
def summary(rows,metrics,methods=METHODS,comparisons=None):
    methods=tuple(methods)
    if methods==METHODS:return summarize_rows(rows,list(metrics))
    groups={}
    for r in rows:
        for m in methods:
            for k in metrics:
                v=r['methods'][m].get(k)
                if v is not None:groups.setdefault(r['scene'],{}).setdefault(m,{}).setdefault(k,[]).append(v)
    scenes={s:{m:{k:float(np.mean(v)) for k,v in a.items()} for m,a in b.items()} for s,b in groups.items()}
    means={m:{k:float(np.mean(v)) if (v:=[a[m][k] for a in scenes.values() if k in a.get(m,{})]) else None for k in metrics} for m in methods}
    comparisons=comparisons if comparisons is not None else [(m,'mconf') for m in methods if m!='mconf']
    return dict(pair_count=len(rows),scene_count=len(set(r['scene'] for r in rows)),parent_or_child_count=sum(r.get('valid',0) for r in rows),
        scene_equal=means,per_scene=scenes,undefined_pair_counts={m:{k:sum(r['methods'][m].get(k) is None for r in rows) for k in metrics} for m in methods},
        defined_scene_counts={m:{k:sum(k in a.get(m,{}) for a in scenes.values()) for k in metrics} for m in methods},
        paired_differences={a+'-'+b:{k:scene_bootstrap(rows,k,a,b,2000,2026) for k in metrics} for a,b in comparisons})
def advantage_change(before,after,metric='score_AUROC',baseline='mconf'):
    lookup={(r['pair_id'],r.get('direction','0to1')):r for r in before};common=[]
    for r in after:
        q=lookup.get((r['pair_id'],r.get('direction','0to1')))
        if not q:continue
        vals=[x['methods'][m].get(metric) for x in [q,r] for m in ['head',baseline]]
        if any(x is None for x in vals):continue
        d0=vals[0]-vals[1];d1=vals[2]-vals[3]
        common.append(dict(**base(r),methods={'after':{metric:d1},'before':{metric:d0}}))
    out=scene_bootstrap(common,metric,'after','before',2000,2026)
    out['per_scene_difference']={s:float(np.mean([r['methods']['after'][metric]-r['methods']['before'][metric] for r in common if r['scene']==s])) for s in sorted({r['scene'] for r in common})}
    out['common_pair_ids']=[r['pair_id'] for r in common];out['definition']='(head-baseline)_after - (head-baseline)_before, on common defined pairs only'
    return out
def metric_csv(p,summaries):
    rows=[]
    for name,s in summaries.items():
        for method,vals in s['scene_equal'].items():
            for metric,val in vals.items():rows.append(dict(analysis=name,method=method,metric=metric,value=val,pairs=s['pair_count'],undefined_pairs=s['undefined_pair_counts'][method][metric],defined_scenes=s['defined_scene_counts'][method][metric]))
    csvout(p,rows)
def distributions(values):
    a=np.asarray(values,float);finite(a,'distribution')
    return dict(count=len(a),mean=float(a.mean()) if len(a) else None,std=float(a.std()) if len(a) else None,
        quantiles=dict(zip(['min','p01','p05','p25','p50','p75','p95','p99','max'],np.quantile(a,[0,.01,.05,.25,.5,.75,.95,.99,1]).tolist())) if len(a) else None)
def curve_stats(err,score,ids):
    err=np.asarray(err,float);score=np.asarray(score,float);finite(err,'curve error');finite(score,'curve score')
    order=np.lexsort((ids,-score))
    out=[]
    for fraction in FRACTIONS:
        n=int(np.ceil(fraction*len(err)));e=err[order[:n]]
        out.append(dict(fraction=float(fraction),count=n,mean=float(e.mean()) if n else None,median=float(np.median(e)) if n else None,
            **{'precision_lt_'+str(t):float((e<t).mean()) if n else None for t in [1,2,3]}))
    return out
def curve_summary(rows,methods,exclude_scene=None):
    rows=[r for r in rows if r['scene']!=exclude_scene];result={}
    for method in methods:
        per_scene={}
        for scene in sorted({r['scene'] for r in rows}):
            rs=[r for r in rows if r['scene']==scene];cs=[]
            for i,f in enumerate(FRACTIONS):
                a=[r['curves'][method][i] for r in rs if r['curves'][method][i]['count']>0]
                cs.append(dict(fraction=float(f),pairs=len(a),**{k:float(np.mean([v[k] for v in a])) if a else None for k in ['mean','median','precision_lt_1','precision_lt_2','precision_lt_3']}))
            per_scene[scene]=cs
        avg=[]
        for i,f in enumerate(FRACTIONS):
            a=[cs[i] for cs in per_scene.values() if cs[i]['mean'] is not None]
            avg.append(dict(fraction=float(f),defined_scenes=len(a),**{k:float(np.mean([v[k] for v in a])) if a else None for k in ['mean','median','precision_lt_1','precision_lt_2','precision_lt_3']}))
        result[method]=dict(scene_equal=avg,per_scene=per_scene)
    return dict(methods=result,pairs=len(rows),excluded_scene=exclude_scene,
        aggregation='Within-pair statistic, then pair mean within scene, then scene equal; median column is scene-equal mean of pair medians.')
def heartbeat(run,stage,**values):save(Path(run)/stage/'heartbeat.json',dict(time_beijing=now(),unix=time.time(),pid=os.getpid(),stage=stage,**values))
def committed(run,stage,record):
    path=Path(run)/stage/'pairs'/(pairname(record['pair_id'])+'.json')
    if not path.exists():return None
    receipt=read(path)
    if receipt['pair_id']!=record['pair_id'] or receipt['direction']!=record.get('direction','0to1') or receipt['frozen_identity_sha']!=EXPECTED:raise GateFailure('Diagnostic committed identity mismatch')
    for item in receipt['files'].values():
        target=path.parent/item['name']
        if target.parent.resolve()!=path.parent.resolve() or sha(target)!=item['sha256']:raise GateFailure('Diagnostic committed checksum mismatch')
    return receipt
def commit(run,stage,record,result,arrays=None,source_refs=None):
    folder=Path(run)/stage/'pairs';folder.mkdir(parents=True,exist_ok=True)
    prior=committed(run,stage,record)
    if prior:return prior
    name=pairname(record['pair_id']);files={}
    resultpath=folder/(name+'.result.json');save(resultpath,result);files['result']=dict(name=resultpath.name,sha256=sha(resultpath))
    if arrays is not None:
        npz=folder/(name+'.npz');npzout(npz,arrays);files['arrays']=dict(name=npz.name,sha256=sha(npz))
    receipt=dict(status='committed',**base(record),frozen_identity_sha=EXPECTED,files=files,source_refs=source_refs,
        producer_sha256={p.name:sha(p) for p in (ROOT/'diagnostics_phase3').glob('*.py')},time_beijing=now())
    save(folder/(name+'.json'),receipt);return receipt
def result_of(run,stage,receipt):return read(Path(run)/stage/'pairs'/receipt['files']['result']['name'])
def arrays_of(run,stage,receipt):return npzload(Path(run)/stage/'pairs'/receipt['files']['arrays']['name'])
def start_runtime(run,stage):
    import torch
    from reliability_v1.runtime import Runtime,load_checkpoint
    cfg=read(CONFIG);runtime=Runtime(cfg,Path(run)/stage/'runtime')
    runtime.model.texture_threshold=read(SOURCE/'texture_calibration.json')['threshold']
    ident=read(Path(run)/'frozen_identity.json')
    obj=load_checkpoint(SOURCE/'formal_train/best.json',runtime.head,None,ident,restore_rng=False)
    runtime.head.eval().requires_grad_(False)
    # No optimizer or checkpoint entry is exposed to diagnostic code.
    del runtime.optimizer
    runtime._diagnostic_head_hash=state_hash(runtime.head)
    lrpath=Path(obj['extra']['lr_path'])
    if sha(lrpath)!=obj['extra']['lr_sha']:raise GateFailure('Selected LR checksum changed')
    return runtime,read(SOURCE/'lr_standardization.json'),read(lrpath)
def check_runtime(runtime):
    runtime.check_frozen()
    if state_hash(runtime.head)!=runtime._diagnostic_head_hash or any(p.grad is not None or p.requires_grad for p in runtime.head.parameters()):raise AssertionError('Frozen diagnostic head changed')
    return dict(backbone_and_BN_unchanged=True,head_unchanged=True,optimizer_updates=0,checkpoint_writes=0,calls=runtime.calls)
def forward_scores(runtime,data,stats,fit):
    import torch
    from reliability_v1.scalar_lr import scalar_features,lr_logits
    from src.texture_budget import local_texture
    with torch.no_grad():head=runtime.logits(data).cpu().numpy()
    conf=data['coarse_confidence'].cpu().numpy()
    x=scalar_features(conf,data['coarse_descriptor0'].float().norm(dim=-1).cpu().numpy(),data['coarse_descriptor1'].float().norm(dim=-1).cpu().numpy())
    texture,count=local_texture(data['image0'],data['valid_mask0'],data['hw0_c'])
    low=((texture[0]<=runtime.model.texture_threshold)&data['mask0'].flatten()&(count[0]>0))[data['i_idx_c']].cpu().numpy()
    out=dict(raw_id=data['coarse_match_id'].cpu().numpy(),head=head,lr=lr_logits(x,stats,fit),mconf=conf,low=low)
    for m in METHODS:finite(out[m],m)
    return out
