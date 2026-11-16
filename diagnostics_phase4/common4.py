"""Phase 4 CPU-only utilities; immutable Phase 2/3 sources."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path('/home/bupt/projects/SLiM-master')
sys.path.insert(0,str(ROOT/'diagnostics_phase3'))
from common import *
from scipy.special import expit
P3=ROOT/'runs/phase3_diag_20261116T100427'
CODE=ROOT/'diagnostics_phase4'
SCORE_NAMES=('mconf','head_p','fine','S_m','S_h','random')
PAIR_COMPS=[('S_m','fine'),('S_m','mconf'),('S_h','S_m'),('S_h','fine'),('head_p','mconf')]
def results(run):return Path(read(Path(run)/'context.json')['results'])
def artifact(run,name,obj=None,rows=None):
    out=results(run)
    if obj is not None:save(out/(name+'.json'),obj)
    if rows is not None:csvout(out/(name+'.csv'),rows)
def source_json(rel):
    """Trust the Phase 3 delivery hash manifest as the committed root receipt."""
    index=read(P3/'delivery_artifact_sha256.json')
    p=P3/rel
    if rel not in index:raise GateFailure('No committed aggregate checksum: '+rel)
    if sha(p)!=index[rel]['sha256']:raise GateFailure('Source aggregate checksum differs: '+rel)
    return read(p)
def own_commit(run,stage,r,value,arrays=None):
    folder=Path(run)/stage/'pairs';folder.mkdir(parents=True,exist_ok=True)
    existing=own_receipt(run,stage,r)
    if existing:return existing
    name=pairname(r['pair_id']);path=folder/(name+'.result.json');save(path,value)
    files=dict(result=dict(name=path.name,sha256=sha(path)))
    if arrays is not None:
        p=folder/(name+'.npz');npzout(p,arrays);files['arrays']=dict(name=p.name,sha256=sha(p))
    obj=dict(status='committed',**base(r),identity_sha=EXPECTED,files=files,time_beijing=now(),
             producer_sha256={p.name:sha(p) for p in CODE.glob('*.py')})
    save(folder/(name+'.json'),obj);return obj
def own_receipt(run,stage,r):
    p=Path(run)/stage/'pairs'/(pairname(r['pair_id'])+'.json')
    if not p.exists():return None
    x=read(p)
    if x['pair_id']!=r['pair_id'] or x['identity_sha']!=EXPECTED:raise GateFailure('Phase 4 receipt identity changed')
    for f in x['files'].values():
        q=p.parent/f['name']
        if q.parent!=p.parent or sha(q)!=f['sha256']:raise GateFailure('Phase 4 receipt checksum changed')
    return x
def own_read(run,stage,r,arrays=False):
    c=own_receipt(run,stage,r)
    if not c:return None
    p=Path(run)/stage/'pairs'/c['files']['arrays' if arrays else 'result']['name']
    return npzload(p) if arrays else read(p)
def data(run,r):return own_read(run,'P0',r,True)
def score_arrays(a):return {m:a[m] for m in SCORE_NAMES if m in a}
def child_tie_ids(cp):
    order=np.lexsort((np.arange(len(cp)),cp));ids=np.empty(len(cp),np.int64);ids[order]=np.arange(len(cp));return ids
def metric_rows_summary(rows,metrics,methods,comparisons):
    # Frozen scene_bootstrap for each common defined comparison.
    scenes={}
    for r in rows:
        for m in methods:
            for k in metrics:
                v=r['methods'].get(m,{}).get(k)
                if v is not None:scenes.setdefault(r['scene'],{}).setdefault(m,{}).setdefault(k,[]).append(v)
    by={s:{m:{k:float(np.mean(v)) for k,v in a.items()} for m,a in b.items()} for s,b in scenes.items()}
    means={m:{k:float(np.mean(v)) if (v:=[d[m][k] for d in by.values() if k in d.get(m,{})]) else None for k in metrics} for m in methods}
    diffs={}
    for a,b in comparisons:
        sub=[dict(r,methods={a:r['methods'].get(a,{}),b:r['methods'].get(b,{})}) for r in rows]
        diffs[a+'-'+b]={k:scene_bootstrap(sub,k,a,b,2000,2026) for k in metrics}
        for k,x in diffs[a+'-'+b].items():
            x['per_scene_difference']={s:float(np.mean([r['methods'][a][k]-r['methods'][b][k] for r in sub if r['scene']==s and r['methods'][a].get(k) is not None and r['methods'][b].get(k) is not None])) for s in sorted({r['scene'] for r in sub if r['methods'][a].get(k) is not None and r['methods'][b].get(k) is not None})}
    return dict(scene_equal=means,per_scene=by,paired_differences=diffs,pair_count=len(rows),scene_count=len(by),
        undefined_pair_counts={m:{k:sum(r['methods'].get(m,{}).get(k) is None for r in rows) for k in metrics} for m in methods},
        counts=dict(total=sum(r.get('count',0) for r in rows)))
def flat_metrics(summary,analysis):
    return [dict(analysis=analysis,method=m,metric=k,value=v,undefined_pairs=summary['undefined_pair_counts'][m][k]) for m,vs in summary['scene_equal'].items() for k,v in vs.items()]
def flat_diffs(summary,analysis):
    return [dict(analysis=analysis,comparison=comp,metric=k,**{q:x.get(q) for q in ['mean_difference','ci95','pair_count','scene_count','win','tie','loss']}) for comp,vs in summary['paired_differences'].items() for k,x in vs.items()]
def ratio(n,d):return float(n/d) if d else None
def describe(x):
    x=np.asarray(x)
    return dict(count=len(x),min=float(x.min()) if len(x) else None,max=float(x.max()) if len(x) else None,
        mean=float(x.mean()) if len(x) else None,quantiles={str(q):float(np.quantile(x,q)) for q in [.01,.05,.25,.5,.75,.95,.99]} if len(x) else {})
def scene_avg(rows,keys):
    by={}
    for s in sorted({r['scene'] for r in rows}):
        by[s]={k:float(np.mean(v)) if (v:=[r[k] for r in rows if r['scene']==s and r.get(k) is not None]) else None for k in keys}
    return dict(per_scene=by,scene_equal={k:float(np.mean(v)) if (v:=[x[k] for x in by.values() if x[k] is not None]) else None for k in keys})
def checkpoint_free(run):
    return not any(Path(run).rglob('*.pt')) and not any(Path(run).rglob('*.ckpt'))
