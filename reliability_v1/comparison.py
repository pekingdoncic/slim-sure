"""Three-method ranking and final-coordinate EPE, paired by physical scene."""
import hashlib,json,math,os,sqlite3
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from .runtime import atomic_json,sha,digest
METHODS=("head","lr","mconf")

def ranks(y,s):
    y=np.asarray(y,np.int8);s=np.asarray(s,np.float64)
    if len(y)!=len(s) or not np.isfinite(s).all() or not np.isin(y,[0,1]).all():raise ValueError("Invalid ranking inputs")
    if len(y)<2 or len(np.unique(y))<2:return dict(score_AUROC=None,AP=None,undefined_reason="single_class_or_no_valid_parents")
    order=np.argsort(-s,kind="stable");y=y[order];s=s[order]
    ends=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(y)-1];tp=np.cumsum(y,dtype=float)[ends];fp=ends+1-tp
    recall=tp/tp[-1];fpr=fp/fp[-1]
    return dict(score_AUROC=float(np.trapz(np.r_[0,recall],np.r_[0,fpr])),
                AP=float(np.sum(np.diff(np.r_[0,recall])*tp/(tp+fp))),undefined_reason=None)

def spearman(score,error):
    s=np.asarray(score,float);e=np.asarray(error,float)
    if len(s)!=len(e) or not np.isfinite(s).all() or not np.isfinite(e).all():raise ValueError("Invalid EPE scores")
    if len(s)<2 or np.ptp(s)==0 or np.ptp(e)==0:return None
    return float(np.corrcoef(rankdata(s),rankdata(e))[0,1])

def sparsification(score,error,parent_ids):
    s=np.asarray(score,float);e=np.asarray(error,float);ids=np.asarray(parent_ids)
    if not (len(s)==len(e)==len(ids)) or not np.isfinite(s).all() or not np.isfinite(e).all() or (e<0).any():raise ValueError("Invalid sparsification inputs")
    fractions=np.arange(1,11)/10
    if not len(e):return dict(AUSE=None,fractions=fractions.tolist(),curve=None,oracle=None)
    order=np.lexsort((ids,-s));oracle_order=np.lexsort((ids,e))
    counts=np.ceil(fractions*len(e)).astype(int)
    curve=[float(e[order[:n]].mean()) for n in counts];oracle=[float(e[oracle_order[:n]].mean()) for n in counts]
    return dict(AUSE=float(np.trapz(np.asarray(curve)-oracle,fractions)),fractions=fractions.tolist(),
                counts=counts.tolist(),curve=curve,oracle=oracle,units="target_input_pixels",integration_interval=[.1,1.])

def parent_errors(parent_ids,child_parent_ids,child_error,child_valid):
    parents=np.asarray(parent_ids,np.int64);cp=np.asarray(child_parent_ids,np.int64);err=np.asarray(child_error,float);valid=np.asarray(child_valid,bool)
    if len(np.unique(parents))!=len(parents) or not(len(cp)==len(err)==len(valid)):raise ValueError("Invalid parent-child arrays")
    if not np.isin(cp,parents).all():raise ValueError("Child refers to absent raw parent")
    good=valid & np.isfinite(err)
    out_ids=[];out_err=[];no_output=multiple=without_valid=0
    for pid in parents:
        all_child=cp==pid;ok=all_child&good
        no_output+=not bool(all_child.any());multiple+=int(all_child.sum())>1
        without_valid+=bool(all_child.any()) and not bool(ok.any())
        if ok.any():out_ids.append(int(pid));out_err.append(float(np.median(err[ok])))
    return dict(parent_ids=np.asarray(out_ids,np.int64),error=np.asarray(out_err,float),
                counts=dict(original_parents=len(parents),parents_without_fine_output=int(no_output),multi_child_parents=int(multiple),
                parents_with_no_valid_final_gt=int(without_valid),invalid_final_gt_children=int((~good).sum()),
                evaluable_parents=len(out_ids),valid_fine_children=int(good.sum())),
                fine_child_mean_epe=float(err[good].mean()) if good.any() else None)

def final_source_errors(data):
    from reliability_data import project_visible,_numpy
    source=_numpy(data["fine_coord_0"]);target=_numpy(data["fine_coord_1"])
    if not np.isfinite(source).all() or not np.isfinite(target).all():raise FloatingPointError("Nonfinite native final coordinates")
    gt=project_visible(source,data)
    error=np.linalg.norm(target-gt["points1_gt_input"],axis=1)
    return error,gt["valid_gt"],gt

def scene_bootstrap(rows,metric,a,b,repeats=2000,seed=2026):
    common=[r for r in rows if r["methods"][a].get(metric) is not None and r["methods"][b].get(metric) is not None]
    groups={}
    for r in common:groups.setdefault(r["scene"],[]).append(r["methods"][a][metric]-r["methods"][b][metric])
    values=np.array([np.mean(groups[s]) for s in sorted(groups)],float);count=len(values)
    higher_better=metric not in ("AUSE","Spearman")
    signed=values if higher_better else -values
    result=dict(metric=metric,comparison=a+"-"+b,pair_count=len(common),scene_count=count,
                mean_difference=float(values.mean()) if count else None,
                win=int((signed>1e-12).sum()),tie=int((np.abs(signed)<=1e-12).sum()),loss=int((signed< -1e-12).sum()),
                ci95=None,bootstrap_replicates=repeats,seed=seed,
                limitation="single training seed; validation participates in selection; fewer than 15 scenes" if count<15 else "single training seed; validation participates in selection")
    if count>1:
        rng=np.random.default_rng(seed)
        sampled=np.array([values[rng.integers(count,size=count)].mean() for _ in range(repeats)])
        result["ci95"]=np.percentile(sampled,[2.5,97.5]).tolist()
    else:result["undefined_reason"]="at most one evaluable physical scene"
    return result

def summarize_rows(rows,metrics):
    scenes={}
    for r in rows:
        for method in METHODS:
            target=scenes.setdefault(r["scene"],{}).setdefault(method,{})
            for metric in metrics:
                value=r["methods"][method].get(metric)
                if value is not None:target.setdefault(metric,[]).append(value)
    scene_means={s:{m:{k:float(np.mean(v)) for k,v in d.items()} for m,d in methods.items()} for s,methods in scenes.items()}
    average={m:{k:float(np.mean(v)) if (v:=[d[m][k] for d in scene_means.values() if m in d and k in d[m]]) else None for k in metrics} for m in METHODS}
    undefined={m:{k:sum(r["methods"][m].get(k) is None for r in rows) for k in metrics} for m in METHODS}
    diffs={a+"-"+b:{k:scene_bootstrap(rows,k,a,b) for k in metrics} for a,b in [("head","lr"),("head","mconf"),("lr","mconf")]}
    denominators={}
    for r in rows:
        d=denominators.setdefault(r["scene"],dict(pairs=0,valid_parents=0,positive_parents=0,ignored_parents=0,original_parents=0))
        d["pairs"]+=1;d["valid_parents"]+=r.get("valid",r.get("counts",{}).get("evaluable_parents",0))
        d["positive_parents"]+=r.get("positive",0);d["ignored_parents"]+=r.get("ignore",0)
        d["original_parents"]+=r.get("total",r.get("counts",{}).get("original_parents",0))
    for d in denominators.values():
        d["positive_fraction"]=d["positive_parents"]/d["valid_parents"] if d["valid_parents"] and "score_AUROC" in metrics else None
    return dict(per_scene_denominators=denominators,scene_equal=average,per_scene=scene_means,undefined_pair_counts=undefined,paired_differences=diffs,scene_count=len(scenes),pair_count=len(rows),defined_scene_counts={m:{k:sum(k in d.get(m,{}) for d in scene_means.values()) for k in metrics} for m in METHODS})

class PooledRanks:
    def __init__(self,path):
        self.db=sqlite3.connect(path);self.db.executescript("PRAGMA temp_store=FILE;PRAGMA cache_size=-8192;CREATE TABLE IF NOT EXISTS points(pair_id TEXT,y INTEGER,head REAL,lr REAL,mconf REAL,low INTEGER);CREATE TABLE IF NOT EXISTS done(pair_id TEXT PRIMARY KEY);")
    def add(self,pid,y,scores,low):
        if self.db.execute("SELECT 1 FROM done WHERE pair_id=?",(pid,)).fetchone():return
        self.db.executemany("INSERT INTO points VALUES(?,?,?,?,?,?)",((pid,int(y[i]),*[float(scores[m][i]) for m in METHODS],int(low[i])) for i in range(len(y))))
        self.db.execute("INSERT INTO done VALUES(?)",(pid,));self.db.commit()
    def summary(self,low=False):
        where=" WHERE low=1" if low else "";n,p=self.db.execute("SELECT COUNT(*),COALESCE(SUM(y),0) FROM points"+where).fetchone();result={}
        for method in METHODS:
            if n<2 or p in (0,n):result[method]=dict(score_AUROC=None,AP=None);continue
            tp=fp=auc=ap=0.
            for score,count,pos in self.db.execute("SELECT "+method+",COUNT(*),SUM(y) FROM points"+where+" GROUP BY "+method+" ORDER BY "+method+" DESC"):
                oldtp,oldfp=tp,fp;tp+=pos;fp+=count-pos
                auc+=(fp-oldfp)*(tp+oldtp)/2;ap+=(tp-oldtp)/p*tp/(tp+fp)
            result[method]=dict(score_AUROC=auc/(p*(n-p)),AP=ap)
        return dict(parent_count=n,positive_count=p,positive_fraction=p/n if n else None,methods=result)
    def close(self):self.db.close()

def evaluate(records,predict,out,identity,kind="coarse"):
    """Real and synthetic callers use this same resumable committed-pair entry."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    source=dict(identity=identity,kind=kind,pair_ids=[r["pair_id"] for r in records])
    ip=out/"identity.json"
    if ip.exists() and json.loads(ip.read_text())!=source:raise ValueError("Evaluation identity changed")
    atomic_json(ip,source);rows=[];lowrows=[];pooled=PooledRanks(out/"pooled.sqlite") if kind=="coarse" else None
    for record in records:
        pid=record["pair_id"];name=hashlib.sha256(pid.encode()).hexdigest()[:24];commit=out/(name+".json")
        if commit.exists():
            receipt=json.loads(commit.read_text())
            if receipt["identity_sha256"]!=digest(source) or sha(out/receipt["payload"])!=receipt["sha256"]:raise ValueError("Committed evaluation changed")
            with np.load(out/receipt["payload"]) as z:pred={k:z[k] for k in z.files}
        else:
            pred=predict(record);payload=out/(name+".npz");tmp=payload.with_suffix(".tmp")
            with tmp.open("wb") as f:np.savez(f,**pred);f.flush();os.fsync(f.fileno())
            os.replace(tmp,payload)
            directory=os.open(out,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(directory)
            finally:os.close(directory)
            atomic_json(commit,dict(payload=payload.name,sha256=sha(payload),identity_sha256=digest(source),pair_id=pid,scene=record["scene"]))
        raw=pred["raw_id"];scores={m:pred[m] for m in METHODS}
        if any(len(scores[m])!=len(raw) or not np.isfinite(scores[m]).all() for m in METHODS):raise ValueError("Scores not aligned/finite")
        low=np.asarray(pred["low"],bool)
        base=dict(pair_id=pid,scene=record["scene"],direction=record.get("direction","0to1"))
        if kind=="coarse":
            y=pred["labels"];valid=y>=0
            methods={m:ranks(y[valid],scores[m][valid]) for m in METHODS}
            methods["head"]["BCE"]=float(np.mean(np.logaddexp(0,scores["head"][valid])-y[valid]*scores["head"][valid])) if valid.any() else None
            methods["lr"]["BCE"]=float(np.mean(np.logaddexp(0,scores["lr"][valid])-y[valid]*scores["lr"][valid])) if valid.any() else None
            methods["mconf"]["BCE"]=None
            rows.append(dict(**base,methods=methods,total=len(y),valid=int(valid.sum()),positive=int((y==1).sum()),ignore=int((~valid).sum())))
            mask=low&valid;lowrows.append(dict(**base,methods={m:ranks(y[mask],scores[m][mask]) for m in METHODS},valid=int(mask.sum()),positive=int((y[mask]==1).sum()),total=int(low.sum()),ignore=int((low&~valid).sum())))
            pooled.add(pid,y[valid],{m:scores[m][valid] for m in METHODS},low[valid])
        else:
            agg=parent_errors(raw,pred["child_parent_id"],pred["child_epe"],pred["child_valid"])
            lookup={int(v):i for i,v in enumerate(raw)};ids=agg["parent_ids"];positions=np.array([lookup[int(v)] for v in ids],int);errors=agg["error"]
            methods={}
            for m in METHODS:
                score=scores[m][positions];methods[m]=dict(Spearman=spearman(score,errors),**sparsification(score,errors,ids))
            child_valid=np.asarray(pred["child_valid"],bool)&np.isfinite(pred["child_epe"])
            child_pos=np.array([lookup[int(v)] for v in pred["child_parent_id"][child_valid]],int)
            child_error=pred["child_epe"][child_valid]
            child_ordinals=np.arange(len(pred["child_epe"]),dtype=np.int64)[child_valid]
            # Rank lexicographic (raw parent ID, native child ordinal), keeping parent ID primary on ties.
            ordered=np.lexsort((child_ordinals,pred["child_parent_id"][child_valid]));child_ids=np.empty(len(ordered),np.int64);child_ids[ordered]=np.arange(len(ordered))
            child_methods={m:dict(Spearman=spearman(scores[m][child_pos],child_error),**sparsification(scores[m][child_pos],child_error,child_ids)) for m in METHODS}
            rows.append(dict(**base,methods=methods,all_valid_fine_child_methods=child_methods,counts=agg["counts"],fine_child_mean_epe=agg["fine_child_mean_epe"],common_parent_sha256=digest(dict(pair_id=pid,direction=base["direction"],raw_parent_ids=ids.tolist()))))
            mask=low[positions];lowrows.append(dict(**base,methods={m:dict(Spearman=spearman(scores[m][positions][mask],errors[mask]),**sparsification(scores[m][positions][mask],errors[mask],ids[mask])) for m in METHODS},valid=int(mask.sum())))
            atomic_json(out/(name+".common_parents.json"),dict(pair_id=pid,direction=base["direction"],raw_parent_ids=ids.tolist(),sha256=digest(dict(pair_id=pid,direction=base["direction"],raw_parent_ids=ids.tolist())),counts=agg["counts"]))
        atomic_json(out/"state.json",dict(status="running",committed_pairs=len(rows),next_pair_position=len(rows),kind=kind))
    metrics=["score_AUROC","AP"] if kind=="coarse" else ["Spearman","AUSE"]
    summary=dict(status="committed",kind=kind,source=source,rows=rows,low_texture_rows=lowrows,
                 ranking=summarize_rows(rows,metrics),low_texture=summarize_rows(lowrows,metrics),AP_is_AUPRC=True)
    if pooled:
        summary.update(pooled=pooled.summary(),pooled_low_texture=pooled.summary(True))
        values=[r["methods"]["head"]["BCE"] for r in rows if r["methods"]["head"]["BCE"] is not None]
        summary["head_pair_equal_BCE"]=float(np.mean(values)) if values else None;pooled.close()
    if kind=="final_epe":
        child_rows=[dict(pair_id=r["pair_id"],scene=r["scene"],methods=r["all_valid_fine_child_methods"],
            valid=r["counts"]["valid_fine_children"]) for r in rows]
        summary["all_valid_fine_children"]=summarize_rows(child_rows,metrics)
    atomic_json(out/"summary.json",summary);atomic_json(out/"state.json",dict(status="committed",committed_pairs=len(rows),kind=kind))
    return summary
