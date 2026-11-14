"""Exact, disk-backed score metrics; no probability claims for native rankings."""
from pathlib import Path
import json
import math
import sqlite3
import numpy as np

SCORE_NAMES = ("reliability", "coarse_confidence", "symmetric_log_norm")


def binary_metrics(labels, scores):
    """Tie-aware ROC area and average precision; constants are explicitly undefined."""
    y = np.asarray(labels, dtype=np.int8).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(y) != len(s) or not np.isin(y, [0, 1]).all() or not np.isfinite(s).all():
        raise ValueError("Invalid binary metric arrays")
    if len(y) < 2 or len(np.unique(y)) < 2 or len(np.unique(s)) < 2:
        return {"score_AUROC": None, "score_AUPRC": None,
                "undefined_reason": "empty_insufficient_or_constant_labels_or_scores"}
    order = np.argsort(-s, kind="stable")
    y, s = y[order], s[order]
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s)-1]
    tp = np.cumsum(y, dtype=np.float64)[ends]
    fp = (ends+1)-tp
    recall = tp/y.sum()
    fpr = fp/(len(y)-y.sum())
    return {"score_AUROC": float(np.trapz(np.r_[0, recall], np.r_[0, fpr])),
            "score_AUPRC": float(np.sum(np.diff(np.r_[0, recall])*tp/(tp+fp))),
            "undefined_reason": None}


def bce_from_logits(labels, logits):
    y = np.asarray(labels, dtype=np.float64)
    z = np.asarray(logits, dtype=np.float64)
    return np.maximum(z, 0)-z*y+np.log1p(np.exp(-np.abs(z)))


def normalize_prediction(pred):
    y = np.asarray(pred["labels"], dtype=np.int8).reshape(-1)
    z = np.asarray(pred["logits"], dtype=np.float64).reshape(-1)
    conf = np.asarray(pred["coarse_confidence"], dtype=np.float64).reshape(-1)
    n0 = np.asarray(pred["norm0"], dtype=np.float64).reshape(-1)
    n1 = np.asarray(pred["norm1"], dtype=np.float64).reshape(-1)
    if any(len(x) != len(y) for x in (z, conf, n0, n1)):
        raise ValueError("Parent-aligned score arrays differ in length")
    if not np.isin(y, [-1,0,1]).all():
        raise ValueError("Labels must be -1 ignore, 0 incorrect, 1 correct")
    if not all(np.isfinite(x).all() for x in (z,conf,n0,n1)) or (n0<0).any() or (n1<0).any():
        raise FloatingPointError("Nonfinite scores or invalid descriptor norms")
    # Logits are monotonic in sigmoid and avoid saturation for ranking.
    return y,z,conf,(np.log1p(n0)+np.log1p(n1))/2


class ScoreAccumulator:
    """SQLite stores O(points) scalar evidence on disk; rank scans use bounded memory."""
    def __init__(self, path):
        self.path=Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(str(self.path))
        self.db.execute("PRAGMA temp_store=FILE")
        self.db.execute("CREATE TABLE IF NOT EXISTS points (pair_id TEXT, y INTEGER, reliability REAL, coarse_confidence REAL, symmetric_log_norm REAL, bce REAL, low_texture INTEGER)")
        self.db.execute("CREATE TABLE IF NOT EXISTS pairs (pair_id TEXT PRIMARY KEY, summary TEXT)")
        self.db.commit()

    def add(self,pair_id,pred):
        if self.db.execute("SELECT 1 FROM pairs WHERE pair_id=?",(pair_id,)).fetchone():
            raise ValueError("Duplicate validation pair: "+pair_id)
        y,z,conf,norm=normalize_prediction(pred)
        valid=y>=0
        losses=bce_from_logits(y[valid],z[valid])
        low=np.asarray(pred.get("low_texture_mask",np.zeros(len(y),bool)),dtype=bool).reshape(-1)
        if len(low)!=len(y):raise ValueError("low_texture_mask must follow raw parents")
        scores=dict(zip(SCORE_NAMES,(z[valid],conf[valid],norm[valid])))
        summary={"pair_id":pair_id,"scene":str(pred.get("scene","")),"parent_count":len(y),
                 "valid_count":int(valid.sum()),"ignore_count":int((~valid).sum()),
                 "positive_count":int((y==1).sum()),"pair_bce":float(losses.mean()) if len(losses) else None,
                 "rankings":{k:binary_metrics(y[valid],s) for k,s in scores.items()},
                 "ignore_reasons":pred.get("ignore_reasons",{}),
                 "local_texture_count":int((low&valid).sum()) if "low_texture_mask" in pred else None}
        self.db.executemany("INSERT INTO points VALUES (?,?,?,?,?,?,?)",
            ((pair_id,int(a),float(b),float(c),float(d),float(e),int(f))
             for a,b,c,d,e,f in zip(y[valid],z[valid],conf[valid],norm[valid],losses,low[valid])))
        self.db.execute("INSERT INTO pairs VALUES (?,?)",(pair_id,json.dumps(summary,sort_keys=True)))
        self.db.commit()
        return summary

    def _rank(self,column,where=""):
        if column not in SCORE_NAMES:raise ValueError(column)
        n,p,distinct=self.db.execute("SELECT COUNT(*),COALESCE(SUM(y),0),COUNT(DISTINCT "+column+") FROM points "+where).fetchone()
        if n<2 or p==0 or p==n or distinct<2:
            return {"score_AUROC":None,"score_AUPRC":None,"count":n,"positive_count":p,
                    "undefined_reason":"empty_insufficient_or_constant_labels_or_scores","quantile_accuracy":[]}
        tp=fp=auc=ap=0.0
        quant=[{"count":0,"positive_count":0} for _ in range(10)]
        seen=0
        for score,count,pos in self.db.execute("SELECT "+column+",COUNT(*),SUM(y) FROM points "+where+" GROUP BY "+column+" ORDER BY "+column+" DESC"):
            prev_tp,prev_fp=tp,fp
            tp+=pos;fp+=count-pos
            auc+=(fp-prev_fp)*(tp+prev_tp)/2
            ap+=(tp-prev_tp)/p*tp/(tp+fp)
            # Tied score groups stay together, assigned by their middle rank.
            q=min(9,int(10*(seen+(count-1)/2)/n))
            quant[q]["count"]+=count;quant[q]["positive_count"]+=pos
            seen+=count
        for q,item in enumerate(quant):
            item.update(group=q,rank_order="descending",accuracy=item["positive_count"]/item["count"] if item["count"] else None)
        return {"score_AUROC":auc/(p*(n-p)),"score_AUPRC":ap,"count":n,"positive_count":p,
                "undefined_reason":None,"quantile_accuracy":quant}

    def summary(self):
        n,p,bce=self.db.execute("SELECT COUNT(*),COALESCE(SUM(y),0),AVG(bce) FROM points").fetchone()
        total_pairs=empty=parents=ignores=valid_pairs=0
        pair_bce_sum=0.0
        rank_sums={k:{"score_AUROC":0.0,"score_AUPRC":0.0,"defined_pairs":0,"undefined_pairs":0} for k in SCORE_NAMES}
        reasons={}
        texture_known=False
        for row, in self.db.execute("SELECT summary FROM pairs ORDER BY pair_id"):
            item=json.loads(row);total_pairs+=1;parents+=item["parent_count"];ignores+=item["ignore_count"]
            texture_known|=item["local_texture_count"] is not None
            if item["pair_bce"] is None:empty+=1
            else:pair_bce_sum+=item["pair_bce"];valid_pairs+=1
            for reason,count in item["ignore_reasons"].items():reasons[reason]=reasons.get(reason,0)+int(count)
            for k,v in item["rankings"].items():
                if v["score_AUROC"] is None:rank_sums[k]["undefined_pairs"]+=1
                else:
                    rank_sums[k]["defined_pairs"]+=1
                    rank_sums[k]["score_AUROC"]+=v["score_AUROC"]
                    rank_sums[k]["score_AUPRC"]+=v["score_AUPRC"]
        for v in rank_sums.values():
            for key in ("score_AUROC","score_AUPRC"):
                v[key]=v[key]/v["defined_pairs"] if v["defined_pairs"] else None
        pair_bce=pair_bce_sum/valid_pairs if valid_pairs else None
        return {"schema":"slim_reliability_score_metrics_v1","pair_bce":pair_bce,
                "pair_count":total_pairs,"empty_label_pairs":empty,
                "parent_count":parents,"valid_count":n,"positive_count":p,
                "positive_fraction":p/n if n else None,
                "ignore_count":ignores,"ignore_fraction":ignores/parents if parents else None,
                "ignore_reasons":reasons,
                "pair_equal":{"bce":pair_bce,"bce_defined_pairs":valid_pairs,"rankings":rank_sums},
                "pooled":{"bce":bce,"rankings":{k:self._rank(k) for k in SCORE_NAMES}},
                "local_texture":{"status":"committed" if texture_known else "blocked",
                    "rankings":{k:self._rank(k,"WHERE low_texture=1") for k in SCORE_NAMES} if texture_known else None},
                "ranking_note":"Native confidence and symmetric mean log1p descriptor norm are rankings, not calibrated probabilities.",
                "pair_equal_note":"Each defined pair receives equal weight; constant/empty pair ranking metrics are undefined and counted."}

    def close(self):
        self.db.execute("PRAGMA optimize")
        self.db.commit();self.db.close()


def pose_auc(errors,thresholds=(5,10,20)):
    e=np.asarray(errors,dtype=np.float64).reshape(-1)
    if not len(e) or np.isnan(e).any():raise ValueError("AUC requires a complete nonempty executed pair set")
    ordered=np.r_[0.,np.sort(e)];recall=np.linspace(0.,1.,len(ordered));result={}
    for t in thresholds:
        j=int(np.searchsorted(ordered,t))
        result["pose_AUC@"+str(t)]=float(100*np.trapz(np.r_[recall[:j],recall[max(j-1,0)]],np.r_[ordered[:j],float(t)])/t)
    return result


def paired_pose_summary(baseline,texture,bootstrap=1000,seed=2026):
    a=np.asarray(baseline,float);b=np.asarray(texture,float)
    if a.ndim!=2 or a.shape[1]!=5 or b.shape!=a.shape or not len(a) or np.isnan(a).any() or np.isnan(b).any():
        raise ValueError("Expected complete [pair,5] errors; unrun records cannot become failures")
    per_seed={method:[pose_auc(v[:,j]) for j in range(5)] for method,v in (("A0",a),("A_texture",b))}
    keys=list(per_seed["A0"][0]);aggregate={}
    for method,rows in per_seed.items():
        aggregate[method]={k:{"mean":float(np.mean([r[k] for r in rows])),"std":float(np.std([r[k] for r in rows],ddof=0))} for k in keys}
    rng=np.random.RandomState(seed);deltas={k:[] for k in keys}
    for _ in range(bootstrap):
        ix=rng.randint(0,len(a),len(a))
        av=[pose_auc(a[ix,j]) for j in range(5)];bv=[pose_auc(b[ix,j]) for j in range(5)]
        for k in keys:deltas[k].append(float(np.mean([y[k]-x[k] for x,y in zip(av,bv)])))
    return {"per_seed":per_seed,"five_seed":aggregate,
            "paired_bootstrap":{"draws":bootstrap,"seed":seed,"unit":"pair carrying all five geometry seeds",
                "delta_A_texture_minus_A0":{k:{"mean":float(np.mean(v)),"ci95":np.quantile(v,[.025,.975]).tolist()} for k,v in deltas.items()}},
            "limitation":"Development validation, one training seed; geometry seeds are not training repeats. Pair bootstrap does not model scene/shared-image dependence."}


def estimate_validation_storage(pair_count,mean_valid_parents,epochs=10,bytes_per_scalar_row=128):
    """Conservative planning estimate; replace row cost with stage-1 measured bytes.

    SQL contains only scalar scores/labels. Temporary sort reserve is twice its
    estimated size; one validation database exists at a time. Per-pair summaries
    remain for initialization plus each epoch. Never reduces manifest coverage.
    """
    if pair_count<0 or mean_valid_parents<0 or epochs<1:raise ValueError("Invalid storage estimate")
    scalar=int(pair_count*mean_valid_parents*bytes_per_scalar_row)
    summaries=int(pair_count*4096)
    return {"status":"estimated","pair_count":int(pair_count),"mean_valid_parents":float(mean_valid_parents),
        "temporary_scalar_bytes":scalar,"temporary_sort_reserve_bytes":2*scalar,
        "retained_per_validation_bytes":summaries,
        "retained_training_validations_bytes":(epochs+1)*summaries,
        "conservative_peak_bytes":3*scalar+(epochs+1)*summaries+2*1024**3,
        "required_free_margin_bytes":2*1024**3,
        "assumptions":{"bytes_per_scalar_row":bytes_per_scalar_row,"summary_bytes_per_pair":4096,
            "full_validation_count":epochs+1,"temporary_databases_retained":False},
        "note":"Estimate only. Uses actual complete validation pair count and stage-1 observed valid-parent mean; disk exhaustion must stop rather than clip pairs."}
