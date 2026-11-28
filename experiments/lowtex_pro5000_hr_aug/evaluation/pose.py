from collections import defaultdict
from time import perf_counter
import cv2
import numpy as np
import torch
from utils.metrics import estimate_pose,relative_pose_error,symmetric_epipolar_distance
from .lowtex_metrics import rates, ratio


def auc(errors, thresholds=(5,10,20)):
    errors = np.r_[0.,np.sort([np.inf if x is None else x for x in errors])]
    if len(errors)<2:
        raise ValueError("empty evaluation")
    recall = np.arange(len(errors))/(len(errors)-1)
    integrate = getattr(np,"trapezoid",None) or np.trapz
    result = {}
    for t in thresholds:
        end = np.searchsorted(errors,t)
        result[f"auc@{t}"] = float(100*integrate(np.r_[recall[:end],recall[end-1]],np.r_[errors[:end],t])/t)
    return result


def pair_metrics(meta,p,q):
    start = perf_counter()
    cv2.setRNGSeed(66)
    result = dict(scene=meta["scene"],pair_id=meta["pair_id"],matches=len(p))
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        raise FloatingPointError("nonfinite matcher coordinates")
    if "H" in meta:
        plan = meta["plans"][0]
        corners = np.float64([[[0,0],[plan.w-1,0],[plan.w-1,plan.h-1],[0,plan.h-1]]])
        h = None
        if len(p)>=4:
            try:
                h,_ = cv2.findHomography(p,q,cv2.RANSAC,3.)
            except cv2.error:
                pass
        error = None
        if h is not None:
            true = cv2.perspectiveTransform(corners,meta["H"])
            err = np.linalg.norm(cv2.perspectiveTransform(corners,h)-true,axis=-1).mean()
            if np.isfinite(err):
                error=float(err)
        return dict(**result,error=error,ransac_ms=1000*(perf_counter()-start))
    k,l,t = [np.asarray(meta[x],np.float64) for x in ("K0","K1","T_0to1")]
    try:
        pose = estimate_pose(p,q,k,l,.5,.99999)
    except cv2.error:
        pose = None
    ransac = (perf_counter()-start)*1000
    err,n = None,0
    if pose is not None:
        r,v,mask = pose
        angles = relative_pose_error(t,r,v)
        if np.isfinite(angles).all():
            err,n = float(max(angles)),int(mask.sum())
    tx,ty,tz = t[:3,3]
    e = np.array([[0,-tz,ty],[tz,0,-tx],[-ty,tx,0]])@t[:3,:3]
    epi = symmetric_epipolar_distance(*[torch.from_numpy(x) for x in (p,q,e,k,l)])
    def cover(points,plan):
        inside = ((points>=0)&(points<np.array([plan.w,plan.h]))).all(1)
        bins = np.floor(points[inside]/32)
        return float(len(np.unique(bins,axis=0))/(np.ceil(plan.w/32)*np.ceil(plan.h/32)))
    return dict(**result,error=err,success=float(err is not None),inliers=n,inlier_ratio=n/len(p) if len(p) else 0.,
                epi_correct=int((epi<1e-4).sum()),coverage0=cover(p,meta["plans"][0]),coverage1=cover(q,meta["plans"][1]),ransac_ms=ransac)


def aggregate(rows,homography=False):
    def one(items):
        out = auc([r["error"] for r in items],(3,5,10) if homography else (5,10,20))
        out.update(pairs=len(items),failed=sum(r["error"] is None for r in items))
        for key in ("success","inliers","matches","inlier_ratio","coverage0","coverage1","edges"):
            if key in items[0]:
                out[key] = float(np.mean([r[key] for r in items]))
        if "epi_correct" in items[0]:
            out["epi_precision"] = ratio(sum(r["epi_correct"] for r in items),sum(r["matches"] for r in items))
        for band in ("lt","other"):
            if band in items[0]:
                summed = {k:sum(r[band][k] for r in items) for k in items[0][band]}
                out[band+"_counts"] = summed
                out.update({band+"_"+k:v for k,v in rates(summed).items()})
        return out
    groups = defaultdict(list)
    for row in rows:
        groups[row["scene"]].append(row)
    scenes = {s:one(g) for s,g in groups.items()}
    # Frozen LT selection macro-averages evaluable pairs within scene, then scenes equally.
    for s,items in groups.items():
        for band in ("lt","other"):
            if band in items[0]:
                rec = [r[band]["success"]/r[band]["den"] for r in items if r[band]["den"]]
                scenes[s][band+"_recall"] = float(np.mean(rec)) if rec else None
    macro = {}
    for k in next(iter(scenes.values())):
        if k in {"pairs","failed"} or k.endswith("_counts"):
            continue
        vals = [g[k] for g in scenes.values() if g[k] is not None]
        macro[k] = float(np.mean(vals)) if vals else None
    return dict(global_metrics=one(rows),scene_equal=macro,per_scene=scenes)
