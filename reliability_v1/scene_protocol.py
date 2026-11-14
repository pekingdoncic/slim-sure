"""Frozen scene-balanced protocol. No full-pool Python materialization."""
import hashlib,json
from pathlib import Path
import numpy as np
from .runtime import atomic_json,sha,digest

PROTOCOL=dict(name="scene_balanced_without_replacement_v2",seed=2026,epochs=10,train_per_scene=100,
              validation_per_scene=50,final_epe_pairs=100,max_parent_samples=1024,
              overlap={"comparison":">","threshold":0.0},replacement_within_epoch=False)

def derived_seed(seed,*parts):
    return int.from_bytes(hashlib.sha256(json.dumps([seed,*parts],separators=(",",":")).encode()).digest()[:8],"big")

def uniform_indices(n,k,seed):
    """Floyd uniform sample: O(k) memory even when a scene has millions of pairs."""
    rng=np.random.default_rng(seed);chosen=set()
    for j in range(n-min(n,k),n):
        t=int(rng.integers(j+1));chosen.add(j if t in chosen else t)
    result=np.array(sorted(chosen),dtype=np.int64)
    rng.shuffle(result);return result

def scene_ranges(dataset):
    offset=0;result=[]
    for scene,n in dataset.connection.execute("SELECT scene,COUNT(*) FROM pairs WHERE split=? GROUP BY scene ORDER BY scene",(dataset.split,)):
        result.append(dict(scene=scene,start=offset,count=n));offset+=n
    if offset!=len(dataset):raise ValueError("Scene ranges do not cover manifest")
    return result

def plan_indices(ranges,epoch,cap=100,seed=2026,domain="train"):
    selected=[]
    for row in ranges:
        ids=uniform_indices(row["count"],cap,derived_seed(seed,domain,epoch,row["scene"]))
        selected.extend(int(row["start"]+i) for i in ids)
    rng=np.random.default_rng(derived_seed(seed,domain,epoch,"visit_order"));rng.shuffle(selected)
    return selected

def freeze_plan(dataset,out,epoch,cap=100,domain="train",seed=2026):
    ranges=scene_ranges(dataset);indices=plan_indices(ranges,epoch,cap,seed,domain)
    records=[dataset[i] for i in indices]
    value=dict(schema="scene_visit_plan_v2",scope=domain,epoch=epoch,seed=seed,source_manifest_sha256=sha(dataset.path),
               policy=PROTOCOL,scene_ranges=ranges,visits=[dict(position=j,manifest_position=i,pair_id=r["pair_id"],scene=r["scene"],direction=r["direction"]) for j,(i,r) in enumerate(zip(indices,records))])
    value["content_sha256"]=digest(value)
    path=Path(out)
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError("Frozen visit plan changed")
    else:atomic_json(path,value)
    return value,records

def fixed_epe(records,count=100):
    scenes={}
    for r in records:scenes.setdefault(r["scene"],[]).append(r)
    for v in scenes.values():v.sort(key=lambda r:r["pair_id"])
    selected=[]
    for rank in range(max(map(len,scenes.values()),default=0)):
        for scene in sorted(scenes):
            if rank<len(scenes[scene]):selected.append(scenes[scene][rank])
            if len(selected)==count:return selected
    return selected

def parent_indices(valid,raw_ids,seed,epoch,pair_id,cap=1024):
    valid=np.asarray(valid,bool);raw=np.asarray(raw_ids)
    if len(valid)!=len(raw) or len(np.unique(raw))!=len(raw):raise ValueError("Parent identity mismatch")
    ids=np.flatnonzero(valid);ids=ids[np.argsort(raw[ids],kind="stable")]
    if len(ids)>cap:ids=ids[uniform_indices(len(ids),cap,derived_seed(seed,"parents",epoch,pair_id))]
    return ids

def skip_guard(state,reason):
    state=dict(state);state["planned_visits"]=state.get("planned_visits",0)+1
    state.setdefault("skip_reasons",{});state["skip_reasons"]=dict(state["skip_reasons"])
    if reason:
        state["skip_reasons"][reason]=state["skip_reasons"].get(reason,0)+1
        if state["planned_visits"]<=100:state["first100_skips"]=state.get("first100_skips",0)+1
    state["pause"]=state.get("first100_skips",0)>20
    return state

def scene_alerts(rows):
    scenes={}
    for r in rows:
        s=scenes.setdefault(r["scene"],dict(pairs=0,positive=0,valid=0,ignore=0,total=0))
        s["pairs"]+=1
        for k in ("positive","valid","ignore","total"):s[k]+=r[k]
    for s in scenes.values():
        s["positive_rate"]=s["positive"]/s["valid"] if s["valid"] else None
        s["ignore_rate"]=s["ignore"]/s["total"] if s["total"] else None
    pm=np.median([v["positive_rate"] for v in scenes.values() if v["positive_rate"] is not None]) if any(v["valid"] for v in scenes.values()) else None
    im=np.median([v["ignore_rate"] for v in scenes.values() if v["ignore_rate"] is not None]) if any(v["total"] for v in scenes.values()) else None
    for s in scenes.values():
        reasons=[]
        if s["pairs"]>=5 and s["ignore_rate"] is not None and s["ignore_rate"]>=.95:reasons.append("ignore_at_least_95pct")
        if s["pairs"]>=5 and s["valid"]>=100 and (s["positive_rate"]<.01 or s["positive_rate"]>.99):reasons.append("positive_extreme")
        for key,med in [("positive_rate",pm),("ignore_rate",im)]:
            if s[key] is not None and med is not None and abs(s[key]-med)>=.30:reasons.append(key+"_30pp_from_scene_median")
        s["review_reasons"]=reasons;s["insufficient_pairs_for_extreme_rule"]=s["pairs"]<5
    return scenes
