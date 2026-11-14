"""Data freezing and real forward-only Phase 1 evidence."""
import collections,json,os,sqlite3,time
from pathlib import Path
import numpy as np
import torch
from .runtime import atomic_json,sha,digest,Runtime,state_hash,resource_sample
from .scene_protocol import PROTOCOL,freeze_plan,fixed_epe,scene_ranges,parent_indices,scene_alerts
from .scalar_lr import scalar_features
from .engineering import calibration
from reliability_data import ManifestDataset,load_pair,label_contract

class FrozenRecords:
    def __init__(self,path):
        self.path=Path(path);obj=json.loads(self.path.read_text());self.records=obj["records"]
    def __len__(self):return len(self.records)
    def __getitem__(self,i):return self.records[i]
    def scene_round_robin_indices(self,count):return list(range(min(count,len(self))))

def freeze_records(path,records,scope):
    value=dict(scope=scope,records=records,coordinate_contract=label_contract())
    value["content_sha256"]=digest(value)
    path=Path(path)
    if path.exists() and json.loads(path.read_text())!=value:raise ValueError("Frozen records differ")
    atomic_json(path,value);return value

def verify_unique_files(run):
    """Current path/open/size checks plus explicit inherited deep-audit coverage."""
    run=Path(run);audit=sqlite3.connect("file:"+str(run/"data_audit.sqlite")+"?mode=ro",uri=True)
    cache=sqlite3.connect(run/"file_presence.sqlite");cache.execute("CREATE TABLE IF NOT EXISTS checks(path TEXT PRIMARY KEY,expected_hash TEXT,signature TEXT,status TEXT,reason TEXT)")
    bad=[];known_excluded=[];n=valid=total=0;started=time.monotonic()
    for path,resolved,h,size,status,reason,shape,detail in audit.execute("SELECT * FROM files ORDER BY path"):
        n+=1;total+=size
        try:
            p=Path(path);st=p.stat();signature=[st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,str(p.resolve())]
            if st.st_size!=size or str(p.resolve())!=resolved:raise ValueError("Cached file size or symlink target changed")
            with p.open("rb") as f:f.read(1)
            result="pass" if status=="valid" else "known_excluded";why=reason;valid+=int(status=="valid")
        except (OSError,ValueError) as error:
            if isinstance(error,OSError) and error.errno in (5,6,19):raise
            signature=None;result="blocked" if status=="valid" else "known_excluded";why=repr(error)
            if status=="valid":bad.append(path)
        if status!="valid":known_excluded.append(path)
        cache.execute("INSERT OR REPLACE INTO checks VALUES(?,?,?,?,?)",(path,h,json.dumps(signature),result,why))
        if n%256==0:
            cache.commit();atomic_json(run/"prepare_heartbeat.json",dict(stage="unique_file_presence",checked=n,time=time.time()))
    cache.commit();cache.close()
    with (run/"damaged_file_references.jsonl").open("w") as f:
        for path in bad:
            for row in audit.execute("SELECT p.pair_id,p.scene,p.index_id,p.ordinal FROM pairs p JOIN frames f ON (p.frame0=f.id OR p.frame1=f.id) WHERE f.image=? OR f.depth=?",(path,path)):
                f.write(json.dumps(dict(file=path,pair_id=row[0],scene=row[1],index_id=row[2],ordinal=row[3]))+"\n")
    if known_excluded:
        import gzip
        affected={}
        for path in known_excluded:
            for fid, in audit.execute("SELECT id FROM frames WHERE image=? OR depth=?",(path,path)):affected.setdefault(fid,[]).append(path)
        with gzip.open(run/"candidate_audit.jsonl.gz","rt") as ledger,(run/"damaged_file_references.jsonl").open("a") as f:
            for line in ledger:
                item=json.loads(line)
                paths=set(affected.get(item.get("frame0"),[])+affected.get(item.get("frame1"),[]))
                for path in sorted(paths):f.write(json.dumps(dict(file=path,candidate_reference=item))+"\n")
    audit.close()
    value=dict(status="pass" if not bad else "blocked",path_size_symlink_open_checked=n,current_pass=valid,
        failed_files=bad,known_excluded_files=known_excluded,expected_historical_and_new_content_hashes_preserved=True,
        deep_decode_audit_cached_files=valid,file_sizes_sum_bytes=total,
        bytes_note="Sum of distinct audited logical file sizes, not RAM usage, decoded bytes, or this execution IO throughput",
        seconds=time.monotonic()-started)
    atomic_json(run/"file_coverage.json",value)
    if bad:raise RuntimeError("Audited file identities/presence changed; see damaged_file_references.jsonl")
    return value

def freeze_sampling(config,run):
    run=Path(run);summary=json.loads((run/"data_audit_summary.json").read_text())
    if not summary["audit_complete"] or summary["expected_indices"]!=368 or summary["committed_indices"]!=368:raise RuntimeError("Complete 368-index audit required")
    if summary["counts"].get("missing_indices",0):raise RuntimeError("Missing source index")
    train=ManifestDataset(run/"train_manifest.jsonl");val=ManifestDataset(run/"val_manifest.jsonl")
    out=run/"plans";out.mkdir(exist_ok=True)
    unique=set();plans=[]
    for epoch in range(10):
        plan,records=freeze_plan(train,out/f"epoch_{epoch:02d}.json",epoch)
        unique.update(r["pair_id"] for r in records)
        plans.append(dict(epoch=epoch,visits=len(records),fraction_of_train_pool=len(records)/len(train),cumulative_unique_pairs=len(unique),cumulative_unique_fraction=len(unique)/len(train),sha256=sha(out/f"epoch_{epoch:02d}.json")))
    _,validation=freeze_plan(val,out/"validation_selection.json",0,50,"validation")
    # Fixed validation is decoded before any formal model launch.
    checked=set();deep_started=time.monotonic()
    for j,r in enumerate(validation):
        data=load_pair(r,device="cpu")
        for side in (0,1):
            depth=data["_gt"][f"depth{side}"]
            if depth.ndim!=2 or not np.any(np.isfinite(depth)&(depth>0)):raise ValueError("Fixed validation lacks valid depth")
        checked.update(r[k] for k in ("image0","image1","depth0","depth1"));del data
        atomic_json(run/"prepare_heartbeat.json",dict(stage="validation_deep_check",position=j+1,total=len(validation),time=time.time()))
    freeze_records(run/"fixed_validation.json",validation,"fixed_validation")
    epe=fixed_epe(validation);freeze_records(run/"fixed_final_epe.json",epe,"planned_real_final_epe")
    # Fixed, score/label-independent statistics and diagnostic selections.
    calibration_records=[train[i] for i in train.scene_round_robin_indices(min(100,len(train)))]
    freeze_records(run/"statistics_train_pairs.json",calibration_records,"train_statistics_only")
    probes=[]
    for dataset,domain in [(train,"train_labels"),(val,"val_labels")]:
        _,records=freeze_plan(dataset,out/(domain+".json"),0,5,domain)
        probes.extend(records)
    freeze_records(run/"label_probe_pairs.json",probes,"forward_only_cross_scene_diagnostics")
    mapping=[]
    db=train.connection
    import re
    scene_path_counts=collections.Counter()
    for scene,image in db.execute("SELECT scene,image FROM frames WHERE status='valid'"):
        observed=re.search(r"/MegaDepth_v1/([^/]+)/",image)
        if observed is None or observed.group(1)!=scene:raise ValueError("Physical image path does not prove declared scene: "+image)
        scene_path_counts[scene]+=1
    observed_indices={}
    for index_id,scene in db.execute("SELECT index_id,scene FROM pairs GROUP BY index_id,scene"):
        observed_indices.setdefault(index_id,[]).append(scene)
    for index_id,path,source_sha,mapping_path,mapping_sha,n,status in db.execute("SELECT * FROM indices ORDER BY id"):
        scenes=observed_indices.get(index_id,[])
        # Files/frames in indexed pairs establish scene identity; duplicate-only subsets may have no first-owner rows.
        stem_scene=index_id.split("_")[0]
        if scenes and scenes!=[stem_scene]:raise ValueError("Physical scene differs from index label")
        mapping.append(dict(index=index_id,subset=index_id,scene_id=stem_scene,source_sha256=source_sha,mapping_sha256=mapping_sha,
                            evidence="source stem and actual frame/path identities; deduplicated first-owner rows",observed_pair_scenes=scenes,physical_path_frames_verified=scene_path_counts[stem_scene],candidates=n,status=status))
    atomic_json(run/"index_scene_mapping.json",mapping)
    result=dict(status="pass" if len(epe)==100 else "blocked",protocol=PROTOCOL,train_pool_pairs=len(train),val_pool_pairs=len(val),
                train_scenes=scene_ranges(train),validation_scenes=scene_ranges(val),epoch_plans=plans,
                fixed_validation_pairs=len(validation),final_epe_pairs=len(epe),validation_deep_unique_files=len(checked),
                validation_deep_seconds=time.monotonic()-deep_started,
                statistical_limit="fewer than 15 validation scenes" if len(scene_ranges(val))<15 else None,
                manifest_sha256=summary["manifest_sha256"],data_audit_sha256=summary["data_audit_sha256"])
    atomic_json(run/"sampling_freeze.json",result);train.close();val.close()
    if len(epe)!=100:raise RuntimeError("Insufficient final EPE pairs")
    return result

def geometry_review(data,labels,record):
    """Bounded independent scalar projection checks for label diagnostics."""
    gt=data["_gt"];details={};errors=[];cases=[]
    for side in (0,1):
        k=gt[f"K{side}"];pose=gt[f"pose{side}"];depth=gt[f"depth{side}"]
        rotation=pose[:3,:3]
        if not np.isfinite(k).all() or not np.isfinite(pose).all() or np.linalg.det(k)==0 or min(k[0,0],k[1,1])<=0:
            raise ValueError("Invalid diagnostic camera")
        details[str(side)]=dict(K=k.tolist(),pose=pose.tolist(),depth_shape=list(depth.shape),
            depth_finite_fraction=float(np.isfinite(depth).mean()),depth_positive_fraction=float((np.isfinite(depth)&(depth>0)).mean()),
            rotation_determinant=float(np.linalg.det(rotation)),rotation_orthogonal_error=float(np.max(np.abs(rotation@rotation.T-np.eye(3)))))
    source=data["coarse_coord_0"].cpu().numpy();target=data["coarse_coord_1"].cpu().numpy()
    reason=np.asarray(labels["reasons"]);lab=np.asarray(labels["labels"]);chosen=[]
    for code in np.unique(reason):chosen.extend(np.flatnonzero(reason==code)[:3].tolist())
    for code in (0,1):chosen.extend(np.flatnonzero(lab==code)[:3].tolist())
    for idx in sorted(set(chosen)):
        xy=source[idx].astype(float);orig=xy*np.asarray(gt["plan0"]["scale_input_to_orig_xy"])
        h,w=gt["plan0"]["orig_hw"];dmap=gt["depth0"];u,v=np.rint(orig*np.array([dmap.shape[1]/w,dmap.shape[0]/h])).astype(int)
        sample=dict(raw_parent_id=int(labels["raw_coarse_match_id"][idx]),source_input=xy.tolist(),selected_target_input=target[idx].tolist(),
                    label=int(lab[idx]),ignore_reason=int(reason[idx]),source_original=orig.tolist())
        if 0<=u<dmap.shape[1] and 0<=v<dmap.shape[0] and np.isfinite(dmap[v,u]) and dmap[v,u]>0:
            depth=float(dmap[v,u]);point=np.linalg.solve(gt["K0"],np.r_[orig,1.])*depth
            transformed=gt["T_0to1"]@np.r_[point,1.];uv=gt["K1"]@transformed[:3]
            if uv[2]!=0 and np.isfinite(uv).all():
                independent=uv[:2]/uv[2]/np.asarray(gt["plan1"]["scale_input_to_orig_xy"])
                expected=labels["points1_gt_input"][idx];delta=float(np.max(np.abs(independent-expected)));errors.append(delta)
                sample.update(source_depth=depth,independent_projected_target_input=independent.tolist(),projection_abs_error=delta,
                              target_camera_depth=float(transformed[2]),observed_target_depth=float(labels["sampled_depth1"][idx]),
                              residual_input=labels["residual_input"][idx].tolist())
        cases.append(sample)
    maximum=max(errors,default=0.)
    if maximum>1e-5:raise AssertionError("Independent projection disagrees with production contract")
    return dict(status="pass",parent_anchor_contract_checked=labels["contract_checked"],stride_xy=labels["stride_xy"].tolist(),
                raw_parent_ids_unique=len(np.unique(labels["raw_coarse_match_id"]))==len(lab),
                independent_projection_cases=len(errors),maximum_projection_abs_error=maximum,cameras=details,cases=cases,
                file_sha256=record["file_sha256"],scope="all cameras and depth arrays; bounded representatives per label/ignore reason")

def real_statistics(config,run):
    run=Path(run);records=FrozenRecords(run/"statistics_train_pairs.json")
    threshold=calibration(config,records,run)
    runtime=Runtime(config,run);runtime.model.texture_threshold=threshold["threshold"]
    before=state_hash(runtime.head);sum1=np.zeros(3);sum2=np.zeros(3);visits=parents=0;rows=[];lookup={}
    probe=FrozenRecords(run/"label_probe_pairs.json");stats_ids={r["pair_id"] for r in records.records}
    work={r["pair_id"]:r for r in records.records}
    for r in probe.records:work.setdefault(r["pair_id"],r)
    from src.texture_budget import local_texture
    for position,r in enumerate(work.values()):
        data,labels=runtime.forward(r,full=False)
        valid=labels["valid"].detach().cpu().numpy().astype(bool);y=labels["y"].detach().cpu().numpy().astype(np.float32)
        raw=data["coarse_match_id"].detach().cpu().numpy()
        x=scalar_features(data["coarse_confidence"].detach().cpu().numpy(),
                          data["coarse_descriptor0"].float().norm(dim=-1).cpu().numpy(),
                          data["coarse_descriptor1"].float().norm(dim=-1).cpu().numpy())
        selected=parent_indices(valid,raw,config["seed"],0,r["pair_id"])
        if r["pair_id"] in stats_ids and len(selected):
            sample=x[selected].astype(np.float64);sum1+=sample.mean(axis=0);sum2+=(sample*sample).mean(axis=0);visits+=1;parents+=len(selected)
        texture,count=local_texture(data["image0"],data["valid_mask0"],data["hw0_c"])
        low=((texture[0]<=threshold["threshold"])&data["mask0"].flatten()&(count[0]>0))[data["i_idx_c"]].cpu().numpy()
        row=dict(pair_id=r["pair_id"],scene=r["scene"],split=r["split"],total=len(y),valid=int(valid.sum()),positive=int((y[valid]==1).sum()),
                 ignore=int((~valid).sum()),ignore_reasons=labels["ignore_counts"],low_texture_valid=int((low&valid).sum()),
                 geometry_review=geometry_review(data,labels,r),file_identity=r["file_sha256"],plans=[data["_gt"]["plan0"],data["_gt"]["plan1"]],resources=resource_sample())
        rows.append(row)
        with (run/"real_label_probes.jsonl").open("a") as f:f.write(json.dumps(row,sort_keys=True)+"\n");f.flush()
        atomic_json(run/"prepare_heartbeat.json",dict(stage="forward_only_statistics",position=position+1,total=len(work),pair_id=r["pair_id"],formal_optimizer_updates=0,time=time.time()))
        del data,labels,x,texture,count
    if not visits:raise RuntimeError("No training supervision for standardization")
    mean=sum1/visits;variance=np.maximum(0,sum2/visits-mean*mean);std=np.sqrt(variance);scale=np.where(std>1e-12,std,1.)
    stats=dict(mean=mean.tolist(),std=std.tolist(),scale=scale.tolist(),zero_variance=bool((std<=1e-12).any()),
               zero_variance_rule="replace std <=1e-12 with scale=1",weighting="equal weight per valid training pair, shared max1024 valid parent sample",
               features=["mconf","log1p(norm0)","log1p(norm1)"],visits=visits,parents=parents,
               sample_manifest_sha256=sha(records.path),parent_sampling=PROTOCOL,formal_optimizer_updates=0)
    atomic_json(run/"lr_standardization.json",stats)
    alerts=scene_alerts(rows)
    # Flagged scenes require a recorded substantive review; do not silently mark them cleared.
    pending=[s for s,v in alerts.items() if v["review_reasons"]]
    runtime.check_frozen()
    if state_hash(runtime.head)!=before:raise AssertionError("Phase1 changed the formal head")
    result=dict(status="review_required" if pending else "pass",scenes=alerts,pending_scene_reviews=pending,
                formal_optimizer_updates=0,head_before=before,head_after=state_hash(runtime.head),
                original_state_sha=runtime.frozen_hash,threshold=threshold["threshold"],lr_standardization=stats,
                evidence_insufficient_scenes=[s for s,v in alerts.items() if v["insufficient_pairs_for_extreme_rule"]])
    atomic_json(run/"label_probe_summary.json",result)
    return result
