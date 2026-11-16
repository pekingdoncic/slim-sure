"""Shared P3/P4 CPU geometry from cached arrays, with exact committed-unit resume."""
from common4 import *
from pose_prepare import occupancy
from pose_statistics import verified_unit,mean_auc
from reliability_v1.geometry_worker import prediction_identity
from reliability_v1.metrics import pose_auc
import subprocess
PYTHON='/home/bupt/miniconda3/envs/hybrid-eval/bin/python'
FILTERS=['fine','S_m','S_h','mconf','random']
def select_children(a,method,rate):
    n=len(a['cp'])
    if method not in a or len(a[method])!=n or not np.isfinite(a[method]).all():
        return None,dict(status='不可判定',reason='Missing score on complete child population; all post-filter metrics unavailable')
    k=int(np.ceil(n*rate));score=a[method]
    order=np.lexsort((a['ordinal'],a['cp'],-score))
    selected=np.sort(order[:k])
    if not k:return selected,dict(count=0,cutoff_equal_score_count=0,tie_broken_selected=0,tie_broken_excluded=0)
    cutoff=score[order[k-1]];eq=int((score==cutoff).sum());above=int((score>cutoff).sum());within=k-above;excluded=eq-within
    return selected,dict(count=k,total=n,rounding='ceil(rate*N)',cutoff=float(cutoff),cutoff_equal_score_count=eq,
        selected_at_cutoff=within,excluded_at_cutoff=excluded,tie_straddles_cutoff=bool(eq>1 and excluded>0),
        tie_broken_selected=within if eq>1 and excluded>0 else 0,tie_broken_excluded=excluded if eq>1 and excluded>0 else 0)
def epipolar(a,target,ix):
    protocol=source_json('A4/protocol.json')
    sys.path.insert(0,str(Path(protocol['hybrid_root']).parent))
    from hybrid_match.evaluation.metrics import symmetric_epipolar_distance
    return symmetric_epipolar_distance(a['source'][ix]*a['scale0'],target[ix]*a['scale1'],a['K0'],a['K1'],a['T'])
def pair_metrics(a,ix,target,error):
    n=len(ix);valid=a['valid'][ix];e=error[ix][valid];d=epipolar(a,target,ix)
    output=np.unique(a['cp']);lowids=output[a['parent_low'][positions(a['raw_id'],output)]]
    keepids=np.unique(a['cp'][ix]);nl=int(np.isin(keepids,lowids).sum())
    cov0=occupancy(a['source'][ix],a['hw0']);cov1=occupancy(target[ix],a['hw1'])
    return dict(children=n,valid_children=int(valid.sum()),GT_evaluable_rate=ratio(int(valid.sum()),n),epipolar_precision=ratio(int((d<1e-4).sum()),n),
        epipolar_correct=int((d<1e-4).sum()),nonfinite_epipolar_distances=int((~np.isfinite(d)).sum()),
        precision_lt1=float(np.mean(e<1)) if len(e) else None,precision_lt2=float(np.mean(e<2)) if len(e) else None,precision_lt3=float(np.mean(e<3)) if len(e) else None,
        mean_EPE=float(e.mean()) if len(e) else None,median_EPE=float(np.median(e)) if len(e) else None,
        epe_lt1=int((e<1).sum()),epe_lt2=int((e<2).sum()),epe_lt3=int((e<3).sum()),
        low_texture_before=len(lowids),low_texture_retained=nl,low_texture_retention=ratio(nl,len(lowids)),
        coverage0=cov0['occupancy'],coverage1=cov1['occupancy'],coverage0_detail=cov0,coverage1_detail=cov1)
def prepare(run,stage):
    run=Path(run);out=run/stage;protocol=source_json('A4/protocol.json');save(out/'protocol.json',protocol)
    recs=records('epe');units={};order=[];allprep=[];cap=read(run/'P0/summary.json')['capabilities']
    combos=[(m,rate,m+'_'+str(int(rate*100))) for rate in [.8,.5] for m in FILTERS if cap['P3'][m]] if stage=='P3' else [('trajectory',t,'t'+str(t)) for t in range(4)] if cap['P4'] else []
    for i,r in enumerate(recs):
        name=pairname(r['pair_id']);folder=out/'predictions'/name;folder.mkdir(parents=True,exist_ok=True);receipt_path=folder/'commit.json'
        if receipt_path.exists():
            receipt=read(receipt_path)
            if receipt['pair_id']!=r['pair_id'] or receipt['protocol_sha256']!=digest(protocol):raise GateFailure('Prediction identity changed')
            for q in receipt['predictions'].values():
                if sha(folder/q['payload'])!=q['sha256']:raise GateFailure('Prediction changed')
        else:
            a=data(run,r);refs={};metrics={};ties={}
            for method,val,combo in combos:
                if stage=='P3':
                    ix,tie=select_children(a,method,val)
                    if ix is None:ties[combo]=tie;continue
                    target=a['target'];error=a['EPE'][-1];ties[combo]=tie
                else:
                    ix=np.arange(len(a['cp']));target=a['trajectory'][val];error=np.linalg.norm(target.astype(np.float64)-a['gt'],axis=1)
                    finite(error[a['valid']],'valid trajectory EPE')
                points=dict(points0_orig=a['source'][ix]*a['scale0'],points1_orig=target[ix]*a['scale1'],K0=a['K0'],K1=a['K1'],T_0to1=a['T'])
                for k in ['points0_orig','points1_orig']:finite(points[k],k)
                p=folder/(combo+'.npz');npzout(p,dict(**points,pair_id=np.array(r['pair_id'])))
                refs[combo]=dict(payload=p.name,sha256=sha(p),identity=prediction_identity(points,protocol))
                metrics[combo]=pair_metrics(a,ix,target,error)
            baseline=pair_metrics(a,np.arange(len(a['cp'])),a['target'],a['native_final_EPE'])
            receipt=dict(status='committed',**base(r),predictions=refs,metrics=metrics,baseline_metrics=baseline,cutoff_ties=ties,protocol_sha256=digest(protocol),
                source_receipt=str(run/'P0/pairs'/(name+'.json')),source_receipt_sha256=sha(run/'P0/pairs'/(name+'.json')))
            save(receipt_path,receipt)
        allprep.append(receipt)
        for method,val,combo in combos:
            if combo not in receipt['predictions']:continue
            pred=receipt['predictions'][combo]
            for seed in range(5):
                key=name+'/'+combo+'/'+str(seed);order.append(key)
                units[key]=dict(**base(r),method=method,rate=val,combo=combo,seed=seed,prediction=str(folder/pred['payload']),prediction_sha256=pred['sha256'],directory=name+'/'+combo,independent_repeat=False)
        heartbeat(run,stage,phase='prepare cached geometry inputs',prepared_pairs=i+1,total_pairs=100)
    scenes=sorted({r['scene'] for r in recs});first={s:sorted([r for r in recs if r['scene']==s],key=lambda r:r['pair_id'])[0] for s in scenes}
    timing=[]
    for k in range(min(20,len(order))):
        r=first[scenes[k%len(scenes)]];combo=combos[k%len(combos)][2];key=pairname(r['pair_id'])+'/'+combo+'/'+str(k%5)
        if key in units and key not in timing:timing.append(key)
    plan=dict(output=str(out/'geometry'),protocol=str(out/'protocol.json'),protocol_sha256=digest(protocol),units=units,formal_unit_order=order,
        initial_timing_units=timing,formal_units=len(order),expected_full_units=5000 if stage=='P3' else 2000,combos=[x[2] for x in combos],
        status='frozen',source_population='All native fine children before any GT check',identity_sha=EXPECTED,stage=stage)
    file=out/'geometry_plan.json'
    if file.exists() and read(file)!=plan:raise GateFailure('Frozen geometry plan changed')
    if not file.exists():save(file,plan)
    save(out/'prediction_preparation.json',dict(status='completed',rows=allprep))
    return plan
def spawn(run,stage,plan,keys,name):
    out=Path(run)/stage;jobs=out/(name+'_jobs.json')
    if jobs.exists() and read(jobs)!=keys:raise GateFailure('Worker partition changed')
    if not jobs.exists():save(jobs,keys)
    cmd=[PYTHON,'-B','-u',str(ROOT/'diagnostics_phase3/pose_units.py'),'--plan',str(out/'geometry_plan.json'),'--jobs',str(jobs),'--worker-name',name]
    log=(out/(name+'.log')).open('ab')
    child=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1'))
    save(out/(name+'_launch.json'),dict(pid=child.pid,command=cmd,started=now(),CPU_only=True));return child,log
def wait(run,stage,jobs,phase):
    out=Path(run)/stage
    while any(c.poll() is None for c,f in jobs):
        for c,f in jobs:
            if c.poll() not in (None,0):raise RuntimeError('Geometry worker failed '+str(c.returncode))
        workers={p.stem:read(p) for p in (out/'geometry').glob('*_heartbeat.json')}
        heartbeat(run,stage,phase=phase,workers=workers);time.sleep(5)
    for c,f in jobs:
        f.close()
        if c.returncode:raise RuntimeError('Geometry worker failed '+str(c.returncode))
def execute(run,stage,plan):
    jobs=[];out=Path(run)/stage;timing=plan['initial_timing_units']
    if not timing:return
    try:
        if not all(verified_unit(plan,k) is not None for k in timing):
            jobs=[spawn(run,stage,plan,timing,'timing20')];wait(run,stage,jobs,'first 20 fixed formal units');jobs=[]
        times=[verified_unit(plan,k)['geometry_seconds'] for k in timing]
        remaining=sum(verified_unit(plan,k) is None for k in plan['formal_unit_order'])
        eta=dict(time_beijing=now(),timed_units=len(timing),mean_seconds=float(np.mean(times)),median_seconds=float(np.median(times)),max_seconds=max(times),
            remaining_units=remaining,cpu_workers=3,ETA_geometry_seconds_at_3_workers=remaining*float(np.mean(times))/3,
            unit_ids=timing,assumption='Fixed spread over scenes, methods/steps, budgets and seeds; geometry-only ETA excludes preparation, I/O and bootstrap.')
        if not (out/'first20_timing_and_ETA.json').exists():save(out/'first20_timing_and_ETA.json',eta)
        print(json.dumps(dict(event='first20_complete',**eta)),flush=True)
        rest=[k for k in plan['formal_unit_order'] if k not in set(timing)]
        for i in range(3):jobs.append(spawn(run,stage,plan,rest[i::3],'cpu_worker_'+str(i)))
        wait(run,stage,jobs,'formal geometry');jobs=[]
    finally:
        for c,f in jobs:
            if c.poll() is None:c.terminate()
            try:c.wait(timeout=10)
            except subprocess.TimeoutExpired:c.kill();c.wait()
            f.close()
def summarize_pose(run,stage,plan,comparisons):
    run=Path(run);recs=records('epe');names=['baseline_100']+plan['combos'];errors={k:np.full((100,5),np.nan) for k in names};lookup={r['pair_id']:i for i,r in enumerate(recs)}
    baseline=read(run/'P0/baseline_reuse.json')['rows']
    for row in baseline:errors['baseline_100'][lookup[row['pair_id']],row['seed']]=row['error'] if row['solver_status']=='success' else np.inf
    units=[];missing=[]
    for key in plan['formal_unit_order']:
        row=verified_unit(plan,key)
        if row is None:missing.append(key);continue
        u=plan['units'][key];error=row['pose_error_deg'] if row['solver_status']=='success' else np.inf
        errors[u['combo']][lookup[u['pair_id']],u['seed']]=error
        units.append(dict(**base(u),combo=u['combo'],seed=u['seed'],pose_error=row['pose_error_deg'],solver_status=row['solver_status'],inliers=row['num_inliers'],children=row['num_points'],seconds=row['geometry_seconds']))
    save(run/stage/'geometry_coverage.json',dict(planned=plan['formal_units'],committed=len(units),missing=len(missing),missing_unit_ids=missing,
        executed_failures=sum(r['solver_status']=='geometry_failure' for r in units),baseline_reused=len(baseline),baseline_new_solves=0))
    csvout(run/stage/'geometry_units.csv',units)
    complete=[k for k in names if not np.isnan(errors[k]).any()]
    scene_names=sorted({r['scene'] for r in recs});scene_idx={s:np.array([i for i,r in enumerate(recs) if r['scene']==s]) for s in scene_names}
    values={k:mean_auc(errors[k]) for k in complete};scene_values={s:{k:mean_auc(errors[k][idx]).tolist() for k in complete} for s,idx in scene_idx.items()}
    rng=np.random.default_rng(2026);boot=np.empty((2000,len(complete),3))
    for b in range(2000):
        idx=np.concatenate([scene_idx[scene_names[j]] for j in rng.integers(len(scene_names),size=len(scene_names))])
        for j,k in enumerate(complete):boot[b,j]=mean_auc(errors[k][idx])
        if (b+1)%100==0:heartbeat(run,stage,phase='whole-scene AUC bootstrap',completed_replicates=b+1,total_replicates=2000)
    comps={};flat=[];keys=['AUC@5','AUC@10','AUC@20']
    for a,b in comparisons:
        if a not in complete or b not in complete:comps[a+'-'+b]=dict(status='不可判定',reason='Incomplete paired 100x5 coverage');continue
        diff=values[a]-values[b];ci=np.percentile(boot[:,complete.index(a)]-boot[:,complete.index(b)],[2.5,97.5],axis=0)
        comps[a+'-'+b]=dict(status='完成',mean_difference=dict(zip(keys,diff)),ci95={k:[ci[0,i],ci[1,i]] for i,k in enumerate(keys)},pairs=100,scenes=15,seeds_per_pair=5,bootstrap_replicates=2000)
        for i,k in enumerate(keys):flat.append(dict(comparison=a+'-'+b,metric=k,difference=diff[i],ci95_low=ci[0,i],ci95_high=ci[1,i]))
    obj=dict(pooled_seed_mean={k:dict(zip(keys,v)) for k,v in values.items()},paired_comparisons=comps,per_scene=scene_values,
        scene_equal_sensitivity={k:np.mean([scene_values[s][k] for s in scene_names],axis=0).tolist() for k in complete},scale='0-100',
        primary='All 100 pairs pooled per seed; 5 seed mean. Whole scenes resampled; method and seeds remain coupled.',
        completed_combos=complete,missing_combos=[k for k in names if k not in complete])
    artifact(run,stage+'_pose',obj,flat);npzout(run/stage/'pose_errors.npz',errors)
    return obj,units
