"""A4 paired pose metrics, whole-scene bootstrap with seeds kept together."""
from common import *
from reliability_v1.metrics import pose_auc
from pose_prepare import COMBOS,combo_name
def unit_path(plan,unit):
    d=Path(plan['output'])/unit['directory'];return d/('seed_%d.json'%unit['seed']),d/('seed_%d_commit.json'%unit['seed'])
def verified_unit(plan,key):
    u=plan['units'][key];p,c=unit_path(plan,u)
    if not p.exists() or not c.exists():return None
    meta=read(c);r=read(p)
    if meta['sha256']!=sha(p) or r['status']!='committed' or r['identity']!=meta['identity'] or r['prediction_sha256']!=u['prediction_sha256']:
        raise GateFailure('Committed geometry receipt mismatch: '+key)
    if r['identity']['pair_id']!=u['pair_id'] or r['identity']['seed']!=u['seed']:raise GateFailure('Geometry pair/seed mismatch')
    return r
def mean_auc(errors):
    return np.mean([[pose_auc(errors[:,s])[k] for k in ['pose_AUC@5','pose_AUC@10','pose_AUC@20']] for s in range(5)],axis=0)
def summarize(run,plan,recs):
    run=Path(run);out=run/'A4';combo_names=[combo_name(m,f) for m,f in COMBOS];indices={r['pair_id']:i for i,r in enumerate(recs)}
    errors={k:np.full((100,5),np.nan) for k in combo_names};solves=[];failures=collections.Counter();missing=[]
    for key in plan['formal_unit_order']:
        row=verified_unit(plan,key)
        if row is None:missing.append(key);continue
        u=plan['units'][key];value=row['pose_error_deg'] if row['solver_status']=='success' else float('inf')
        if row['solver_status']=='geometry_failure':failures[row['failure_reason']]+=1
        errors[u['combo']][indices[u['pair_id']],u['seed']]=value
        solves.append(dict(pair_id=u['pair_id'],scene=u['scene'],combo=u['combo'],method=u['method'],rate=u['rate'],seed=u['seed'],
            solver_status=row['solver_status'],pose_error_deg=row['pose_error_deg'],num_points=row['num_points'],num_inliers=row['num_inliers'],geometry_seconds=row['geometry_seconds'],backend_invoked=row['backend_invoked']))
    coverage=dict(planned_formal_units=8500,committed_units=len(solves),executed_geometric_failures=sum(failures.values()),failure_reasons=dict(failures),missing_units=len(missing),missing_unit_ids=missing)
    save(out/'geometry_coverage.json',coverage);csvout(out/'geometry_units.csv',solves)
    primary=['baseline_100','head_80','mconf_80']
    primary_complete=all(not np.isnan(errors[k]).any() for k in primary)
    if missing:
        result=dict(status='不可判定',reason='Missing geometry units cannot enter denominator',coverage=coverage,primary_complete=primary_complete,G3={'status':'不可判定'})
        save(out/'summary.json',result);return result
    zeros=pose_auc(np.zeros(5));large=pose_auc(np.full(5,100.));fail=pose_auc(np.full(5,np.inf))
    if any(v!=100. for v in zeros.values()) or any(v!=0. for v in large.values()) or any(v!=0. for v in fail.values()):raise GateFailure('Frozen AUC scale test failed')
    scale_test=dict(status='pass',function='reliability_v1.metrics.pose_auc',zeros=zeros,above_threshold=large,executed_failures=fail,scale='0-100 already; no multiplication applied')
    save(out/'AUC_scale_test.json',scale_test)
    repeat_rows=[];repeat_ok=True
    for key in plan['independent_repeat_units']:
        repeat=verified_unit(plan,key);original=verified_unit(plan,plan['units'][key]['baseline_unit'])
        if repeat is None:repeat_rows.append(dict(unit=key,status='missing'));repeat_ok=False;continue
        equal=repeat['solver_status']==original['solver_status'] and repeat['num_inliers']==original['num_inliers']
        delta=0.
        if equal and repeat['solver_status']=='success':
            delta=max(abs(repeat['pose_error_deg']-original['pose_error_deg']),float(np.max(np.abs(np.asarray(repeat['R'])-original['R']))),float(np.max(np.abs(np.asarray(repeat['t'])-original['t']))))
            equal=delta<=1e-9
        repeat_rows.append(dict(unit=key,status='pass' if equal else 'failed',maximum_absolute_deviation=delta,original_unit=plan['units'][key]['baseline_unit']))
        repeat_ok&=equal
    save(out/'baseline_independent_recheck.json',dict(status='pass' if repeat_ok else 'failed',planned_pairs=5,planned_calls=25,committed_calls=sum(x['status']!='missing' for x in repeat_rows),rows=repeat_rows))
    scene_names=sorted({r['scene'] for r in recs});scene_idx={s:np.array([i for i,r in enumerate(recs) if r['scene']==s]) for s in scene_names}
    values={k:mean_auc(a) for k,a in errors.items()};scene_values={s:{k:mean_auc(a[idx]).tolist() for k,a in errors.items()} for s,idx in scene_idx.items()}
    per_seed={k:[pose_auc(a[:,s]) for s in range(5)] for k,a in errors.items()}
    # Methods and five seeds remain coupled inside each resampled scene cluster.
    rng=np.random.default_rng(2026);boot=np.empty((2000,len(combo_names),3))
    for b in range(2000):
        picked=rng.integers(len(scene_names),size=len(scene_names));idx=np.concatenate([scene_idx[scene_names[s]] for s in picked])
        for j,k in enumerate(combo_names):boot[b,j]=mean_auc(errors[k][idx])
        if (b+1)%100==0:heartbeat(run,'A4',phase='whole-scene AUC bootstrap',completed_replicates=b+1,total_replicates=2000,committed_geometry_units=8500)
    pairs=[]
    for rate in [.9,.8,.7,.5]:
        head=combo_name('head',rate);mconf=combo_name('mconf',rate);pairs.append((head,mconf,'same_budget'))
        for method in ['head','lr','mconf','random']:pairs.append((combo_name(method,rate),'baseline_100','versus_100_percent'))
    comparisons={};table=[]
    for a,b,kind in pairs:
        ia=combo_names.index(a);ib=combo_names.index(b);delta=values[a]-values[b];ci=np.percentile(boot[:,ia]-boot[:,ib],[2.5,97.5],axis=0)
        comparisons[a+'-'+b]=dict(kind=kind,method_a=a,method_b=b,mean_difference=dict(zip(['AUC@5','AUC@10','AUC@20'],delta.tolist())),
            ci95={k:[float(ci[0,i]),float(ci[1,i])] for i,k in enumerate(['AUC@5','AUC@10','AUC@20'])},bootstrap_scene_count=15,bootstrap_repeats=2000,seed=2026,pairs=100,seeds_per_pair=5)
        for i,k in enumerate(['AUC@5','AUC@10','AUC@20']):
            table.append(dict(comparison=a+'-'+b,kind=kind,metric=k,value_a=values[a][i],value_b=values[b][i],difference=delta[i],ci95_low=ci[0,i],ci95_high=ci[1,i]))
    save(out/'pose_AUC_comparisons.json',dict(primary_aggregation='All 100 pairs pooled, compute each seed AUC then mean 5 seeds',per_seed=per_seed,
        pooled_seed_mean={k:dict(zip(['AUC@5','AUC@10','AUC@20'],v.tolist())) for k,v in values.items()},paired_comparisons=comparisons,
        per_scene=scene_values,scene_equal_sensitivity={k:np.mean([scene_values[s][k] for s in scene_names],axis=0).tolist() for k in combo_names},scale='0-100'))
    csvout(out/'pose_AUC_comparisons.csv',table)
    cap={k:np.minimum(np.median(a,axis=1),20.) for k,a in errors.items()};paired_error={}
    for a,b,kind in pairs:
        delta=cap[a]-cap[b]
        rows=[dict(**base(r),methods={a:{'capped_pose_error_deg':float(cap[a][i])},b:{'capped_pose_error_deg':float(cap[b][i])}}) for i,r in enumerate(recs)]
        paired=scene_bootstrap(rows,'capped_pose_error_deg',a,b,2000,2026)
        per_scene={s:float(delta[idx].mean()) for s,idx in scene_idx.items()}
        paired.update(metric_direction='lower is better',win=sum(v< -1e-12 for v in per_scene.values()),tie=sum(abs(v)<=1e-12 for v in per_scene.values()),loss=sum(v>1e-12 for v in per_scene.values()))
        paired_error[a+'-'+b]=dict(kind=kind,pooled_pair_mean_difference=float(delta.mean()),pair_win=int((delta< -1e-12).sum()),pair_tie=int((np.abs(delta)<=1e-12).sum()),pair_loss=int((delta>1e-12).sum()),
            per_scene=per_scene,scene_equal_bootstrap=paired,per_pair=[dict(**base(r),a=float(cap[a][i]),b=float(cap[b][i]),difference=float(delta[i])) for i,r in enumerate(recs)])
    save(out/'capped_pose_error_comparisons.json',paired_error)
    prep=read(out/'prediction_preparation.json')['rows'];coverage_rows=[];quality=[]
    for k in combo_names:
        ms=[r['metrics'][k] for r in prep];n=sum(x['children'] for x in ms);good=sum(x['epipolar_correct'] for x in ms)
        unitrows=[r for r in solves if r['combo']==k]
        quality.append(dict(combo=k,children_total_per_seed=n,children_total_over_5_seeds=n*5,epipolar_correct=good,epipolar_precision=good/n if n else None,
            inliers_total_over_5_seeds=sum(x['num_inliers'] for x in unitrows),inliers_mean_per_pair_seed=float(np.mean([x['num_inliers'] for x in unitrows])),
            low_texture_parents_before=sum(x['low_texture_before'] for x in ms),low_texture_retained=sum(x['low_texture_retained'] for x in ms),
            low_texture_retention=sum(x['low_texture_retained'] for x in ms)/sum(x['low_texture_before'] for x in ms)))
        if k=='baseline_100' or k.endswith('_80'):
            for r,x in zip(prep,ms):
                coverage_rows.append(dict(**base(r),combo=k,children=x['children'],low_texture_before=x['low_texture_before'],low_texture_retained=x['low_texture_retained'],low_texture_retention=x['low_texture_retention'],
                    coverage_R832_input=x['coverage_input'],coverage_original_pixels=x['coverage_original']))
    save(out/'filter_quality_and_counts.json',quality);csvout(out/'filter_quality_and_counts.csv',quality)
    save(out/'primary_coverage_per_pair.json',coverage_rows);csvout(out/'primary_coverage_per_pair.csv',coverage_rows)
    main=comparisons['head_80-mconf_80'];ci=main['ci95']['AUC@5']
    g3status='不可判定' if not primary_complete or not repeat_ok else '触发' if ci[0]>0 else '未触发'
    g3=dict(status=g3status,difference=main['mean_difference']['AUC@5'],ci95=ci,ci_half_width=(ci[1]-ci[0])/2,
        primary_retention=.8,all_100x5_primary_units_complete=primary_complete,baseline_independent_recheck=repeat_ok,
        expand_750_primary_only_recommended=g3status=='未触发' and (ci[1]-ci[0])/2>1.,
        expansion_executed=False)
    result=dict(status='completed',G3=g3,coverage=coverage,formal_solver_units=8500,independent_rechecks=25,pose_AUC_scale='0-100, no extra conversion',
        baseline_shared=True,solver_failures=sum(failures.values()),missing_units=0,primary_counts=[x for x in quality if x['combo'] in ['baseline_100','head_80','mconf_80','random_80']],
        paired_comparisons=main,limitations='100 validation pairs across 15 scenes; seeds stay coupled, not five-fold independent samples; non-trigger is not evidence of no effect')
    save(out/'summary.json',result);return result
