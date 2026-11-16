"""P0 verifies committed cache fields and derives CPU-only canonical pair arrays."""
from common4 import *
from c_extract import child_alignment,max_abs
from reliability_v1.geometry_worker import prediction_identity,array_sha
from pose_statistics import verified_unit,mean_auc
from c_statistics import tied_child_ids
def main(run):
    run=Path(run);recs=records('epe');prep=source_json('A4/prediction_preparation.json');prep={r['pair_id']:r for r in prep['rows']}
    protocol=source_json('A4/protocol.json');plan=source_json('A4/geometry_plan.json')
    reference=source_json('C/quality_signals.json')['all/child']['scene_equal']
    checks=[];rows=[];provenance=[];baseline=[];counts=collections.Counter()
    for ix,r in enumerate(recs):
        prior=own_receipt(run,'P0',r)
        if prior:
            z=own_read(run,'P0',r);a=own_read(run,'P0',r,True)
        else:
            refs={};issues={}
            try:p,pref=cache(r);refs['phase2_epe']=pref
            except (FileNotFoundError,KeyError,GateFailure) as exc:p=None;issues['phase2_epe']=repr(exc)
            try:
                c_receipt=committed(P3,'C',r)
                if not c_receipt:raise FileNotFoundError('Phase 3 C receipt')
                c=arrays_of(P3,'C',c_receipt);refs['phase3_C']=dict(receipt=str(P3/'C/pairs'/(pairname(r['pair_id'])+'.json')),files=c_receipt['files'])
            except (FileNotFoundError,KeyError,GateFailure) as exc:c=None;issues['phase3_C']=repr(exc)
            try:
                b_receipt=committed(P3,'B',r)
                if not b_receipt:raise FileNotFoundError('Phase 3 B receipt')
                b=arrays_of(P3,'B',b_receipt);refs['phase3_B']=dict(receipt=str(P3/'B/pairs'/(pairname(r['pair_id'])+'.json')),files=b_receipt['files'])
            except (FileNotFoundError,KeyError,GateFailure) as exc:b=None;issues['phase3_B']=repr(exc)
            a={};check={};caps={}
            if p is not None:
                for key in ['raw_id','child_parent_id','child_valid','final_source_input','final_target_input','input_to_orig0','input_to_orig1']:
                    if key not in p:issues['phase2_field_'+key]='missing'
                if not any(k.startswith('phase2_field_') for k in issues):
                    a.update(raw_id=p['raw_id'],cp=p['child_parent_id'],valid=p['child_valid'].astype(bool),source=p['final_source_input'],target=p['final_target_input'],
                        scale0=p['input_to_orig0'],scale1=p['input_to_orig1'],native_final_EPE=p['child_epe'],parent_low=p['low'].astype(bool))
            if c is not None:
                if p is not None and 'cp' in a:
                    align,why=child_alignment(dict(child_parent_id=p['child_parent_id'],final_source_input=p['final_source_input']),
                                             dict(child_parent_id=c['child_parent_id'],final_source_input=c['source_input']))
                else:align=np.arange(len(c['child_parent_id']));why='Phase3 native order; Phase2 absent'
                check['child_alignment']=why
                if align is not None:
                    if 'cp' not in a:a.update(raw_id=c['raw_id'],cp=c['child_parent_id'],valid=c['child_valid'].astype(bool),parent_low=c['low'].astype(bool))
                    if not np.array_equal(a['valid'],c['child_valid'][align].astype(bool)):raise GateFailure('GT masks differ')
                    for dst,key in [('fine','signal_fine_confidence'),('mconf','signal_mconf'),('head_logit','signal_head'),('low','child_low')]:
                        if key in c:a[dst]=c[key][align]
                        else:issues[key]='missing'
                    if 'EPE' in c:a['EPE']=c['EPE'][:,align]
                    if 'trajectory_coords' in c:a['trajectory']=c['trajectory_coords'][:,align]
                    if 'pixel_updates' in c:a['updates']=c['pixel_updates'][:,align]
                    if 'gt_target_input' in c:a['gt']=c['gt_target_input'][align]
                    if 'intermediate_input' in c:a['initial']=c['intermediate_input'][align]
                    if 'head_logit' in a:a['head_p']=expit(a['head_logit'].astype(np.float64))
                    if 'fine' in a:
                        if 'mconf' in a:a['S_m']=a['mconf'].astype(np.float64)*a['fine'].astype(np.float64)
                        if 'head_p' in a:a['S_h']=a['head_p']*a['fine'].astype(np.float64)
                    if 'updates' in a and 'target' in a:
                        reconstructed=a['initial'][None]+np.cumsum(a['updates'].astype(np.float32),axis=0,dtype=np.float32)
                        err=max_abs(reconstructed[-1],a['target'])
                        check['trajectory_max_px']=err;check['trajectory_status']='pass' if err<=1e-2 else 'failed'
                        a['trajectory']=np.concatenate([a['initial'][None],reconstructed],axis=0)
                else:issues['child_alignment']='ambiguous; no score/GT merge'
            if b is not None and 'raw_id' in a:
                idx=positions(b['raw_id'],a['raw_id'])
                for dst,key in [('labels','labels'),('corner_residual','corner_residual'),('center_labels','center_labels')]:
                    if key in b:a[dst]=b[key][idx]
                if 'center_gt' in b and 'coarse1' in b:a['center_residual']=b['center_gt'][idx]-b['coarse1'][idx]
                for side in [0,1]:
                    if 'resized_hw'+str(side) in b:a['hw'+str(side)]=b['resized_hw'+str(side)]
            if 'cp' in a:
                n=len(a['cp']);seed=int(digest(dict(seed=2026,pair_id=r['pair_id'],purpose='fixed_child_random_prefix'))[:8],16)
                permutation=np.random.default_rng(seed).permutation(n);score=np.empty(n,float);score[permutation]=np.arange(n,0,-1,dtype=float)
                a['random']=score;a['ordinal']=np.arange(n);a['tie_ids']=child_tie_ids(a['cp'])
                if 'low' not in a:a['low']=a['parent_low'][positions(a['raw_id'],a['cp'])]
                check.update(children=n,valid_children=int(a['valid'].sum()),invalid_children=int((~a['valid']).sum()),random_seed=seed)
            pred=None;baseline_rows=[]
            if p is not None and all(k in a for k in ['source','target','scale0','scale1']):
                pred=dict(points0_orig=a['source']*a['scale0'],points1_orig=a['target']*a['scale1'],K0=np.asarray(r['K0']),K1=np.asarray(r['K1']),T_0to1=np.asarray(r['pose1'])@np.linalg.inv(np.asarray(r['pose0'])))
                for k in pred:finite(pred[k],k)
                identity=prediction_identity(pred,protocol);old=prep[r['pair_id']]
                check['baseline_hash_status']='pass' if identity==old['predictions']['baseline_100']['identity'] else 'failed'
                check['baseline_array_hashes']={k:array_sha(v) for k,v in pred.items()}
                pr=old['predictions']['baseline_100'];pp=P3/'A4/predictions'/pairname(r['pair_id'])/pr['payload']
                if sha(pp)!=pr['sha256']:check['baseline_hash_status']='failed'
                refs['baseline_prediction']=dict(payload=str(pp),sha256=pr['sha256'],receipt=str(P3/'A4/prediction_preparation.json'))
                for side in [0,1]:
                    a['hw'+str(side)]=np.asarray(old['valid_input_hw'][side])
                    a['K'+str(side)]=pred['K'+str(side)]
                a['T']=pred['T_0to1']
                for seed in range(5):
                    key=pairname(r['pair_id'])+'/baseline_100/'+str(seed)
                    g=verified_unit(plan,key)
                    if g is None:check['baseline_hash_status']='missing_geometry_unit';continue
                    baseline_rows.append(dict(seed=seed,error=g['pose_error_deg'] if g['solver_status']=='success' else None,solver_status=g['solver_status'],num_inliers=g['num_inliers'],
                        file=str(Path(plan['output'])/plan['units'][key]['directory']/('seed_%d.json'%seed)),
                        receipt=str(Path(plan['output'])/plan['units'][key]['directory']/('seed_%d_commit.json'%seed))))
            caps['P1']=all(k in a for k in ['corner_residual','labels','cp','valid','EPE'])
            caps['P2']={m:all(k in a for k in [m,'valid','EPE']) and len(a[m])==len(a['valid']) for m in SCORE_NAMES}
            caps['P2b_actual']=all(k in a for k in ['fine','valid','EPE','cp']) and len(a['fine'])==len(a['cp'])
            caps['P3']={m:all(k in a for k in [m,'cp','source','target','scale0','scale1','K0','K1','T']) and len(a[m])==len(a['cp']) for m in ['fine','S_m','S_h','mconf','random']}
            caps['P4']=all(k in a for k in ['trajectory','source','target','scale0','scale1','K0','K1','T']) and check.get('trajectory_status')=='pass' and check.get('baseline_hash_status')=='pass'
            # Actual finite model scores/valid coordinates are failures, not absent-data fallbacks.
            for m in SCORE_NAMES:
                if m in a:finite(a[m],m)
            if 'EPE' in a:finite(a['EPE'][:,a['valid']],'valid cached EPE')
            z=dict(**base(r),checks=check,capabilities=caps,issues=issues,sources=refs,baseline=baseline_rows)
            own_commit(run,'P0',r,z,a)
        checks.append(z);provenance.append(dict(**base(r),sources=z['sources'],fields={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in a.items()},capabilities=z['capabilities'],issues=z['issues']))
        counts.update({k:z['checks'].get(k,0) for k in ['children','valid_children','invalid_children']})
        if all(k in a for k in ['EPE','valid','fine','head_logit','mconf']):
            v=a['valid'];ids=a['tie_ids'][v];e=a['EPE'][-1,v]
            rows.append(dict(**base(r),methods={m:dict(Spearman=spearman(a[key][v],e),**sparsification(a[key][v],e,ids)) for m,key in [('mconf','mconf'),('fine_confidence','fine'),('head','head_logit')]}))
        baseline.extend(dict(**base(r),**g) for g in z['baseline'])
        heartbeat(run,'P0',phase='verify committed fields and hashes',committed_pairs=ix+1,total_pairs=100)
    numeric=metric_rows_summary(rows,['Spearman','AUSE'],['mconf','fine_confidence','head'],[])
    deviations={m:{k:abs(numeric['scene_equal'][m][k]-reference[m][k]) if numeric['scene_equal'][m][k] is not None else None for k in ['Spearman','AUSE']} for m in reference if m in numeric['scene_equal']}
    numeric_ok=len(rows)==100 and counts['valid_children']==361254 and all(v is not None and v<=1e-6 for d in deviations.values() for v in d.values())
    berrors=np.full((100,5),np.nan)
    for i,r in enumerate(recs):
        for x in [x for x in baseline if x['pair_id']==r['pair_id']]:
            berrors[i,x['seed']]=x['error'] if x['solver_status']=='success' else np.inf
    historical=source_json('A4/pose_AUC_comparisons.json')['pooled_seed_mean']['baseline_100']
    observed=mean_auc(berrors) if not np.isnan(berrors).any() else None
    baseline_ok=all(x['checks'].get('baseline_hash_status')=='pass' for x in checks) and observed is not None
    baseline_info=dict(status='pass' if baseline_ok else '不可判定',reused_units=len(baseline),new_baseline_solves=0,
        observed_AUC=observed,historical_AUC=historical,rows=baseline,hash_checks=[dict(pair_id=x['pair_id'],status=x['checks'].get('baseline_hash_status'),hashes=x['checks'].get('baseline_array_hashes')) for x in checks])
    save(run/'P0/baseline_reuse.json',baseline_info)
    caps=dict(P1=all(x['capabilities']['P1'] for x in checks),
              P2={m:numeric_ok and all(x['capabilities']['P2'][m] for x in checks) for m in SCORE_NAMES},
              P2b_actual=numeric_ok and all(x['capabilities']['P2b_actual'] for x in checks),
              P3={m:baseline_ok and all(x['capabilities']['P3'][m] for x in checks) for m in ['fine','S_m','S_h','mconf','random']},
              P4=baseline_ok and all(x['capabilities']['P4'] for x in checks))
    summary=dict(status='completed',identity_status='pass',counts=dict(counts),numeric_reproduction_status='pass' if numeric_ok else 'failed',
        numeric_max_deviations=deviations,baseline_hash_status=baseline_info['status'],baseline_AUC=observed,
        trajectory_max_deviation_px=max(x['checks'].get('trajectory_max_px',0) for x in checks),
        capabilities=caps,head_source='Phase3 C.signal_head is a logit from Runtime.logits; head_p=scipy.special.expit(float64(logit)); no calibration/fitting',
        score_arithmetic='Cached values retained; sigmoid and products computed in float64 to avoid additional rounding.',
        EPE_scope='P1/P2/P2b/P3 use committed Phase3 reconstructed EPE_T for exact reproduction; P4 t4 uses historical native final coordinates and native_final_EPE.',
        no_new_model_forwards=0)
    artifact(run,'P0_verification',summary);artifact(run,'P0_field_sources',provenance)
    csvout(results(run)/'P0_field_sources.csv',[dict(pair_id=x['pair_id'],field=k,shape=v['shape'],dtype=v['dtype'],sources=x['sources']) for x in provenance for k,v in x['fields'].items()])
    save(run/'P0/summary.json',summary)
    return summary
