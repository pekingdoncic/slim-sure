"""Read-only lineage and independent cache/count audits."""
from common4 import *
from geometry4 import select_children
from phase4_p2b import weighted_median
from reliability_v1.phase2_runtime import make_identity
import ast
def field_lineage(run):
    rows=[];provs=read(results(run)/'P0_field_sources.json');root_refs=read(Path(run)/'source_roots.json')['phase3_frozen_references']
    C=dict(fine='signal_fine_confidence',mconf='signal_mconf',head_logit='signal_head',low='child_low',EPE='EPE',updates='pixel_updates',gt='gt_target_input',initial='intermediate_input')
    B=dict(labels='labels',corner_residual='corner_residual',center_labels='center_labels')
    E=dict(raw_id='raw_id',cp='child_parent_id',valid='child_valid',source='final_source_input',target='final_target_input',scale0='input_to_orig0',scale1='input_to_orig1',native_final_EPE='child_epe',parent_low='low')
    for r in provs:
        for field,meta in r['fields'].items():
            group=None;formula=''
            if field in C:group='phase3_C';srcfield=C[field]
            elif field in B:group='phase3_B';srcfield=B[field]
            elif field in E:group='phase2_epe';srcfield=E[field]
            elif field=='center_residual':group='phase3_B';srcfield='center_gt, coarse1';formula='center_gt - coarse1'
            elif field=='head_p':group='phase3_C';srcfield='signal_head';formula='expit(float64(logit))'
            elif field in ['S_m','S_h']:group='phase3_C';srcfield='signal_mconf,signal_fine_confidence' if field=='S_m' else 'signal_head,signal_fine_confidence';formula='float64(mconf)*float64(fine)' if field=='S_m' else 'expit(float64(head_logit))*float64(fine)'
            elif field=='trajectory':group='phase3_C';srcfield='intermediate_input,pixel_updates';formula='initial and initial + cumsum(pixel_updates,float32)'
            elif field in ['ordinal','tie_ids','random']:group='phase2_epe';srcfield='child_parent_id + native file row order';formula={'ordinal':'arange(N)','tie_ids':'Phase3 frozen encoding of (raw parent ID,native ordinal), passed unchanged to frozen sparsification; no perturbation of score','random':"SHA256 digest({'seed':2026,'pair_id':pair_id,'purpose':'fixed_child_random_prefix'})[:8] -> default_rng permutation of arange(N); descending rank score"}[field]
            if group:
                ref=r['sources'][group]
                if group=='phase2_epe':payload=ref['payload'];sha256=ref['sha256'];receipt=ref['receipt']
                else:
                    receipt=ref['receipt'];payload=str(Path(receipt).parent/ref['files']['arrays']['name']);sha256=ref['files']['arrays']['sha256']
            elif field in ['K0','K1','T']:
                payload=str(SOURCE/'fixed_final_epe.json');sha256=root_refs[payload];receipt=str(P3/'input_refs.json');srcfield=field if field!='T' else 'pose1,pose0';formula='' if field!='T' else 'pose1 @ inverse(pose0)'
            elif field in ['hw0','hw1']:
                payload=str(P3/'A4/prediction_preparation.json');sha256=read(P3/'delivery_artifact_sha256.json')['A4/prediction_preparation.json']['sha256'];receipt=str(P3/'delivery_artifact_sha256.json');srcfield='rows[pair].valid_input_hw['+field[-1]+']';formula='Cached R832 valid image dimensions; no new image read'
            else:raise AssertionError('Unmapped lineage '+field)
            rows.append(dict(pair_id=r['pair_id'],field=field,source_field=srcfield,payload=payload,source_sha256=sha256,receipt=receipt,derivation=formula,shape=meta['shape'],dtype=meta['dtype']))
    artifact(run,'P0_field_lineage',dict(rows=rows),rows)
    return len(rows)
def main(run):
    run=Path(run);checks=[]
    if digest(make_identity(read(CONFIG),SOURCE))!=EXPECTED:raise GateFailure('GLOBAL IDENTITY MISMATCH')
    srcroots=read(run/'source_roots.json')
    if sha(srcroots['phase3_hash_receipt'])!=srcroots['phase3_hash_receipt_sha256']:raise GateFailure('Phase 3 commit root changed')
    for path,h in srcroots['phase3_frozen_references'].items():
        if sha(path)!=h:raise GateFailure('Frozen reference changed')
    for p in CODE.glob('*.py'):ast.parse(p.read_text())
    checks.append(dict(check='frozen identity, source roots and diagnostic module syntax',status='pass'))
    total=valid=0;P3_counts=collections.Counter();P4_counts=collections.Counter();reconstruction=0.;threshold_discrepancies=collections.Counter();baseline_scores=[]
    p3prep=read(run/'P3/prediction_preparation.json')['rows'] if (run/'P3/prediction_preparation.json').exists() else []
    p3by={r['pair_id']:r for r in p3prep}
    p4by={r['pair_id']:r for r in read(run/'P4/prediction_preparation.json')['rows']}
    for r in records('epe'):
        a=data(run,r);n=len(a['cp']);total+=n;valid+=int(a['valid'].sum())
        if len(a['fine'])!=n:raise GateFailure('No-GT child fine score missing')
        for threshold in [1,2,3]:
            threshold_discrepancies[str(threshold)]+=int(((a['EPE'][-1]<threshold)!=(a['native_final_EPE']<threshold))[a['valid']].sum())
        for m in ['fine','S_m','S_h','mconf','random']:
            for rate in [.8,.5]:
                ix,ties=select_children(a,m,rate);combo=m+'_'+str(int(rate*100))
                if ix is None or len(ix)!=int(np.ceil(n*rate)) or np.any(np.diff(ix)<=0):raise GateFailure('P3 count/order contract')
                actual=p3by[r['pair_id']]['metrics'][combo]
                if actual['children']!=len(ix) or actual['valid_children']!=int(a['valid'][ix].sum()):raise GateFailure('P3 denominator mismatch')
                ref=p3by[r['pair_id']]['predictions'][combo];predpath=run/'P3/predictions'/pairname(r['pair_id'])/ref['payload']
                if sha(predpath)!=ref['sha256']:raise GateFailure('P3 prediction checksum')
                pred=npzload(predpath)
                if not np.array_equal(pred['points0_orig'],a['source'][ix]*a['scale0']) or not np.array_equal(pred['points1_orig'],a['target'][ix]*a['scale1']):raise GateFailure('P3 selected coordinate/order mismatch')
                P3_counts[combo]+=len(ix)
        for t in range(4):
            target=a['initial'] if t==0 else a['initial']+np.cumsum(a['updates'].astype(np.float32),axis=0,dtype=np.float32)[t-1]
            if not np.array_equal(target,a['trajectory'][t]):raise GateFailure('P4 cached replay coordinate mismatch')
            ref=p4by[r['pair_id']]['predictions']['t'+str(t)];predpath=run/'P4/predictions'/pairname(r['pair_id'])/ref['payload']
            if sha(predpath)!=ref['sha256']:raise GateFailure('P4 prediction checksum')
            pred=npzload(predpath)
            if not np.array_equal(pred['points0_orig'],a['source']*a['scale0']) or not np.array_equal(pred['points1_orig'],target*a['scale1']):raise GateFailure('P4 replayed coordinate/order mismatch')
            P4_counts['t'+str(t)]+=len(target)
        reconstruction=max(reconstruction,float(np.max(np.abs(a['trajectory'][-1]-a['target']),initial=0)))
    if (total,valid)!=(484473,361254) or reconstruction>1e-2:raise GateFailure('Child count/trajectory tolerance failed')
    checks.append(dict(check='all 484473 children retained in selection population and 361254 valid GT children',status='pass'))
    checks.append(dict(check='P3 ceil-per-pair counts, native restoration and GT-after-selection denominators',status='pass',counts=dict(P3_counts)))
    checks.append(dict(check='P4 cached displacements reproduce all-child trajectories; t4 is historical',status='pass',reconstruction_max_px=reconstruction,counts=dict(P4_counts),reconstructed_vs_native_EPE_threshold_disagreements=dict(threshold_discrepancies)))
    p1=read(results(run)/'P1_residual_bins.json')['pooled']
    for row in p1:
        if row['parents']!=row['without_fine']+row['fine_without_valid_GT']+row['valid_final_parents']:raise AssertionError('P1 count conservation')
    checks.append(dict(check='P1 first-layer complete partition',status='pass'))
    denoms=read(run/'P2b/window_details.json');supplement=[]
    for region in ['all','low_texture']:
        windows=[v for row in denoms if row['region']==region for v in row.get('window_counts',[])]
        supplement.append(dict(region=region,all_windows=len(windows),actual_n_ge2=sum(w['n']>=2 for w in windows),
            actual_n_ge2_with_any_valid_GT=sum(w['n']>=2 and w['v']>0 for w in windows),
            common_valid_v_ge2=sum(w['v']>=2 for w in windows),single_valid=sum(w['v']==1 for w in windows),
            actual_n_ge2_single_valid=sum(w['n']>=2 and w['v']==1 for w in windows),zero_valid=sum(w['v']==0 for w in windows)))
    artifact(run,'P2b_denominator_supplement',dict(rows=supplement),supplement)
    assert weighted_median([1,2,9,9,9,9],[.5,.5,.25,.25,.25,.25])==2
    checks.append(dict(check='P2b equal-window random-mixture weighted median and denominator stratification',status='pass'))
    nlineage=field_lineage(run)
    checks.append(dict(check='each derived field has source payload, receipt, SHA256 and derivation',status='pass',rows=nlineage))
    obj=dict(status='pass',checks=checks,time_beijing=now(),model_forward_calls=0,optimizer_updates=0,checkpoint_writes=0)
    artifact(run,'independent_checks',obj);return obj
