"""Supplement E0 using committed Phase5 full native outputs, without GPU or retraining."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evidential_fine.common import *
def execute(run):
    run=Path(run);identity_gate();rows=[];proof=[];counts=dict(children=0,valid=0);max_coord=0.;max_error=0.
    for r in records('epe'):
        a=get_unit(run,'A/validation_00',r,True);c=historical('C',r);e=get_unit(run,'E0',r,True)
        if a is None:raise RuntimeError('First full Phase5 validation not complete')
        for key,other in [('native_cp','child_parent_id'),('native_source','source_input'),('native_valid','child_valid')]:
            if not np.array_equal(a[key],c[other]):raise IdentityFailure('Native reference identity mismatch '+key)
        dc=float(np.max(np.abs(a['native_target']-c['target_input']),initial=0))
        de=float(np.max(np.abs(a['native_epe'][a['native_valid']]-c['native_final_EPE'][c['child_valid']]),initial=0))
        if dc>1e-5 or de>1e-5:raise RuntimeError('Native final output/reference mismatch')
        max_coord=max(max_coord,dc);max_error=max(max_error,de);counts['children']+=len(a['native_cp']);counts['valid']+=int(a['native_valid'].sum())
        methods={f't{t}':{'lt1':error_metrics((a['native_epe'] if t==4 else e['EPE'][t])[a['native_valid']])['lt1']} for t in [0,1,4]}
        rows.append(dict(**base(r),methods=methods))
        proof.append(dict(**base(r),new_receipt_sha=sha(run/'A/validation_00/pairs'/(pairname(r['pair_id'])+'.json')),max_coordinate_deviation=dc,max_EPE_deviation=de,
            current_coordinate_sha=array_sha(a['native_target']),historical_coordinate_sha=array_sha(c['target_input'])))
    result=scene_summary(rows,['lt1'],['t0','t1','t4'])
    refpath=ROOT/'experiment_results/phase4_20261116T193724/P4_step_quality.json';ref=read(refpath)['summary']['scene_equal']
    differences={k:abs(result['scene_equal'][k]['lt1']-ref[k]['precision_lt1']) for k in ['t0','t1','t4']}
    if counts!={'children':484473,'valid':361254} or max(differences.values())>1e-6:raise RuntimeError('Phase4 exact native reference reproduction failed')
    record=dict(status='pass',time_beijing=now(),counts=counts,reference=str(refpath),reference_sha=sha(refpath),exact_phase4_reference={k:ref[k]['precision_lt1'] for k in differences},
        observed={k:result['scene_equal'][k]['lt1'] for k in differences},aggregate_deviations=differences,max_coordinate_deviation=max_coord,max_EPE_deviation=max_error,
        initial_E0_limitation='Initial E0 checked float32 reconstructed t4 against Phase3 reconstructed t4. Phase4 uses exact final native coordinates; that first aggregate differs by 1.6312831456755101e-6 and is not the required t4 reference.',
        correction='Verified exact native t4 from new Phase5 A epoch1 full validation on the same fixed100 pairs. Native parameter/BN state was checked throughout. No replay, changes to training, or T output dependency.',
        affected='B native t4 baseline gate only; A center supervision and both head selftests unchanged. B formal training must wait for this receipt.',
        source_stage='A/validation_00',proof=proof)
    p=run/'E0/native_final_reference_review.json'
    if p.exists():
        old=read(p)
        if old['observed']!=record['observed'] or old['proof']!=record['proof']:raise IdentityFailure('Existing supplemental review changed')
        return old
    save(p,record);return record
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
    out=execute(a.run);print(json.dumps({k:v for k,v in out.items() if k!='proof'},ensure_ascii=False))
