"""End-to-end CPU analysis smoke on an explicitly synthetic one-pair fixture."""
import os,sys
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evidential_fine.common import *
import evidential_fine.statistics as st
def run(realrun):
    realrun=Path(realrun);folder=realrun/'checks'/('statistics_pipeline_'+sha(ROOT/'evidential_fine/statistics.py')[:12]);out=folder/'results'
    if (folder/'summary.json').exists():return read(folder/'summary.json')
    out.mkdir(parents=True,exist_ok=True);save(folder/'context.json',dict(results=str(out),synthetic_test_only=True))
    r=records('epe')[0];a=get_unit(realrun,'A/validation_00',r,True);e=get_unit(realrun,'E0',r,True)
    a['native_one_target']=a['trajectory'][1];a['native_one_epe']=a['trajectory_epe'][1]
    b=dict(a);m=a['native_parent_pos'];n=len(m);offset=e['gt']-a['trajectory'][0]
    b.update(cp=a['native_cp'],ordinal=np.arange(n),source=a['native_source'],initial=a['trajectory'][0],target=a['trajectory'][0],gt=e['gt'],valid=a['native_valid'],
        inside=e['B_inside'],offset=offset,epe=np.linalg.norm(offset,axis=1),low=a['parent_low'][m],mconf=a['mconf'][m],fine=a['native_fine'],
        S_m=a['mconf'][m].astype(float)*a['native_fine'],epistemic=1/(a['native_fine'].astype(float)+1e-6),aleatoric=2/(a['native_fine'].astype(float)+1e-6),
        gt_reasons=e['gt_reasons'],mu=np.zeros((n,2)))
    b['epipolar']=st.array_epipolar(b,b['source'],b['target'])
    for route in ['A','B']:save(folder/('final_'+route)/'summary.json',dict(status='committed',synthetic_test_only=True))
    save(folder/'efficiency/summary.json',dict(summary=dict(scene_equal={'A':{'seconds':.01},'native4':{'seconds':.05}}),synthetic_test_only=True))
    oldrecords,oldget=st.records,st.get_unit
    st.records=lambda *args:[r]
    st.get_unit=lambda root,stage,record,arrays=False:a if stage=='final_A' else b
    try:result=st.analyze(folder)
    finally:st.records,st.get_unit=oldrecords,oldget
    assert result['counts']['A']['outputs']==len(a['cp']) and result['counts']['B']['outputs']==n
    assert result['quality']['B/all']['scene_equal']['t0']['lt1']==error_metrics(b['epe'][b['valid']])['lt1']
    assert result['within_window']['low_texture/common']['scene_equal']['random']['median_EPE'] is None
    support=result['filter']['B/all']['metric_denominators']['epistemic_80']['GT_evaluable_rate']
    ix=st.select(b,-b['epistemic'],.8)[0]
    assert support['pooled_denominator']==len(ix) and support['pooled_numerator']==int(b['valid'][ix].sum())
    record=dict(status='pass',scope='Explicitly synthetic one-pair CPU pipeline, no GPU/model/optimizer; not formal training/evaluation evidence.',
        statistics_sha=sha(ROOT/'evidential_fine/statistics.py'),all_quality_window_filter_branches_executed=True,zero_low_GT_undefined_preserved=True,filter_denominators_verified=True,
        time_beijing=now(),command='python -B evidential_fine/test_pipeline.py '+str(realrun))
    save(folder/'summary.json',record);print(json.dumps(record));return record
if __name__=='__main__':run(sys.argv[1])
