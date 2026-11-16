"""Small protocol tests supplement, never replace, real A0/B0/C0 gates."""
from common import *
from task_b import bins,ba_row
from c_extract import child_alignment
from reliability_v1.metrics import pose_auc
r=dict(pair_id='synthetic',scene='s0',direction='0to1')
y0=np.zeros(7,np.int8);y1=np.ones(7,np.int8);valid=np.ones(7,bool)
b=bins(np.array([0,.5,1,2,4,4.0001,7]),valid,y0,y1,[.5,1,2,4],['a','b','c','d','e'])
assert [v['count'] for v in b]==[2,1,1,1,2]
b=bins(np.array([.7,.8,.9,1.1,1.25,1.5,1.5001]),valid,y0,y1,[.8,.9,1.1,1.25,1.5],list('abcdef'),True)
assert [v['count'] for v in b]==[1,1,1,1,2,1,0]
assert ranks([1,1],[.1,.9])['score_AUROC'] is None
assert spearman([1,1],[1,2]) is None
before=[dict(**r,methods=dict(head={'score_AUROC':.9},mconf={'score_AUROC':.6})),dict(pair_id='undefined',scene='s0',methods=dict(head={'score_AUROC':None},mconf={'score_AUROC':.8}))]
after=[dict(**r,methods=dict(head={'score_AUROC':.8},mconf={'score_AUROC':.7})),dict(pair_id='undefined',scene='s0',methods=dict(head={'score_AUROC':1.},mconf={'score_AUROC':.8}))]
change=advantage_change(before,after)
assert change['pair_count']==1 and abs(change['mean_difference']+.2)<1e-12
x=dict(child_parent_id=np.array([5,5,9]),final_source_input=np.array([[1.,2.],[2.,3.],[5.,6.]]))
z=dict(child_parent_id=np.array([9,5,5]),final_source_input=x['final_source_input'][[2,1,0]])
idx,_=child_alignment(x,z);assert np.array_equal(idx,[2,1,0])
truth=np.array([0,0,1,1]);v=ba_row(r,{'corner':np.array([0,1,1,1])},truth)
assert v['methods']['corner']['BA']==.75
print(json.dumps(dict(status='pass',checks=['explicit displacement/scale bin endpoints','undefined single-class and constant-score statistics','common-pair-only advantage change','child identity reorder','balanced accuracy formula'])))
