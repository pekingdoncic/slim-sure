from common4 import *
from phase4_p2b import weighted_median,actual_stats
from geometry4 import select_children
import ast
def main():
    checks=[]
    for p in CODE.glob('*.py'):ast.parse(p.read_text())
    checks.append('diagnostic Python modules parse')
    assert weighted_median([1,2,9,9,9,9],[.5,.5,.25,.25,.25,.25])==2
    assert weighted_median([],[]) is None
    checks.append('random mixture median uses equal window mass, not equal child mass')
    a=actual_stats([0,.5],[0,1],[0,0]);assert a['GT_evaluable_rate']==.5 and a['valid_and_correct_probability']==.25 and a['conditional_correct_rate']==.5
    checks.append('actual random conditional is ratio of expected counts')
    a=dict(cp=np.array([9,3,3,7]),ordinal=np.arange(4),fine=np.array([1.,1.,1.,.1]),valid=np.array([False,True,True,True]))
    selected,tie=select_children(a,'fine',.5);assert selected.tolist()==[1,2] and tie['cutoff_equal_score_count']==3 and tie['tie_broken_selected']==2
    a['fine']=np.array([2.,1.,.5,.1]);selected,_=select_children(a,'fine',.5);assert selected.tolist()==[0,1]
    a['fine']=np.array([2.,1.,.5]);selected,why=select_children(a,'fine',.5);assert selected is None and why['status']=='不可判定'
    checks.append('P3 selects all children before GT; missing no-GT score makes selection indeterminate; parent/ordinal cutoff ties')
    print(json.dumps(dict(status='pass',checks=checks),ensure_ascii=False))
    save(ROOT/'runs/phase4_diag_20261116T193724/contract_checks.json',dict(status='pass',checks=checks,time_beijing=now()))
if __name__=='__main__':main()
