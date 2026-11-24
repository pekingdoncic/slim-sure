"""Mechanism decomposition and coverage gates on fixed100 committed outputs."""
from report import *
from policies import select,tests
def capture_of(a):return {k[len('capture_'):]:v for k,v in a.items() if k.startswith('capture_')}
def coverage(method='C4'):
 if method!='C4':assert (OUT/'Stage2B_Report.md').exists(),'Optional work only after checkpoint2B'
 radius=int(method[1:]) if method[1:].isdigit() else 4;affine=method=='Caffine';rows=[];fallback=Counter()
 out=RUN/'coverage'/method;out.mkdir(parents=True,exist_ok=True)
 for r in recs():
  a=get('A1',r,True);p=select(capture_of(a),radius=radius,affine=affine)
  # Coverage uses ONLY the captured model scores, legal low mask, and grid coordinates.
  n={k:int(p[k].sum()) for k in ['U','H','V']};n['S']=len(p['sources'])
  assert np.array_equal(p['B'],a['plan_B'])
  change=p['C'][:,1]!=p['B'][:,1];fallback.update(p['reasons'].tolist())
  row=dict(**base(r),**n,C_changed=int(change.sum()),exactly3_valid=int((p['V']&(p['nanchor']==3)).sum()),ge4_valid=int((p['V']&(p['nanchor']>=4)).sum()))
  for name,x,y in [('H_over_U','H','U'),('V_over_U','V','U'),('V_over_S','V','S')]:row[name]=ratio(n[x],n[y])
  # Offline GT annotation cannot feed back into coverage or selection.
  s=p['sources'];gt=a['grid_correct_target'][s];v=a['grid_denominator'][s];slots=p['slots'];ct=p['C'][slots,1];bt=p['B'][slots,1]
  row.update(changed_GT=int((v&(ct!=bt)).sum()),wrong_to_right=int((v&(ct==gt)&(bt!=gt)).sum()),right_to_wrong=int((v&(ct!=gt)&(bt==gt)).sum()),wrong_to_wrong=int((v&(ct!=bt)&(ct!=gt)&(bt!=gt)).sum()),unknown_GT_changed=int((~v&(ct!=bt)).sum()))
  rows.append(row);npzout(out/(pairname(r['pair_id'])+'.npz'),p)
 total={k:sum(r[k] for r in rows) for k in ['S','U','H','V','C_changed','exactly3_valid','ge4_valid','changed_GT','wrong_to_right','right_to_wrong','wrong_to_wrong','unknown_GT_changed']}
 scenes=sorted({r['scene'] for r in rows if r['V']});reasons=[]
 if total['U']==0:reasons.append('U=0')
 if total['U'] and total['H']/total['U']<.05:reasons.append('H/U<0.05')
 if total['U'] and total['V']/total['U']<.05:reasons.append('V/U<0.05')
 if total['V']<30:reasons.append('V<30')
 if len(scenes)<3:reasons.append('V_scenes<3')
 res=dict(method=method,radius=radius,affine=affine,counts=total,ratios=dict(H_over_U=ratio(total['H'],total['U']),V_over_U=ratio(total['V'],total['U']),V_over_S=ratio(total['V'],total['S'])),valid_scenes=scenes,gate_pass=not reasons,skip_reasons=reasons,fallback=dict(fallback),per_pair=rows,per_scene={s:{k:sum(r[k] for r in rows if r['scene']==s) for k in total} for s in sorted({r['scene'] for r in rows})},coverage_does_not_access_GT=True,**pair_metadata(rows))
 save(OUT/('coverage_'+method+'.json'),res);csvwrite('coverage_'+method+'.csv',rows);return res
def source_transition(a,b,mask):
 x=a['grid_any_good'];y=b['grid_any_good'];den=mask&a['grid_denominator']
 assert np.array_equal(a['grid_denominator'],b['grid_denominator'])
 c=dict(denominator=int(den.sum()),common_success=int((den&x&y).sum()),rescued=int((den&~x&y).sum()),lost=int((den&x&~y).sum()),common_failure=int((den&~x&~y).sum()))
 c['net']=c['rescued']-c['lost'];c['delta_recall']=ratio(c['net'],c['denominator'])
 assert c['net']==int((den&y).sum())-int((den&x).sum())
 return c
def parent_stats(a,mask):
 src=a['source_grid'];good=a['good_child_counts'];pc=a['child_counts'];vc=a['valid_child_counts']
 return dict(parents=int(mask.sum()),parents_good=int((mask&(good>0)).sum()),good_children=int(good[mask].sum()),successful_source_grids=len(np.unique(src[mask&(good>0)])),no_fine=int((mask&(pc==0)).sum()),all_child_GT_invalid=int((mask&(pc>0)&(vc==0)).sum()),valid_children=int(vc[mask].sum()))
def summarize_methods(methods):
 rows=[];metrics=['recall','coverage','precision']
 for m in methods:
  for r in recs():
   x=get(m,r)
   for region,v in x['metrics'].items():rows.append(dict(**base(r),method=m,region=region,**v))
 summary={}
 for m in methods:
  summary[m]={}
  for region in ['low','nonlow']:
   rr=[x for x in rows if x['method']==m and x['region']==region];pooled={k:sum(x[k] for x in rr) for k in ['denominator','success','both_pass','correct_parent','block_denominator','block_success','valid_children','good_children','no_child','no_valid_child_GT']}
   pooled.update(recall=ratio(pooled['success'],pooled['denominator']),coverage=ratio(pooled['block_success'],pooled['block_denominator']),precision=ratio(pooled['good_children'],pooled['valid_children']))
   summary[m][region]=dict(pooled=pooled,scene_equal={k:estimate(rr,k) for k in metrics})
 return rows,summary
def compare(methods,pairs):
 rows=[];summ={}
 for after,before in pairs:
  for r in recs():
   a=get(before,r,True);b=get(after,r,True);ra=get(before,r);rb=get(after,r)
   for region,mask in [('low',a['grid_low0']),('nonlow',~a['grid_low0'])]:
    c=source_transition(a,b,mask);c.update(**base(r),comparison=after+'-'+before,region=region)
    for metric in ['coverage','precision']:
     x=rb['metrics'][region][metric];y=ra['metrics'][region][metric];c['delta_'+metric]=x-y if x is not None and y is not None else None
    rows.append(c)
  name=after+'-'+before;summ[name]={}
  for reg in ['low','nonlow']:
   rr=[r for r in rows if r['comparison']==name and r['region']==reg]
   summ[name][reg]=dict(pooled={k:sum(r[k] for r in rr) for k in ['denominator','common_success','rescued','lost','common_failure','net']},**{k:estimate(rr,'delta_'+k) for k in ['recall','coverage','precision']})
 return rows,summ
def stage2a():
 methods=['A1','B0','B','oracle'];budgets=[];change=[];tail=[];parents=[];oracles=[]
 for r in recs():
  data={m:get(m,r,True) for m in methods};a=data['A1'];p={k[len('plan_'):]:v for k,v in a.items() if k.startswith('plan_')}
  M=len(a['raw_id']);q=int(p['q']);n=int(p['r']);src=p['sources'];slots=p['slots'];gt=a['grid_correct_target'][src];v=a['grid_denominator'][src]
  inside=v&(p['targets']==gt[:,None]).any(1);b0=data['B0'];b=data['B'];o=data['oracle']
  for m,x in data.items():
   assert len(x['raw_id'])==len(np.unique(x['raw_id']))==M
   if m!='A1':
    for k in ['candidate0','candidate1','capture_P_sha','capture_sim_sha','plan_sources','plan_targets','plan_slots']:assert np.array_equal(x[k],a[k]),(m,k)
    assert np.array_equal(x['raw_id'][:M-n],a['raw_id'][:M-n])
   budgets.append(dict(**base(r),method=m,M=M,q=q,r=n,unique_coarse_pairs=len(np.unique(x['raw_id'])),protected=M-n,pool_nonempty_low_sources=int((a['capture_pool_p']>0).any(1).sum()),quota_shortfall=q-n,quota_shortfall_reason='insufficient distinct low positive-pool sources not covered by protected A' if n<q else '',sources_already_native_tail=int(p['tail_source_reused'].sum())))
  j0=b0['target_grid'][slots];jb=b['target_grid'][slots];jo=o['target_grid'][slots];D=j0!=jb;U=p['U'];doublefirst=(p['b'][:,0]==1)
  assert not np.any(D&doublefirst)
  row=dict(**base(r),S=n,U=int(U.sum()),D=int(D.sum()),D_over_S=ratio(D.sum(),n),D_over_U=ratio(D.sum(),U.sum()),D_GT=int((D&v).sum()),D_no_GT=int((D&~v).sum()),double_first=int(doublefirst.sum()),double_first_changed=0,wrong_to_right=int((D&v&(j0!=gt)&(jb==gt)).sum()),right_to_wrong=int((D&v&(j0==gt)&(jb!=gt)).sum()),wrong_to_wrong=int((D&v&(j0!=gt)&(jb!=gt)).sum()))
  for name,sel in [('changed',D&v),('all_GT',v)]:
   x=b0['grid_any_good'][src];y=b['grid_any_good'][src]
   for key,val in [('common_success',x&y),('rescued',~x&y),('lost',x&~y),('common_failure',~x&~y)]:row[name+'_'+key]=int((sel&val).sum())
  change.append(row)
  oracle=dict(**base(r),S=n,GT_valid=int(v.sum()),GT_missing=int((~v).sum()),GT_in_pool=int(inside.sum()),GT_outside_pool=int((v&~inside).sum()),GT_in_native_beyond_row5=int((v&~inside&np.isin(gt,a['candidate1'])).sum()))
  for m,x in data.items():
   if m=='A1':continue
   hits=int((v&(x['target_grid'][slots]==gt)).sum());oracle[m+'_hits']=hits;oracle[m+'_gap']=int(inside.sum())-hits
  assert oracle['oracle_hits']==oracle['GT_in_pool']
  for name,sel in [('C_valid',p['V']&v),('C_changeable',p['U']&v)]:
   oracle[name+'_GT']=int(sel.sum());oracle[name+'_pool_upper']=int((sel&inside).sum());oracle[name+'_B_hits']=int((sel&(jb==gt)).sum());oracle[name+'_C_hits']=int((sel&(p['C'][slots,1]==gt)).sum())
  oracle['ranking_gain_fraction']=ratio(oracle['B_hits']-oracle['B0_hits'],oracle['oracle_hits']-oracle['B0_hits'])
  oracles.append(oracle)
  removed=~np.isin(a['raw_id'],b0['raw_id']);added=~np.isin(b0['raw_id'],a['raw_id']);assert removed.sum()==added.sum()
  assert not removed[:M-n].any()
  tr=dict(**base(r),tail_size=n,tail_reselected=int(np.isin(a['raw_id'][M-n:],b0['raw_id']).sum()) if n else 0,true_removed=int(removed.sum()),true_added=int(added.sum()))
  for region,mask in [('low',a['grid_low0']),('nonlow',~a['grid_low0'])]:
   transition=source_transition(a,b0,mask)
   for k,val in transition.items():tr[region+'_'+k]=val
   lostgoodsrc=np.unique(a['source_grid'][removed&(a['good_child_counts']>0)&mask[a['source_grid']]])
   eligible=lostgoodsrc[a['grid_denominator'][lostgoodsrc]]
   tr[region+'_lost_good_parent_sources']=len(lostgoodsrc);tr[region+'_lost_good_parent_evaluable_sources']=len(eligible)
   tr[region+'_lost_parent_still_covered']=int(b0['grid_any_good'][eligible].sum());tr[region+'_lost_parent_really_lost']=int((~b0['grid_any_good'][eligible]).sum())
   for op,x,pm in [('removed',a,removed),('added',b0,added)]:
    stats=parent_stats(x,pm&mask[x['source_grid']]);parents.append(dict(**base(r),region=region,operation=op,**stats))
  common,ia,ib=np.intersect1d(a['raw_id'],b0['raw_id'],return_indices=True);ga=a['good_child_counts'][ia]>0;gb=b0['good_child_counts'][ib]>0
  for region,mask in [('low',a['grid_low0']),('nonlow',~a['grid_low0'])]:
   pm=mask[a['source_grid'][ia]]
   tr[region+'_common_parent_good_lost']=int((pm&ga&~gb).sum());tr[region+'_common_parent_good_gained']=int((pm&~ga&gb).sum())
   tr[region+'_common_parent_good_child_delta']=int((b0['good_child_counts'][ib][pm]-a['good_child_counts'][ia][pm]).sum())
  tail.append(tr)
 metrics,summary=summarize_methods(methods);comp,comparisons=compare(methods,[('B0','A1'),('B','B0'),('oracle','B')])
 cov=coverage('C4');ch={k:sum(r[k] for r in change) for k in ['S','U','D','D_GT','D_no_GT','double_first','double_first_changed','wrong_to_right','right_to_wrong','wrong_to_wrong','changed_common_success','changed_rescued','changed_lost','changed_common_failure']}
 ch.update(D_scenes=sorted({r['scene'] for r in change if r['D']}),D_GT_scenes=sorted({r['scene'] for r in change if r['D_GT']}),D_over_S=ratio(ch['D'],ch['S']),D_over_U=ratio(ch['D'],ch['U']))
 ch['sparse_intervention']=ch['D']<30 or len(ch['D_scenes'])<3
 oo={k:sum(x[k] for x in oracles) for k in oracles[0] if k not in ['pair_id','scene','index_id','direction','frame0','frame1','ranking_gain_fraction'] and isinstance(oracles[0][k],(int,float))}
 oo['ranking_gain_fraction']=ratio(oo['B_hits']-oo['B0_hits'],oo['oracle_hits']-oo['B0_hits'])
 tails={k:sum(r[k] for r in tail) for k in tail[0] if isinstance(tail[0][k],(int,float)) and not k.endswith('delta_recall')}
 parenttot={}
 for reg in ['low','nonlow']:
  for op in ['removed','added']:
   rr=[r for r in parents if r['region']==reg and r['operation']==op];parenttot[reg+'_'+op]={k:sum(r[k] for r in rr) for k in ['parents','parents_good','good_children','successful_source_grids','no_fine','all_child_GT_invalid','valid_children']}
 result=dict(status='completed',methods=methods,metrics=summary,comparisons=comparisons,B_vs_B0=ch,oracle_upper=oo,tail=tails,parent_cost_gain=parenttot,main_C_gate=cov,**pair_metadata(recs()))
 save(OUT/'stage2A.json',result)
 for name,x in [('equal_budget.csv',budgets),('B_vs_B0.csv',change),('A_tail_source_transitions.csv',tail),('A_tail_parent_cost_gain.csv',parents),('oracle_upper_bound.csv',oracles),('method_metrics.csv',metrics),('paired_source_transitions.csv',comp)]:csvwrite(name,x)
 lines=['# 阶段 2A：配额、排名、同池 oracle 与主 C 覆盖','',DISCLAIMER,'','A1、B0、B、oracle 均已完成 100/100 对真实细匹配与坐标精修。所有组保持每对 M 条、M 个唯一粗格对；保护槽位、预留源格、池、上游候选及完整 P/相似度哈希逐对一致。固定低纹理分母均为 7,954；零分母 pair 保留。','',
 '| 方法 | 好子点源格/固定分母 | 场景等权低纹理召回 |','|---|---:|---:|']
 for m in methods:
  x=summary[m]['low'];lines.append(f"| {m} | {x['pooled']['success']}/{x['pooled']['denominator']} | {x['scene_equal']['recall']['mean']:.6f} |")
 lines+=['','## Oracle 严格粗格上界','',f"预留 {ch['S']} 个槽位；有效中心 GT {oo['GT_valid']}；GT 在池内 {oo['GT_in_pool']}。B0/B/oracle 严格命中分别为 {oo['B0_hits']}/{oo['B_hits']}/{oo['oracle_hits']}。GT 缺失 {oo['GT_missing']}；池外 {oo['GT_outside_pool']}。",'','Oracle 细匹配召回是一次 GT 辅助观测，不是数学上界；邻格也可能产出更好的子点。oracle 不进入位姿、部署成本或阶段 3 方法排名。','',
 '## B 相对 B0 的实际干预','',f"S={ch['S']}，U={ch['U']}，D={ch['D']}；D/S={ch['D_over_S']}，D/U={ch['D_over_U']}；有 GT 的改选 {ch['D_GT']}，分布于 {len(ch['D_GT_scenes'])} 个场景。错→对 {ch['wrong_to_right']}，对→错 {ch['right_to_wrong']}，错→另一错 {ch['wrong_to_wrong']}，GT 缺失 {ch['D_no_GT']}。行列双第一 {ch['double_first']} 个均保持不变。对照干预稀疏：{ch['sparse_intervention']}。",'',
 '## B0 尾部代价与唯一源格得失','',f"A 尾部 {tails['tail_size']}，重新选回 {tails['tail_reselected']}，真正移除/新增均为 {tails['true_removed']}。低纹理唯一源格救回 {tails['low_rescued']}、丢失 {tails['low_lost']}、净增 {tails['low_net']}；非低纹理净增 {tails['nonlow_net']}。",'',
 '低/非低纹理的父点数、好子点数、成功源格数、无细点与无 GT 分别见 A_tail_parent_cost_gain.csv；仍有其他父点覆盖、真正失去覆盖及公共父点竞争变化见 A_tail_source_transitions.csv。主召回均以唯一源格救回−丢失精确核验。','',
 '## 主 C：R=4 无 GT 覆盖闸门','',f"S/U/H/V={cov['counts']['S']}/{cov['counts']['U']}/{cov['counts']['H']}/{cov['counts']['V']}；H/U={cov['ratios']['H_over_U']}，V/U={cov['ratios']['V_over_U']}，V/S={cov['ratios']['V_over_S']}；有效场景 {len(cov['valid_scenes'])}。闸门通过：{cov['gate_pass']}；原因：{cov['skip_reasons']}。",'',
 '## 配对差值','']
 for name,v in comparisons.items():
  e=v['low']['recall'];lines.append(f"- {name}：场景等权 {e['mean']:+.6f}，95% 场景聚类区间 {e['CI95']}，{e['scenes']} 场景、{e['pairs']} 可评 pair。")
 lines+=['','区间含零时，现有样本证据不足以确定增益方向。逐场景方向、非低纹理、32px 正确覆盖与有效细点精度见 stage2A.json。运行记录包含审计开销，不冒充部署计时；独立验证、位姿及三次同步计时尚未执行。','',
 f"清单 SHA：{result['manifest_sha']}；协议 SHA：{result['protocol_sha']}；选择代码哈希已逐 pair 冻结。",'',
 '续跑：cd /home/bupt/projects/SLiM-master && /home/bupt/miniconda3/envs/slim/bin/python -B diagnostics_coarse_rank/chain.py --methods C4 --background（仅 coverage_C4.json 的 gate_pass 为 true 时）。']
 md('Stage2A_Report.md','\n'.join(lines)+'\n');return result
def stage2b():
 cov=read(OUT/'coverage_C4.json');base=read(OUT/'stage2A.json')
 if cov['gate_pass']:
  methods=['A1','B','C4'];rows,stats=summarize_methods(methods);rr,comp=compare(methods,[('C4','B'),('C4','A1')])
  change=[];upper=[]
  for r in recs():
   a=get('A1',r,True);b=get('B',r,True);c=get('C4',r,True);p=select(capture_of(a));slots=p['slots'];src=p['sources'];v=a['grid_denominator'][src];gt=a['grid_correct_target'][src]
   assert np.array_equal(np.stack([c['source_grid'],c['target_grid']],1),p['C'])
   delta=c['target_grid'][slots]!=b['target_grid'][slots]
   change.append(dict(**globals()['base'](r),S=len(src),changed=int(delta.sum()),unchanged=int((~delta).sum()),GT_changed=int((delta&v).sum()),wrong_to_right=int((delta&v&(c['target_grid'][slots]==gt)).sum()),right_to_wrong=int((delta&v&(b['target_grid'][slots]==gt)).sum()),wrong_to_wrong=int((delta&v&(c['target_grid'][slots]!=gt)&(b['target_grid'][slots]!=gt)).sum()),unknown_GT=int((delta&~v).sum()),C_hits=int((v&(c['target_grid'][slots]==gt)).sum()),GT_in_pool=int((v&(p['targets']==gt[:,None]).any(1)).sum())))
  result=dict(status='completed',metrics=stats,comparisons=comp,changes=change,coverage=cov,**pair_metadata(recs()))
  csvwrite('C4_changes.csv',change);csvwrite('C4_source_transitions.csv',rr);csvwrite('C4_metrics.csv',rows)
  textlines=[]
  for name,v in comp.items():
   e=v['low']['recall'];t=v['low']['pooled'];textlines.append(f"{name}：场景等权低纹理召回差 {e['mean']:+.6f}，95% 区间 {e['CI95']}；救回 {t['rescued']}、丢失 {t['lost']}、净增 {t['net']}。")
 else:
  result=dict(status='skipped',reason=cov['skip_reasons'],coverage=cov,planned=100,completed=0,failed=0,independent_model_runs=0,time=now())
  textlines=['主 C 覆盖闸门未通过：'+', '.join(cov['skip_reasons'])+'。整组细前向跳过；不把跳过写成 C=B 或零收益。']
 save(OUT/'stage2B.json',result)
 md('Stage2B_Report.md','\n\n'.join(['# 阶段 2B：主配置邻域支持',DISCLAIMER,*textlines,
 f"S/U/H/V={cov['counts']['S']}/{cov['counts']['U']}/{cov['counts']['H']}/{cov['counts']['V']}，有效场景 {len(cov['valid_scenes'])}。无干预与逐槽回退 pair 均留在固定 100 对总体内。",
 '所有新增父点均重新经过原生细匹配和坐标精修。GT 仅用于评价与隔离 oracle；主 C 只访问受保护的行列双第一锚点、网格坐标和 P。完整数据、配对区间与逐场景方向见 stage2B.json。',
 '阶段 3 尚未执行，独立集模型运行数 0。下一步展示阶段 0 冻结的完整候选清单哈希和成本，等待用户明确确认；可选项只在此检查点已交付后开始。',
 '恢复命令：cd /home/bupt/projects/SLiM-master && /home/bupt/miniconda3/envs/slim/bin/python -B diagnostics_coarse_rank/mechanisms.py --stage 2b'])+'\n');return result
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--stage',required=True);a=p.parse_args()
 result=stage2a() if a.stage=='2a' else stage2b() if a.stage=='2b' else coverage(a.stage)
 print(json.dumps(dict(stage=a.stage,status=result.get('status','completed'),gate=result.get('gate_pass')),ensure_ascii=False))
