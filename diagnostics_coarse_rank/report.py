"""CPU summaries from committed arrays; all denominators and pairing explicit."""
from common import *
from collections import defaultdict,Counter
import csv
DISCLAIMER='阶段 2 使用了最初发现粗匹配问题的同一批 100 对图像，所有结果均为探索性机制分析。跨数据增益仅由阶段 3 经确认的独立清单及其实际场景覆盖评价。'
def md(name,s):
 p=OUT/name;t=p.with_suffix('.tmp');t.write_text(s,encoding='utf8');os.replace(t,p)
def csvwrite(name,rows):
 if not rows:return
 cols=list(dict.fromkeys(k for r in rows for k in r))
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
def mean_scene(rows,key):
 vals=defaultdict(list)
 for r in rows:
  v=r.get(key)
  if v is not None and np.isfinite(v):vals[r['scene']].append(float(v))
 return {s:float(np.mean(v)) for s,v in sorted(vals.items())}
def estimate(rows,key):
 v=mean_scene(rows,key);a=np.array(list(v.values()));ci=None
 if len(a)>=2:
  rng=np.random.RandomState(2026);boot=np.mean(a[rng.randint(0,len(a),(2000,len(a)))],axis=1);ci=np.quantile(boot,[.025,.975]).tolist()
 return dict(mean=float(a.mean()) if len(a) else None,CI95=ci,scenes=len(a),pairs=int(sum(r.get(key) is not None and np.isfinite(r[key]) for r in rows)),per_scene=v,positive=int((a>0).sum()),zero=int((a==0).sum()),negative=int((a<0).sum()),interpretation='diagnostic exploratory; 2000 whole-scene paired bootstrap seed2026')
def measured(a):
 den=a['grid_denominator']&a['grid_low0'];n=int(den.sum())
 return dict(denominator=n,parents=len(a['raw_id']),both_pass=int((den&a['grid_both_pass']).sum()),correct_parent=int((den&a['grid_correct_parent']).sum()),any_good=int((den&a['grid_any_good']).sum()),recall=ratio((den&a['grid_any_good']).sum(),n))
def pair_metadata(rows):
 return dict(manifest_sha=read(RUN/'context.json')['manifest_sha'],protocol_sha=read(RUN/'context.json')['protocol_sha'],code=code_identity(),time=now(),planned=100,completed=len(rows),failed=100-len(rows),independent_model_runs=0)
def stage0():
 from diagnostics_threshold.shared import get_unit
 diffs=[];data={m:[] for m in ['history','A1','A2']}
 for r in recs():
  arr={'history':get_unit(HISTORY,'S0',r,True),'A1':get('A1',r,True),'A2':get('A2',r,True)}
  for m,a in arr.items():data[m].append(dict(**base(r),method=m,**measured(a)))
  for after,before in [('A1','history'),('A2','history'),('A2','A1')]:
   a,b=arr[after],arr[before];x=measured(a);y=measured(b);name=after+'-'+before
   d=dict(**base(r),comparison=name,**{k:x[k]-y[k] if x[k] is not None and y[k] is not None else None for k in x})
   d.update(parent_removed=len(np.setdiff1d(b['raw_id'],a['raw_id'])),parent_added=len(np.setdiff1d(a['raw_id'],b['raw_id'])),ordered_parents_equal=np.array_equal(a['raw_id'],b['raw_id']),source_exact=np.array_equal(a['source'],b['source']),target_exact=np.array_equal(a['target'],b['target']))
   for k in ['grid_denominator','grid_correct_target','grid_low0','grid_low1','grid_legal0','grid_legal1']:assert np.array_equal(a[k],b[k]),k
   if before=='A1':
    d['P_exact']=str(a['capture_P_sha'])==str(b['capture_P_sha']);d['sim_exact']=str(a['capture_sim_sha'])==str(b['capture_sim_sha'])
    d['candidate_exact']=np.array_equal(a['candidate0'],b['candidate0']) and np.array_equal(a['candidate1'],b['candidate1'])
   diffs.append(d)
  del arr
 total={m:{k:sum(r[k] for r in rr) for k in ['denominator','parents','both_pass','correct_parent','any_good']} for m,rr in data.items()}
 for m in total:total[m]['recall_scene_equal']=estimate(data[m],'recall')['mean']
 tol=read(RUN/'protocol.json')['tolerance'];checks=[];pass_all=True
 assert total['history']['denominator']==7954 and all(total['history'][k]==v for k,v in tol['count_reference'].items())
 for after,before in [('A1','history'),('A2','history'),('A2','A1')]:
  name=after+'-'+before;dr=[r for r in diffs if r['comparison']==name];sc=mean_scene(dr,'recall');sv=np.array(list(sc.values()));pv=np.array([r['recall'] for r in dr if r['recall'] is not None])
  limit=math.ceil(2*len(sv)/3);directional=bool(max(int((sv>.001).sum()),int((sv<-.001).sum()))>=limit) if len(sv) else False
  counts={k:dict(delta=total[after][k]-total[before][k],limit=lim,pass_=abs(total[after][k]-total[before][k])<=lim) for k,lim in tol['count_absolute_limits'].items()}
  rd=total[after]['recall_scene_equal']-total[before]['recall_scene_equal']
  ok=all(x['pass_'] for x in counts.values()) and abs(rd)<=.001 and not directional and total[after]['denominator']==7954
  check=dict(comparison=name,status='pass' if ok else 'failed',counts=counts,recall_delta=rd,recall_limit=.001,directional_shift=directional,pair_max_abs=float(np.abs(pv).max()) if len(pv) else None,pair_signs=[int((pv>0).sum()),int((pv==0).sum()),int((pv<0).sum())],scene_max_abs=float(np.abs(sv).max()) if len(sv) else None,scene_signs=[int((sv>0).sum()),int((sv==0).sum()),int((sv<0).sum())],scenes_over_positive_threshold=int((sv>.001).sum()),scenes_over_negative_threshold=int((sv<-.001).sum()),directional_scene_limit=limit,per_scene=sc)
  checks.append(check);pass_all&=ok
 exact=all(r['ordered_parents_equal'] and r['source_exact'] and r['target_exact'] for r in diffs)
 result=dict(status='pass' if pass_all else 'failed',reproduction='exact' if exact else 'within_preregistered_tolerance' if pass_all else 'out_of_tolerance',totals=total,comparisons=checks,**pair_metadata(data['A1']))
 save(OUT/'stage0_acceptance.json',result);csvwrite('baseline_per_pair.csv',sum(data.values(),[]));csvwrite('baseline_differences.csv',diffs)
 inv=read(RUN/'inventory/route_frozen.json')
 lines=['# 阶段 0：身份、库存与基线验收','',f"状态：{result['status']}；复现分类：{result['reproduction']}。A1 固定作为后续配对参照，A2 仅用于波动核验。",'','| 方法 | 低纹理固定分母 | 双侧入池 | 严格正确粗格 | 好子点源格 | 粗父点 | 场景等权召回 |','|---|---:|---:|---:|---:|---:|---:|']
 for m,x in total.items():lines.append(f"| {m} | {x['denominator']} | {x['both_pass']} | {x['correct_parent']} | {x['any_good']} | {x['parents']} | {x['recall_scene_equal']:.12f} |")
 lines+=['','三项比较按事前冻结容差验收；逐对、逐场景差异及方向性检查见 stage0_acceptance.json 和 baseline_differences.csv。独立进程、预热、软件和内核信息见各组 runtime/ 与 invocation.json。','',
 '输入、低纹理、GT 和合法掩码逐对与历史精确一致；同一 P 回放保留原生 torch.topk 实际并列顺序。开发 3 对的包装器关闭/开启路径逐元素输出一致。','',
 '## 独立集可用性与验证路线','', '路线在本轮模型输出之前冻结：室外仅作描述性复验。官方清单的 2 个场景此前已用于基准与选择分析；其他预存清单只有 8 个可核实未用场景。ScanNet 单列室内跨域，存在以前的选参暴露。','',
 '| 候选 | 原始对 | 无序去重可读对 | 唯一图像 | 物理场景 ID 数 | 可核实未用场景 |','|---|---:|---:|---:|---:|---:|']
 for c in inv['candidates']:lines.append(f"| {c['dataset']} | {c['raw_pairs']} | {c['readable_pairs']} | {c['unique_images']} | {len(c['scenes'])} | {len(c['known_independent_scenes'])} |")
 lines+=['','完整清单、SHA-256、逐场景计数及场景/图像/无序 pair 交集见 independent_inventory.json。ScanNet 按索引第一场景整数合并 scan 后缀；缺少额外建筑地点元数据，场景 ID 不能证明更高层级地点独立。图像交集按规范化数据路径身份核查，跨场景内容重复尚未证明。','',
 f"配置 SHA-256：{result['protocol_sha']}；100 对清单 SHA-256：{result['manifest_sha']}。参考 HEAD：{read(RUN/'context.json')['HEAD']}。代码 SHA 见 JSON。",'',
 '环境：当前 boot 全日志流式审阅未见存储故障；旧启动的 journal 不洁关闭警告保留。有限读、写入 fsync 回读已通过。Windows 虚拟盘与 swap 在 C:，可用容量和事件窗口见 windows_environment.json。未改变配置、包、数据和原权重。','',
 '独立集模型运行数：0。阶段 3 仍需用户确认清单哈希。','',
 '续跑：cd /home/bupt/projects/SLiM-master && /home/bupt/miniconda3/envs/slim/bin/python -B diagnostics_coarse_rank/report.py --stage 1。']
 md('Stage0_Report.md','\n'.join(lines)+'\n')
 assert pass_all,'Stage0 gate failed'
 return result
def stage1():
 assert read(OUT/'stage0_acceptance.json')['status']=='pass'
 rows=[];counts=Counter();filters=Counter();ties=Counter();perpair=[];missing=[]
 for r in recs():
  a=get('A1',r,True);ix=a['rank_source'];low=a['rank_low'];selected=a['grid_correct_parent'][ix];good=a['grid_any_good'][ix]
  assert np.array_equal(selected,a['rank_global_topk']&a['rank_above_threshold'])
  den=a['grid_denominator'];lo=a['grid_low0'];both=a['grid_both_pass'];unselected=~selected
  missing.append(dict(**base(r),low_evaluable=int((den&lo).sum()),low_legal_total=int(lo.sum()),low_center_GT_or_target_unavailable=int((lo&~den).sum()),low_unavailable_reasons=dict(Counter(str(x) for x in a['grid_gt_reasons'][lo&~den])),low_source_outside_matrix=int((den&lo&~np.isin(np.arange(len(den)),a['candidate0'])).sum()),low_GT_target_outside_matrix=int((den&lo&~np.isin(a['grid_correct_target'],a['candidate1'])).sum()),low_either_outside_matrix=int((den&lo&~both).sum())))
  for region,rmask in [('low',low),('nonlow',~low)]:
   for stratum,mask in [('both_candidates',rmask),('strict_missed',rmask&unselected),('strict_missed_and_no_good_child',rmask&unselected&~good)]:
    n=int(mask.sum());counts[region+'_'+stratum]+=n
    for score in ['P','sim']:
     for axis in ['row','col']:
      ranks=a[f'rank_{score}_{axis}'][mask]
      row=dict(**base(r),region=region,stratum=stratum,score=score,axis=axis,denominator=n,rank1=int((ranks==1).sum()),rank2_5=int(((ranks>=2)&(ranks<=5)).sum()),rank6_10=int(((ranks>=6)&(ranks<=10)).sum()),rank_gt10=int((ranks>10).sum()))
      for k in [1,3,5]:row['hit'+str(k)]=ratio((ranks<=k).sum(),n);row['n_hit'+str(k)]=int((ranks<=k).sum())
      rows.append(row)
   lost=rmask&unselected&~good;src=ix[lost]
   counts[region+'_missed_no_child']+=int((~a['grid_any_parent'][src]).sum())
   counts[region+'_missed_all_GT_invalid']+=int((a['grid_any_parent'][src]&~a['grid_has_evaluable_child'][src]).sum())
   counts[region+'_missed_all_valid_EPE_ge2']+=int(a['grid_has_evaluable_child'][src].sum())
   counts[region+'_strict_missed_neighbor_rescued']+=int((rmask&unselected&good).sum())
   for top in [0,1]:
    for threshold in [0,1]:filters[(region,top,threshold)]+=int((rmask&(a['rank_global_topk']==top)&(a['rank_above_threshold']==threshold)).sum())
  for score in ['P','sim']:
   for axis in ['row','col']:
    for k in [1,5]:
     v=a[f'rank_{score}_{axis}_tie{k}'][low];ties[(score,axis,k)]+=int((v>0).sum())
  perpair.append(dict(**base(r),both=int(low.sum()),strict_missed=int((low&unselected).sum()),unrescued=int((low&unselected&~good).sum()),rescued_by_neighbor=int((low&unselected&good).sum())))
 agg=[]
 for region in ['low','nonlow']:
  for stratum in ['both_candidates','strict_missed','strict_missed_and_no_good_child']:
   for score in ['P','sim']:
    for axis in ['row','col']:
     rr=[x for x in rows if (x['region'],x['stratum'],x['score'],x['axis'])==(region,stratum,score,axis)]
     n=sum(x['denominator'] for x in rr);v=dict(region=region,stratum=stratum,score=score,axis=axis,denominator=n)
     for key in ['rank1','rank2_5','rank6_10','rank_gt10']:v[key]=sum(x[key] for x in rr)
     for k in [1,3,5]:v['hit'+str(k)]=dict(pooled=ratio(sum(x['n_hit'+str(k)] for x in rr),n),scene_equal=estimate(rr,'hit'+str(k)))
     agg.append(v)
 filt=[dict(region=k[0],global_topk=bool(k[1]),strict_threshold=bool(k[2]),count=v) for k,v in sorted(filters.items())]
 summary=dict(counts=dict(counts),filters=filt,rankings=agg,boundary_ties=[dict(score=k[0],axis=k[1],cut=k[2],low_rows_crossing=v) for k,v in ties.items()],missing=missing,**pair_metadata(perpair))
 save(OUT/'stage1_rankings.json',summary);csvwrite('rank_loss_per_pair.csv',rows);csvwrite('rank_funnel_per_pair.csv',perpair);csvwrite('filter_cross_tab.csv',filt);csvwrite('correct_grid_outside_matrix.csv',missing)
 target=next(x for x in agg if (x['region'],x['stratum'],x['score'],x['axis'])==('low','strict_missed','P','row'))
 major=max([x for x in filt if x['region']=='low' and not(x['global_topk'] and x['strict_threshold'])],key=lambda x:x['count'])
 lines=['# 阶段 1：正确目标排名与筛选流失','',f"100/100 对真实 A1 缓存，15 场景；固定可评价低纹理源格 7,954。双方入池 {counts['low_both_candidates']}；严格正确目标未被选中 {counts['low_strict_missed']}；其中邻格细匹配已救回 {counts['low_strict_missed_neighbor_rescued']}，仍无好子点 {counts['low_strict_missed_and_no_good_child']}。",'',
 f"严格漏失集合中，P 行前五可达比例：pooled {target['hit5']['pooled']:.6f}；场景等权 {target['hit5']['scene_equal']['mean']:.6f}。这只衡量池内可达性，不能视为端到端增益。",'',
 f"最大的筛选交叉单元：global Top-K={major['global_topk']}、严格相对阈值={major['strict_threshold']}，共 {major['count']} 个低纹理正确对应。两条线均未通过的对应只计一次。",'',
 f"真正未救回集合拆分：无子点 {counts['low_missed_no_child']}；有子点但 GT 全无效 {counts['low_missed_all_GT_invalid']}；有有效 GT 但全部 EPE≥2 px {counts['low_missed_all_valid_EPE_ge2']}。",'',
 '原始温度缩放相似度与 P 分别报告行/列排名、1/2–5/6–10/>10 分档、前 1/3/5 命中率和非低纹理对照；见 stage1_rankings.json、rank_loss_per_pair.csv。矩阵外正确目标单列，零分母保留并标 undefined。每对详细排名、P margin、阈值标记和并列边界成员数在 A1/pairs/*.npz。','',
 '新增排名并列按原始格 ID 破平；A 的原生 Top-K 顺序保持不变。统计用 pair→scene→场景等权，区间为 2,000 次整场景 bootstrap。参数保持预注册值，阶段 1 结果不用于调参。','',
 f"清单 SHA：{summary['manifest_sha']}；协议 SHA：{summary['protocol_sha']}。代码哈希见 JSON。",'',
 '续跑：cd /home/bupt/projects/SLiM-master && /home/bupt/miniconda3/envs/slim/bin/python -B diagnostics_coarse_rank/chain.py --methods B0 B oracle --background。独立集运行数仍为 0。']
 md('Stage1_Report.md','\n'.join(lines)+'\n');return summary
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--stage',required=True);a=p.parse_args()
 result=stage0() if a.stage=='0' else stage1() if a.stage=='1' else None
 print(json.dumps({'stage':a.stage,'status':result.get('status','completed') if result else 'unknown'},ensure_ascii=False))
