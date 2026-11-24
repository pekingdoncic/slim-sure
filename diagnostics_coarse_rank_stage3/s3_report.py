"""Committed prediction analysis, canonical scene bootstrap, final delivery."""
from s3_common import *
from collections import defaultdict,Counter
import csv
sys.path.insert(0,'/home/bupt/projects')
from hybrid_match.evaluation.metrics import error_auc

def csvout(name,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for r in rows for k in r));p=OUT/name;t=p.with_suffix('.tmp')
 with t.open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
  f.flush();os.fsync(f.fileno())
 os.replace(t,p)

def md(name,lines):
 p=OUT/name;t=p.with_suffix('.tmp');t.write_text('\n'.join(lines)+'\n',encoding='utf8');os.replace(t,p)

def estimate(rows,key):
 groups=defaultdict(list)
 for r in rows:
  v=r.get(key)
  if v is not None:
   assert np.isfinite(v);groups[r['scene']].append(float(v))
 sc={s:float(np.mean(v)) for s,v in sorted(groups.items())};a=np.array(list(sc.values()));ci=None
 if len(a)>1:
  rng=np.random.RandomState(2026);ci=np.quantile(a[rng.randint(0,len(a),(2000,len(a)))].mean(1),[.025,.975]).tolist()
 return dict(mean=float(a.mean()) if len(a) else None,CI95=ci,pairs=sum(map(len,groups.values())),undefined_pairs=len(rows)-sum(map(len,groups.values())),scenes=len(sc),per_scene=sc,positive=int((a>0).sum()),zero=int((a==0).sum()),negative=int((a<0).sum()),interpretation='descriptive only; known prior exposure; whole-scene paired bootstrap n=2000 seed=2026')

def auc(e):
 assert len(e)>0 and e.ndim==2 and e.shape[1]==5
 byseed=[error_auc(e[:,s],[5,10,20]) for s in range(5)]
 return {k:float(np.mean([x[k] for x in byseed])) for k in byseed[0]},byseed

def fmt(v,scale=1):
 return 'undefined' if v is None else f'{v*scale:.6f}'

def interpretation(e):
 ci=e['CI95']
 if ci is None:return '仅描述，场景不足以计算区间'
 if ci[0]<=0<=ci[1]:return '现有样本证据不足以确定增益方向'
 if ci[1]<0:return '当前固定配置在此指标上退化'
 return '在本完整已用基准上的描述性正差值，不构成未见独立场景证据'

def pose_analysis(ds,rs,methods,comparisons):
 errors={m:np.full((len(rs),5),np.inf) for m in methods};rows=[];summ={}
 protocol_sha=sha(RUN/'geometry_protocol.json');ident=identity(ds)
 for i,r in enumerate(rs):
  for m in methods:
   prediction=read(paths(ds,m,r)[2])['arrays_sha']
   for seed in range(5):
    p=RUN/'pose'/ds/m/r['pair_id']/('seed_'+str(seed)+'.json');v=read(p);rc=read(p.with_suffix('.receipt.json'))
    key=dict(pair_id=r['pair_id'],dataset=ds,method=m,seed=seed,prediction_sha=prediction,protocol_sha=protocol_sha,identity=ident)
    assert rc['identity']==v['identity']==key and rc['sha256']==sha(p)
    if v['solver_status']=='success':
     assert v['pose_error_deg'] is not None and np.isfinite(v['pose_error_deg']);errors[m][i,seed]=v['pose_error_deg']
    else:assert v['pose_error_deg'] is None
    rows.append(dict(pair_id=r['pair_id'],scene=r['scene'],method=m,seed=seed,**{k:v[k] for k in ['solver_status','failure_reason','num_matches','num_inliers','pose_error_deg','geometry_seconds','backend_seed_controlled']}))
  if i%100==0:beat('report',state='pose_receipt_validation',dataset=ds,index=i)
 csvout(ds+'_pose_per_seed.csv',rows)
 scenes=sorted({r['scene'] for r in rs});groups=[np.array([i for i,r in enumerate(rs) if r['scene']==s]) for s in scenes]
 for m in methods:
  v,perseed=auc(errors[m]);rr=[x for x in rows if x['method']==m];success=sum(x['solver_status']=='success' for x in rr)
  per_scene={s:auc(errors[m][ids])[0] for s,ids in zip(scenes,groups)}
  summ[m]=dict(completed=len(rr),planned=len(rs)*5,solver_success=success,solver_failure=len(rr)-success,failure_reasons=dict(Counter(x['failure_reason'] for x in rr if x['solver_status']!='success')),AUC_seed_mean=v,AUC_per_seed=perseed,AUC_per_scene=per_scene,AUC_scene_equal={k:float(np.mean([x[k] for x in per_scene.values()])) for k in v},matches_per_pair=float(np.mean([x['num_matches'] for x in rr])),inliers_per_pair_seed=float(np.mean([x['num_inliers'] for x in rr])),geometry_seconds=sum(x['geometry_seconds'] for x in rr))
 rng=np.random.RandomState(2026);rep={m:[] for m in methods}
 for b in range(2000):
  chosen=rng.randint(0,len(groups),len(groups));ix=np.concatenate([groups[j] for j in chosen])
  for m in methods:rep[m].append(auc(errors[m][ix])[0])
  if b%100==0:beat('report',state='pose_bootstrap',dataset=ds,replicate=b)
 deltas={}
 for after,before in comparisons:
  name=after+'-'+before;dd={}
  for k in summ[after]['AUC_seed_mean']:
   bb=np.array([a[k]-b[k] for a,b in zip(rep[after],rep[before])]);sc={s:summ[after]['AUC_per_scene'][s][k]-summ[before]['AUC_per_scene'][s][k] for s in scenes};a=np.array(list(sc.values()))
   dd[k]=dict(delta=summ[after]['AUC_seed_mean'][k]-summ[before]['AUC_seed_mean'][k],CI95=np.quantile(bb,[.025,.975]).tolist() if len(scenes)>1 else None,per_scene=sc,scene_equal_delta=float(a.mean()),positive=int((a>0).sum()),zero=int((a==0).sum()),negative=int((a<0).sum()),pairs=len(rs),scenes=len(scenes),scope='pooled pair AUC per seed then seed mean; recomputed in 2000 whole-scene paired bootstrap draws')
  deltas[name]=dd
 return dict(methods=summ,comparisons=deltas,failures='inf included in AUC, never dropped',seeds=list(range(5)),protocol_sha=protocol_sha)

def dataset_report(ds):
 rs=records(ds);cov=read(RUN/ds/'coverage.json');methods=['A','B0','B']+(['C'] if cov['gate_pass'] else [])
 comps=[('B0','A'),('B','B0')]+([('C','B'),('C','A')] if 'C' in methods else [])
 metricrows=[];changes=[];pairmeta=[];timing=[];mechanism=[];missing=Counter()
 for i,r in enumerate(rs):
  data={m:get(ds,m,r,True) for m in methods};results={m:get(ds,m,r) for m in methods};base=dict(pair_id=r['pair_id'],scene=r['scene'])
  a=data['A'];missing.update(str(x) for x in a['grid_gt_reasons'][a['grid_low0']&~a['grid_denominator']])
  for m,z in data.items():
   v=results[m];assert v['selection_identity']==results['A']['selection_identity']
   for k in ['grid_denominator','grid_low0','grid_legal0','grid_correct_target']:assert np.array_equal(z[k],a[k])
   for reg in ['low','nonlow']:
    q=v['metrics'][reg];assert q['success']==int((z['grid_denominator']&(z['grid_low0'] if reg=='low' else ~z['grid_low0'])&z['grid_any_good']).sum())
    metricrows.append(dict(**base,method=m,region=reg,**q,strict_coarse_rate=ratio(q['correct_parent'],q['denominator'])))
   timing.append(dict(**base,method=m,**v['timing']))
   pairmeta.append(dict(**base,method=m,**v['counts'],**{k:v['selection_identity'][k] for k in ['M','q','r']}))
  for after,before in comps:
   x,y=data[before],data[after]
   for reg,mask in [('low',a['grid_low0']),('nonlow',~a['grid_low0'])]:
    mask=mask&a['grid_denominator'];old=x['grid_any_good'];new=y['grid_any_good'];n=int(mask.sum())
    v=dict(**base,comparison=after+'-'+before,region=reg,denominator=n,common_success=int((mask&old&new).sum()),rescued=int((mask&~old&new).sum()),lost=int((mask&old&~new).sum()),common_failure=int((mask&~old&~new).sum()))
    v['net']=v['rescued']-v['lost'];assert v['net']==int((mask&new).sum())-int((mask&old).sum())
    for k in ['recall','coverage','precision']:
     aa,bb=results[after]['metrics'][reg][k],results[before]['metrics'][reg][k];v[k+'_delta']=None if aa is None or bb is None else aa-bb
    v['strict_coarse_rate_delta']=ratio(results[after]['metrics'][reg]['correct_parent']-results[before]['metrics'][reg]['correct_parent'],n)
    changes.append(v)
  src=a['plan_sources'];b0=a['plan_B0'][-len(src):] if len(src) else np.empty((0,2),int);b=a['plan_B'][-len(src):] if len(src) else np.empty((0,2),int)
  ch=b0[:,1]!=b[:,1];ev=a['grid_denominator'][src];gt=a['grid_correct_target'][src];c0=b0[:,1]==gt;cb=b[:,1]==gt
  mechanism.append(dict(**base,S=len(src),U=int(a['plan_U'].sum()),D=int(ch.sum()),D_GT=int((ch&ev).sum()),wrong_to_right=int((ch&ev&~c0&cb).sum()),right_to_wrong=int((ch&ev&c0&~cb).sum()),wrong_to_wrong=int((ch&ev&~c0&~cb).sum()),GT_unavailable=int((ch&~ev).sum())))
  if i%50==0:beat('report',state='prediction_validation',dataset=ds,index=i)
  del data,results
 csvout(ds+'_metrics_per_pair.csv',metricrows);csvout(ds+'_paired_changes.csv',changes);csvout(ds+'_counts_per_pair.csv',pairmeta);csvout(ds+'_timing_per_pair.csv',timing);csvout(ds+'_ranking_decisions.csv',mechanism)
 countkeys=['denominator','success','both_pass','correct_parent','no_coarse_parent','no_child','no_valid_child_GT','all_valid_EPE_ge2','block_denominator','block_success','valid_children','good_children']
 method_summ={}
 for m in methods:
  regions={}
  for region in ['low','nonlow']:
   rr=[x for x in metricrows if x['method']==m and x['region']==region];cnt={k:sum(x[k] for x in rr) for k in countkeys}
   pooled=dict(recall=ratio(cnt['success'],cnt['denominator']),coverage=ratio(cnt['block_success'],cnt['block_denominator']),precision=ratio(cnt['good_children'],cnt['valid_children']),strict_coarse_rate=ratio(cnt['correct_parent'],cnt['denominator']))
   regions[region]=dict(counts=cnt,pooled=pooled,scene_equal={k:estimate(rr,k) for k in pooled})
  tt=[x for x in timing if x['method']==m];pp=[x for x in pairmeta if x['method']==m]
  method_summ[m]=dict(planned=len(rs),completed=len(pp),failed=0,regions=regions,counts={k:sum(x[k] for x in pp) for k in ['parents','children','valid_children','low_legal_grids','legal_grids','low_GT_unavailable','q','r']},timing=dict(pair_median_mean_seconds=float(np.mean([x['median_seconds'] for x in tt])),coarse_pair_median_mean_seconds=float(np.mean([x['coarse_median_seconds'] for x in tt])),scene_equal=estimate(tt,'median_seconds'),coarse_scene_equal=estimate(tt,'coarse_median_seconds'),peak_allocated=max(x['peak_allocated'] for x in tt),peak_reserved=max(x['peak_reserved'] for x in tt)))
  cnt=method_summ[m]['counts'];method_summ[m]['low_legal_fraction']=ratio(cnt['low_legal_grids'],cnt['legal_grids'])
 for m in methods:
  for k in ['pair_median_mean_seconds','coarse_pair_median_mean_seconds']:
   method_summ[m]['timing'][k+'_relative_A']=method_summ[m]['timing'][k]/method_summ['A']['timing'][k]-1
 compare={};scene_rows=[]
 for after,before in comps:
  name=after+'-'+before;v={}
  for region in ['low','nonlow']:
   rr=[x for x in changes if x['comparison']==name and x['region']==region]
   v[region]=dict(counts={k:sum(x[k] for x in rr) for k in ['denominator','common_success','rescued','lost','common_failure','net']},**{k:estimate(rr,k) for k in ['recall_delta','coverage_delta','precision_delta','strict_coarse_rate_delta']})
   for k in ['recall_delta','coverage_delta','precision_delta','strict_coarse_rate_delta']:
    for scene,delta in v[region][k]['per_scene'].items():scene_rows.append(dict(comparison=name,region=region,metric=k,scene=scene,delta=delta))
  compare[name]=v
 csvout(ds+'_paired_scenes.csv',scene_rows)
 pose=pose_analysis(ds,rs,methods,comps)
 result=dict(dataset=ds,status='completed',time=now(),approved_manifest_sha=APPROVED[ds],identity=identity(ds),planned_pairs=len(rs),metadata_scenes=len({r['scene'] for r in rs}),scene_pair_counts=dict(Counter(r['scene'] for r in rs)),prior_exposure=True,interpretation='Separate descriptive replication; not unseen independent-scene evidence',methods=method_summ,comparisons=compare,coverage=cov,pose=pose,low_GT_unavailable_reasons=dict(missing),ranking_decisions={k:sum(x[k] for x in mechanism) for k in ['S','U','D','D_GT','wrong_to_right','right_to_wrong','wrong_to_wrong','GT_unavailable']},ranking_intervention_scenes=len({x['scene'] for x in mechanism if x['D_GT']}))
 save(OUT/(ds+'_summary.json'),result)
 lines=[f'# {ds}：完整官方清单描述性复验','',f"完成 {len(rs)} 对，{result['metadata_scenes']} 个场景 ID。清单 SHA-256：{APPROVED[ds]}。此基准存在历史使用，不能解释为未见独立场景证据。",'','| 方法 | 完成/计划 | 低纹理分母 | 好子点源格 | 场景等权召回 | pooled 召回 | 32px 场景等权覆盖 | 有效子点精度 | 前向 ms |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for m,z in method_summ.items():
  q=z['regions']['low'];lines.append(f"| {m} | {z['completed']}/{z['planned']} | {q['counts']['denominator']} | {q['counts']['success']} | {fmt(q['scene_equal']['recall']['mean'])} | {fmt(q['pooled']['recall'])} | {fmt(q['scene_equal']['coverage']['mean'])} | {fmt(q['scene_equal']['precision']['mean'])} | {fmt(z['timing']['pair_median_mean_seconds'],1000)} |")
 lines+=['','## 配对差值','', '| 比较 | 场景等权低纹理召回差 pp | 95% 描述性区间 pp | 救回/丢失/净增 | 共同可评 pair/scene | 场景正/零/负 |','|---|---:|---|---|---|---|']
 commentary=[]
 for name,z in compare.items():
  e=z['low']['recall_delta'];c=z['low']['counts'];ci=e['CI95'];lines.append(f"| {name} | {fmt(e['mean'],100)} | {str([round(v*100,6) for v in ci]) if ci else 'undefined'} | {c['rescued']}/{c['lost']}/{c['net']} | {e['pairs']}/{e['scenes']} | {e['positive']}/{e['zero']}/{e['negative']} |");commentary.append(name+'：'+interpretation(e)+'。')
 lines+=['',*commentary,'','非低纹理召回、覆盖、精度及严格粗格命中与逐场景差值见 summary.json、paired_scenes.csv。零分母 pair 保留为 undefined；无子点按实际细输出判断，另列无粗父点，避免将有父点但零细点混作 GT 无效。','',f"低纹理合法格占比 {fmt(method_summ['A']['low_legal_fraction'])}，低纹理 GT 不可评格 {method_summ['A']['counts']['low_GT_unavailable']}。冻结阈值 0.0001920461654663086；跨域未调阈值。",'',f"主 C 覆盖：{cov['status']}；S/U/H/V={cov['counts']}；有效场景 {len(cov['valid_scenes'])}；原因 {cov['reasons']}。跳过不记作 C=B 或零收益。",'',f"B 相对 B0 的改选：{result['ranking_decisions']}；有 GT 改选覆盖 {result['ranking_intervention_scenes']} 场景。",'','## 位姿与成本','','canonical LO-RANSAC，原图坐标与 K，0.5 px、confidence 0.99999、种子 0–4。每种子在 pair 上计算 AUC，再取种子均值；失败按 inf 进入分母。每次场景 bootstrap 都重算 AUC。','','| 方法 | AUC@5 | AUC@10 | AUC@20 | 求解成功/计划 | allocated/reserved GiB | 粗匹配 ms |','|---|---:|---:|---:|---|---|---:|']
 for m,z in pose['methods'].items():
  t=method_summ[m]['timing'];v=z['AUC_seed_mean'];lines.append(f"| {m} | {v['auc@5deg']:.6f} | {v['auc@10deg']:.6f} | {v['auc@20deg']:.6f} | {z['solver_success']}/{z['planned']} | {t['peak_allocated']/2**30:.3f}/{t['peak_reserved']/2**30:.3f} | {t['coarse_pair_median_mean_seconds']*1000:.6f} |")
 lines+=['','各方法独立新鲜进程，每 pair 预热一次后同步计时三次取中位数。计入实际纹理、选择与锚点计算；排除 GT、哈希、审计、落盘与 RANSAC。位姿配对区间、各种子 AUC、逐场景结果、求解内点和失败原因均保存在 JSON/CSV。','',f"复跑（按已提交哈希恢复）：/home/bupt/miniconda3/envs/slim/bin/python -B {CODE}/run_stage3.py --background。报告只读重算：/home/bupt/miniconda3/envs/slim/bin/python -B {CODE}/s3_report.py。"]
 md(ds+'_Report.md',lines);return result

def main():
 gate();beat('report',state='start');results={ds:dataset_report(ds) for ds in APPROVED};gate()
 storage=dict(run_bytes=sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file()),linux_free_gib=shutil.disk_usage(RUN).free/2**30,host_free_gib=shutil.disk_usage('/mnt/c').free/2**30)
 minima={};peak_swap=0
 for p in (RUN/'environment_probe').glob('*/resources.jsonl'):
  for line in p.open():
   r=json.loads(line)
   for k in ['host_c_free_bytes','ext4_free_bytes','mem_available_bytes']:
    if r.get(k) is not None:minima[k]=min(minima.get(k,r[k]),r[k])
   peak_swap=max(peak_swap,r.get('swap_used_bytes',0))
 storage.update(measured_minima=minima,peak_swap_used_bytes=peak_swap)
 save(OUT/'completion.json',dict(status='completed',time=now(),datasets={d:dict(pairs=x['planned_pairs'],methods=list(x['methods']),coverage=x['coverage']['status']) for d,x in results.items()},storage=storage,context_sha=sha(RUN/'context.json'),code_sha=sha(RUN/'code_identity.json'),validation=read(RUN/'validation/status.json'),training=False))
 lines=['# 阶段 3 完整官方基准复验交付','',f"完成时间：{now()}。MegaDepth 1,488 对、ScanNet 1,500 对；两份获批完整清单均已完成条件允许的全部方法、五种子位姿和实测计时。未训练，未运行其他室外留出集、oracle、探索半径或仿射。",'','两份基准都存在历史选参/基准使用，分别作为室外和室内描述性复验。室内场景不并入室外场景数；ScanNet 按官方 scene 整数归并 scan 后缀，缺少更高层级建筑地点身份。','']
 for ds,z in results.items():
  lines += [f'## {ds}','',f"报告：{ds}_Report.md；完整指标：{ds}_summary.json。",'']
  for comp,v in z['comparisons'].items():
   e=v['low']['recall_delta'];lines.append(f"- {comp}：低纹理召回差 {fmt(e['mean'],100)} pp；{interpretation(e)}。")
  lines+=['',f"主 C：{z['coverage']['status']}，原因 {z['coverage']['reasons']}。",'']
 lines+=['## 对机制和训练投入的解释','','诊断集已确认：原生全局 Top-K 与相对阈值是主要筛选环节，配额 B0 的收益须与目标重排的 B−B0、邻域 C−B 分开归因。同池 oracle 是诊断集严格粗格上界，不能作为真实部署方法或细点召回保证；其历史结果见父目录 Final_Report.md。']
 for ds,z in results.items():
  b=z['comparisons']['B-B0']['low']['recall_delta'];b0=z['comparisons']['B0-A']['low']['recall_delta'];lines.append(f"{ds}：配额差 {fmt(b0['mean'],100)} pp，排名增量 {fmt(b['mean'],100)} pp。排名判断：{interpretation(b)}。")
  if 'C-B' in z['comparisons']:lines.append('邻域判断：'+interpretation(z['comparisons']['C-B']['low']['recall_delta'])+'。')
  else:lines.append('该域未运行 C：覆盖门槛不满足，不能推断邻域零收益。')
 lines+=['','当前证据可支持固定配置的机制取舍，尚缺未曾用于选择的独立场景复验来支持训练泛化收益。不得把配额增益归给未证实的可学习重排，也不把区间含零解释为排除了该机制。','',f"实际 run 占用 {storage['run_bytes']/2**30:.3f} GiB；结束时宿主 C: {storage['host_free_gib']:.3f} GiB、Linux {storage['linux_free_gib']:.3f} GiB 可用。环境监控保留每次资源采样及内核游标，检查边界见 initial_environment.json 和 environment_probe/；不宣称历史底层存储故障已根治。",'','协议、批准记录和派生清单在 run 根目录；逐对预测及原始选择证据在 dataset/method/pairs；几何结果在 pose；进程与真实命令在 launch.json、logs/*_launch.json；代码冻结在 code_identity.json 与 code_snapshot。','',f"运行目录：{RUN}",f"报告目录：{OUT}",'',f"恢复：cd {ROOT} && /home/bupt/miniconda3/envs/slim/bin/python -B {CODE}/run_stage3.py --background。完成后执行会校验并复用已提交结果，不重复模型前向。"]
 md('Stage3_Final_Report.md',lines)
 receipt={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='report_receipt.json'}
 save(OUT/'report_receipt.json',dict(status='committed',time=now(),files=receipt))
 beat('report',state='completed');save(RUN/'stage_status.json',dict(status='completed',stage='report',time=now(),report=str(OUT/'Stage3_Final_Report.md'),report_receipt_sha=sha(OUT/'report_receipt.json')))
 print(json.dumps(dict(status='completed',report=str(OUT/'Stage3_Final_Report.md')),ensure_ascii=False),flush=True)
if __name__=='__main__':main()