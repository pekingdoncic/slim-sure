"""Finalize optional diagnostics, measured costs, reproducible handoff."""
from mechanisms import *
from proposal import proposal
import shutil,subprocess
def optional():
 methods=['A1','B','C4','C8','C12','Caffine'];rows,stats=summarize_methods(methods);tr,comp=compare(methods,[(m,'B') for m in ['C8','C12','Caffine']]+[('Caffine','C4')])
 paired=[];degen=Counter()
 for r in recs():
  a=get('A1',r,True);t=npzload(RUN/'coverage/C4'/(pairname(r['pair_id'])+'.npz'));f=npzload(RUN/'coverage/Caffine'/(pairname(r['pair_id'])+'.npz'))
  assert np.array_equal(t['sources'],f['sources']) and np.array_equal(t['anchor_ids'],f['anchor_ids'])
  eligible=t['V']&f['V'];s=t['sources'];gt=a['grid_correct_target'][s];valid=a['grid_denominator'][s]
  tt=t['C'][t['slots'],1];ft=f['C'][f['slots'],1];at=get('C4',r,True);af=get('Caffine',r,True)
  e=eligible&valid;row=dict(**base(r),both_fit=int(eligible.sum()),both_fit_GT=int(e.sum()),translation_hits=int((e&(tt==gt)).sum()),affine_hits=int((e&(ft==gt)).sum()),same_target=int((eligible&(tt==ft)).sum()),different_target=int((eligible&(tt!=ft)).sum()))
  for kind,mask in [('all_both_fit',e),('exactly3_both_fit',e&(f['nanchor']==3)),('ge4_both_fit',e&(f['nanchor']>=4))]:
   x=at['grid_any_good'][s];y=af['grid_any_good'][s]
   for name,z in [('common_success',x&y),('rescued',~x&y),('lost',x&~y),('common_failure',~x&~y)]:row[kind+'_'+name]=int((mask&z).sum())
  paired.append(row);degen.update(f['reasons'].tolist())
 result=dict(status='completed',metrics=stats,comparisons=comp,affine_translation_common_fit=paired,affine_degeneracy=dict(degen),coverage={m:read(OUT/('coverage_'+m+'.json')) for m in ['C4','C8','C12','Caffine']},scope='prespecified exploratory diagnostics only; primary remains C4',**pair_metadata(recs()))
 save(OUT/'optional_results.json',result);csvwrite('optional_metrics.csv',rows);csvwrite('optional_source_transitions.csv',tr);csvwrite('affine_translation_common_fit.csv',paired)
 lines=['# 阶段 2 选做结果','',DISCLAIMER,'','全部选做配置各完成 100/100 对。半径与仿射不升级为主方法，不据此选择独立集参数。','',
 '| 配置 | S/U/H/V | V场景数 | 相对B低纹理召回差 | 95%场景区间 |','|---|---|---:|---:|---|']
 for m in ['C8','C12','Caffine']:
  cov=result['coverage'][m];e=comp[m+'-B']['low']['recall'];c=cov['counts'];lines.append(f"| {m} | {c['S']}/{c['U']}/{c['H']}/{c['V']} | {len(cov['valid_scenes'])} | {e['mean']:+.6f} | {e['CI95']} |")
 lines+=['',f"仿射回退计数：{dict(degen)}。有效恰好3锚点 {result['coverage']['Caffine']['counts']['exactly3_valid']}；≥4锚点 {result['coverage']['Caffine']['counts']['ge4_valid']}。恰好3点的精确拟合不能单独证明几何可靠。",'',
 '平移和仿射共用完全相同的 R4 锚点、固定 B 槽位、目标池与权重；锚点 ID 逐项核验。共同可拟合子集的 GT 分母、严格命中、源格救回/丢失见 affine_translation_common_fit.csv。全体 100 对配对差值与逐场景方向见 optional_results.json；区间含零表示现有证据不足以确定增益方向。','',
 f"协议 SHA：{result['protocol_sha']}；清单 SHA：{result['manifest_sha']}。续跑：python -B diagnostics_coarse_rank/finalize.py --stage optional。"]
 md('Stage2_Optional_Report.md','\n'.join(lines)+'\n');return result
def finish():
 gate();methods=['A1','A2','B0','B','oracle','C4','C8','C12','Caffine']
 statuses={m:read(RUN/m/'status.json') for m in methods};assert all(x['status']=='completed' for x in statuses.values())
 timing={m:read(RUN/'timing'/m/'status.json') for m in ['A1','B0','B','C4']};assert all(x['completed']==100 for x in timing.values())
 tm=[]
 for m,x in timing.items():
  tm.append(dict(method=m,seconds=x['mean_pair_median_seconds'],coarse_seconds=x['mean_coarse_median_seconds'],relative_to_A=x['mean_pair_median_seconds']/timing['A1']['mean_pair_median_seconds'],gpu_peak_allocated_gib=x['peak_allocated']/2**30,gpu_peak_reserved_gib=x['peak_reserved']/2**30,completed=x['completed'],fresh_timed_forwards=3*x['completed']))
 save(OUT/'timing_summary.json',dict(scope=timing['A1']['scope'],methods=tm));csvwrite('timing_summary.csv',tm)
 for m,x in timing.items():csvwrite('timing_'+m+'_per_pair.csv',x['rows'])
 proposal()
 stage2=read(OUT/'stage2A.json');stageb=read(OUT/'stage2B.json');opt=read(OUT/'optional_results.json');metrics={**stage2['metrics'],**stageb['metrics'],**opt['metrics']}
 c=stageb['comparisons']['C4-B']['low'];main=stage2['comparisons']['B0-A1']['low'];ranking=stage2['comparisons']['B-B0']['low']
 env=[]
 for p in (RUN/'environment_probe').glob('*/resources.jsonl'):
  for line in p.open():
   try:env.append(json.loads(line))
   except json.JSONDecodeError:pass
 resource=dict(min_host_free_gib=min(x['host_c_free_bytes'] for x in env)/2**30,min_linux_free_gib=min(x['ext4_free_bytes'] for x in env)/2**30,min_mem_available_gib=min(x['mem_available_bytes'] for x in env)/2**30,max_swap_used_gib=max(x['swap_used_bytes'] for x in env)/2**30,run_bytes=sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file()),scope='Sampled every2s during mechanism forwards and between timing pairs; includes no claim of unsampled exact physical disk peak')
 save(OUT/'native_empty_boundary_fix.json',dict(status='pass_with_new_wrapper',original_failure='Native zero-parent fine-window JIT: Global alloc not supported yet',evidence=str(RUN/'recovery/native_empty_window'),guard_sha=sha(ROOT/'diagnostics_coarse_rank/empty_guard.py'),guard_scope='Only zero-parent fine correlation and zero-child refinement; frozen src/slim.py unchanged',all_900_main_pairs_nonempty=all(get(m,r)['counts']['parents']>0 for m in methods for r in recs()),timing_equivalence='100 pairs x4 methods x3 fresh timed forwards exactly match committed parents and final coordinates',time=now()))
 snapshots=RUN/'code_snapshot';snapshots.mkdir(exist_ok=True)
 hashes={}
 for p in sorted((ROOT/'diagnostics_coarse_rank').glob('*.py')):
  target=snapshots/p.name
  if target.exists():assert sha(target)==sha(p)
  else:shutil.copy2(p,target)
  hashes[p.name]=sha(p)
 save(RUN/'experiment_code_identity.json',dict(time=now(),reference_HEAD=read(RUN/'context.json')['HEAD'],files=hashes,default_off='New standalone directory only; frozen native files unchanged'))
 currentgit=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True)
 status=dict(status='stages_0_1_2_completed_awaiting_stage3_confirmation',time=now(),methods={m:dict(planned=100,completed=100,failed=0,seconds=statuses[m]['worker']['seconds']) for m in methods},stage0='pass_exact',stage1='completed',stage2A='completed',stage2B='completed',optional='completed',timing='completed_1200_timed_forwards',stage3='not_started_requires_explicit_manifest_approval',independent_model_runs=0,training='not_performed',resource=resource,code=hashes,git_status=currentgit,active_GPU_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip())
 save(RUN/'stage_status.json',status);save(OUT/'final_status.json',status)
 lines=['# SLiM 低纹理粗匹配排名、配额与邻域重排序：交付报告','',DISCLAIMER,'',
 '阶段 0、1、2 必做项、主 C、三组预声明选做项及诊断集计时已完成。无训练、无 optimizer 更新。阶段 3 尚未启动，独立集模型运行数为 0。','',
 '## 直接结论','',
 '1. 漏失主要发生在双侧入池之后的全局预算与阈值筛选。7,728 个双侧入池的可评低纹理正确对应中，4,152 个未成为严格正确粗匹配；272 个已由邻格精修救回，3,880 个仍漏失。漏失正确目标中 76.66% 在 P 行前五。',
 f"2. 固定 10% 配额 B0 是当前主要收益来源：相对 A 场景等权低纹理召回 +{100*main['recall']['mean']:.3f} 个百分点，95% 区间 {[100*x for x in main['recall']['CI95']]}；714 个救回、22 个丢失，净增692。代价是非低纹理成功源格净减4,082，不能仅据低纹理指标称整体改善。",
 f"3. 行列排名 B 相对 B0 为 {100*ranking['recall']['mean']:+.3f} 个百分点，95% 区间 {[100*x for x in ranking['recall']['CI95']]}。2,677 次改选中只有53次有中心 GT；错→对9，对→错18。现有样本证据不足以确定排名增益方向。",
 f"4. 主 C 平移的有效支持覆盖6,283/17,057槽位，遍及15场景；实际改选749，有中心 GT 的79次改选中错→对37、对→错11。相对B净救回24个低纹理源格，场景等权 +{100*c['recall']['mean']:.3f} 个百分点，95% 区间 {[100*x for x in c['recall']['CI95']]}，仍含零。",
 '5. 固定预留槽位中1,422个有可评中心 GT，1,212个正确目标在池内。B0/B/oracle 严格命中771/762/1,212；主C严格命中见下表。Oracle 的细匹配结果不是数学上界；目标邻格可能得到更好的细点。',
 '6. 暂无充分依据投入重排序头训练：需要真正独立场景的排序或邻域净收益、非低纹理代价与位姿/效率证据。本次只支持继续区分配额收益与排序收益，不将同批100对的机制结果当作跨数据证明。','',
 '| 方法 | 低纹理成功源格/7954 | 场景等权召回 | 非低纹理成功源格 |','|---|---:|---:|---:|']
 for m in ['A1','B0','B','C4','oracle','C8','C12','Caffine']:
  x=metrics[m];lines.append(f"| {m} | {x['low']['pooled']['success']}/7954 | {x['low']['scene_equal']['recall']['mean']:.6f} | {x['nonlow']['pooled']['success']} |")
 c_hits=sum(x['C_hits'] for x in stageb['changes']);lines+=['',f"主C严格命中 {c_hits}/1422，距离同池严格上界尚差 {1212-c_hits}；B尚差450，B0尚差441。各模型仅作固定配置结论。13对低纹理分母为零，保留为 undefined；主召回实际可评87对、15场景。所有指标同时提供 pooled 分子分母及 pair→scene→场景等权结果。",'',
 '## 实测成本与资源','',
 '| 方法 | 全前向均值 ms | 粗匹配均值 ms | 相对A | 峰值 allocated GiB | reserved GiB |','|---|---:|---:|---:|---:|---:|']
 for x in tm:lines.append(f"| {x['method']} | {1000*x['seconds']:.2f} | {1000*x['coarse_seconds']:.2f} | {x['relative_to_A']:.3f} | {x['gpu_peak_allocated_gib']:.3f} | {x['gpu_peak_reserved_gib']:.3f} |")
 lines+=['','每方法独立新进程，每对预热后3次 CUDA 同步完整前向取中位数，再跨100对平均。粗匹配计时包括纹理、池构造、排名、锚点和回退；测量窗口排除 GT、审计、落盘、缓存选择和 RANSAC。计时路径与真实审计路径父点及最终坐标逐元素一致。Oracle 不参与部署计时。',
 f"资源采样：C盘最少 {resource['min_host_free_gib']:.2f} GiB，Linux余量最少 {resource['min_linux_free_gib']:.2f} GiB，可用主存最少 {resource['min_mem_available_gib']:.2f} GiB，swap使用最多 {resource['max_swap_used_gib']:.2f} GiB；本run约 {resource['run_bytes']/2**30:.2f} GiB。",'',
 '## 故障与核验边界','',
 '第一次 A1 启动在首对提交前因 journalctl 的10秒超时安全停止。原 incident 和恢复复查保留；不把监控超时说成已证明存储故障，也不宣称修复底层存储。原门禁随后连续通过，未放宽超时或日志规则。模型、分辨率、AMP、阈值、预算均保持冻结。',
 '补充空输出测试暴露了原生零父点细窗口的 JIT 异常；在新目录 empty_guard.py 增加零父点/零子点保护，原生文件保持只读。原异常完整保留。早先900对真实实验均非空，未触发该分支；随后4方法×100对×3次前向在保护开启时逐元素复现原父点与最终坐标，因此非空结果身份未被重写。',
 'A1/A2两次全100对与历史精确一致。独立 CPU 排名/并列、820组最大可行配额、100对q=0和C禁用回退合同通过；真实全前向空输出检查见 real_empty_forward_contract.json。单元检查不替代各方法100对真实结果。',
 '原生代码、权重、配置与Phase1/PhaseT证据只读；新实现默认不影响生产入口。参考HEAD保持 d033f6dbb47c2e650390ba6053a22ac921fd5f3e；新目录尚未提交Git，以 experiment_code_identity.json 的逐文件SHA和run内源码快照冻结本次实现。','',
 '## 阶段 3 与产物','',
 '完整候选路径、SHA、逐场景规模、历史交集和预计时间/存储见 Stage3_Manifest_Approval.md。官方MD去重1488对/2场景、ScanNet1500对/100场景均有既往分析暴露；其他室外清单456845对/8场景不足10个。大清单需要有界流式存储设计，当前不能照搬约777GiB逐点缓存；不静默缩样本。',
 '阶段0报告：Stage0_Report.md；排名：Stage1_Report.md；必做机制：Stage2A_Report.md；主C：Stage2B_Report.md；选做：Stage2_Optional_Report.md。对应 CSV/JSON 均在本目录。原始逐对证据在 '+str(RUN)+'。',
 '阶段0–2没有位姿AUC结果；位姿与独立集最终评测按任务第3节等待明确清单批准。不得把历史位姿数值挪作本次结果。',
 '复查/重新汇总：cd /home/bupt/projects/SLiM-master && /home/bupt/miniconda3/envs/slim/bin/python -B diagnostics_coarse_rank/finalize.py --stage final。',
 '各方法中断后按已提交hash恢复：python -B diagnostics_coarse_rank/chain.py --methods METHOD --background；已有100有效提交只核验复用，不覆盖。独立集无默认自动启动命令；必须先确认 Stage3_Manifest_Approval.md 中所选清单哈希。']
 md('Final_Report.md','\n'.join(lines)+'\n')
 # Standalone scientific figure, no claims beyond diagnostic scene bootstrap.
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 vals=[main['recall'],ranking['recall'],c['recall'],stageb['comparisons']['C4-A1']['low']['recall']]
 y=np.array([x['mean'] for x in vals])*100;ci=np.array([x['CI95'] for x in vals])*100
 fig,ax=plt.subplots(figsize=(7.5,3.6));ax.errorbar(np.arange(4),y,yerr=np.stack([y-ci[:,0],ci[:,1]-y]),fmt='o',capsize=5,color='#235878')
 ax.axhline(0,color='gray',lw=1);ax.set_xticks(range(4),['B0 - A','B - B0','C4 - B','C4 - A']);ax.set_ylabel('Low-texture recall difference (pp)');ax.set_title('Diagnostic 100 pairs / 15 scenes; scene-cluster 95% CI');fig.tight_layout();fig.savefig(OUT/'paired_recall_differences.png',dpi=180);plt.close(fig)
 files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in OUT.iterdir() if p.is_file() and p.name!='delivery_receipt.json'}
 save(OUT/'delivery_receipt.json',dict(time=now(),status=status['status'],files=files,code=hashes,independent_model_runs=0))
 print(json.dumps(dict(status=status['status'],resource=resource,timing=tm)))
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['optional','final'],required=True);a=p.parse_args();optional() if a.stage=='optional' else finish()
