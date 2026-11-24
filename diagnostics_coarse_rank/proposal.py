from report import *
import csv,shutil
def proposal():
 inv=read(RUN/'inventory/route_frozen.json')
 # Costs are measured diagnostic workflow proxies, not claims about unseen data.
 timing={}
 for m in ['A1','B0','B','C4']:
  p=RUN/'timing'/m/'status.json'
  timing[m]=read(p)['mean_pair_median_seconds'] if p.exists() else float(np.mean([get(m,r)['timing']['forward_with_audit_seconds'] for r in recs()]))
 units=list(csv.DictReader((HISTORY/'pose/units.csv').open()))
 sec=np.array([float(r['seconds']) for r in units if r['seconds']])
 methods=['A1','B0','B','C4_if_coverage_gate'];N={c['dataset']:c['deduplicated_pairs'] for c in inv['candidates']}
 cost=[]
 sizes=[paths(m,r)[0].stat().st_size+paths(m,r)[1].stat().st_size for m in ['A1','B0','B','C4'] for r in recs()]
 perbytes=float(np.mean(sizes))
 for dataset,n in N.items():
  cost.append(dict(dataset=dataset,pairs=n,methods=methods,warmup_plus_3_forward_seconds=n*4*sum(timing.values()),
   legacy_pose_5seeds_4methods_serial_worker_seconds=n*20*float(sec.mean()),
   legacy_pose_three_worker_wall_seconds=n*20*float(sec.mean())/3,
   full_diagnostic_cache_projection_gib=n*4*perbytes/2**30,
   note='Forward proxy includes audit until current audit-free diagnostic timing is complete; pose historical per-unit wall includes old worker launch/import overhead, not pure solver time. Three-worker estimate assumes ideal CPU parallelism and is not guaranteed. Full-cache projection is a storage warning, not authorization to write it.'))
 res=dict(status='awaiting_explicit_manifest_confirmation',time=now(),independent_model_runs=0,candidates=inv['candidates'],sources=inv['inputs'],methods=methods,cost=cost,cost_basis=dict(diagnostic_forward_seconds=timing,legacy_pose_units=len(sec),legacy_pose_mean_worker_seconds=float(sec.mean()),legacy_pose_median_worker_seconds=float(np.median(sec)),source=str(HISTORY/'pose/units.csv')),storage=dict(host_free_gib=shutil.disk_usage('/mnt/c').free/2**30,linux_free_gib=shutil.disk_usage(RUN).free/2**30,reserve_gib=2),restrictions=['No independent inference without explicit approval of exact manifest SHA-256.','Full other-holdout list cannot use full diagnostic-cache persistence on current C: capacity; a bounded metric/pose commit design and concrete peak budget are prerequisites.','No silent downsampling; official two benchmarks are previously seen and cannot establish new-scene selection-independent evidence.','Only at most8 locally unused outdoor scene IDs, below10; physical landmark alias metadata across scene IDs unavailable.'])
 save(OUT/'stage3_proposal.json',res)
 lines=['# 阶段 3 清单确认资料','', '当前状态：等待用户明确确认，独立集模型运行数为 0。以下是阶段 0 在模型结果揭盲前冻结的完整候选；本文件不授权启动。','',
 '拟运行 A/B0/B，主 C 在获批清单上重新过同一无 GT 覆盖闸门后才运行。三份清单按来源/域分别汇总；oracle、R8/R12、仿射不进入阶段 3。','']
 for c,co in zip(inv['candidates'],cost):
  lines += [f"## {c['dataset']}",'',f"完整 pair 清单：{c['manifest']}",f"SHA-256：{c['sha256']}",f"帧与标定清单：{c['frame_manifest']}",f"帧清单 SHA-256：{c['frame_sha256']}",
   f"原始 {c['raw_pairs']} 对；无序/反向去重后 {c['deduplicated_pairs']} 对，全部结构可读；唯一图像 {c['unique_images']}；场景 ID {len(c['scenes'])}。最终 GT 可评场景数尚未实测。",
   f"逐场景完整对数："+'；'.join(f"{x['scene']}={x['pairs']}" for x in c['per_scene']),
   f"与本地已知使用清单无交集的场景：{c['known_independent_scenes']}。"]
  for label,x in c['overlap'].items():lines.append(f"- {label}：场景/图像/无序 pair 交集 = {x['scene_count']}/{x['image_count']}/{x['unordered_pair_count']}。")
  lines+=['',f"初步成本：四方法逐 pair 预热+三次前向约 {co['warmup_plus_3_forward_seconds']/3600:.2f} GPU 小时（当前诊断实测外推）；五种子位姿若沿历史逐单元启动方式，约 {co['legacy_pose_5seeds_4methods_serial_worker_seconds']/86400:.2f} 单工作进程天，理想三工作进程约 {co['legacy_pose_three_worker_wall_seconds']/86400:.2f} 天。该值含旧进程启动/import 开销，不是纯求解时间；未对独立数据实测，误差可能很大。",
   f"照搬本轮完整诊断缓存将约需 {co['full_diagnostic_cache_projection_gib']:.2f} GiB。阶段 3 若批准大清单，必须先完成有界流式指标/位姿提交与峰值预算，不能照搬完整矩阵或逐点缓存。",'']
 lines+=['官方 MegaDepth 与 ScanNet 均和以前 hybrid_match 基准/选择分析清单重叠；它们可作描述性复验或室内迁移描述，不能声称未用独立集。其他 8 场景未发现已知场景、规范路径图像或无序 pair 重叠；缺少跨 scene ID 物理地点别名表及完整离机选参历史。ScanNet 按索引第一场景整数合并重复扫描后缀；100 个序列恰为 100 个 sceneXXXX，均 15 对。','',
 '不缩小清单换取短任务。请明确确认所选完整清单及上述 SHA-256 后才能进入阶段 3；未确认时停下。']
 md('Stage3_Manifest_Approval.md','\n\n'.join(lines)+'\n');return res
if __name__=='__main__':
 x=proposal();print(json.dumps(x['cost'],ensure_ascii=False))
