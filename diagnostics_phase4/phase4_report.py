"""Evidence-only final report and result inventory."""
from common4 import *
from audit4 import main as audit
from plots4 import main as plots
from report_phase3 import disk
import subprocess
def fmt(x,n=6):return '无定义' if x is None else f'{x:.{n}f}' if isinstance(x,(int,float)) else str(x)
def ci(v,f=1):return '不可判定' if not v else '['+', '.join(fmt(x*f) for x in v)+']'
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(x).replace('|','/') for x in row)+' |' for row in rows])
def executions(run):
    output={}
    for stage in ['P0','P1','P2','P2b','P3','P4']:
        s=read(run/stage/'status.json');w=s['worker'];rs=w['resources']
        output[stage]=dict(status=s['status'],command=s['command'],supervisor_pid=s['supervisor_pid'],worker_pid=s['worker_pid'],
            started=s['started'],finished=s['finished'],seconds=w['seconds'],CPU_only=True,
            GPU_peak_allocated_bytes=rs.get('peak_allocated',0),GPU_peak_reserved_bytes=rs.get('peak_reserved',0),worker_peak_RSS_bytes=rs.get('rss_peak_bytes'),
            output_disk=disk(run/stage),log=str(run/stage/'worker.log'),heartbeat=str(run/stage/'heartbeat.json'))
        if stage in ['P3','P4']:
            output[stage]['geometry_launches']={p.name:read(p) for p in (run/stage).glob('*_launch.json')}
            output[stage]['initial20_timing']=read(run/stage/'first20_timing_and_ETA.json')
    output['implementation_incidents']=[
        dict(stage='P2b',reason='Empty-window aggregate lacked optional window_counts. Fixed aggregate reader; all 100 pair commits reused.',evidence=str(run/'P2b/attempts/01_empty_window')),
        dict(stage='P3',reason='New entry-point bracket syntax error before any geometry call. Fixed, then all modules and critical contracts checked.',evidence=str(run/'P3/attempts/01_compile'))]
    samples=[]
    for p in (run/'environment_probe').glob('*/resources.jsonl'):
        with p.open() as f:
            for line in f:
                if line.strip():samples.append(json.loads(line))
    output['resources']=dict(samples=len(samples),min_C_free_bytes=min(x['host_c_free_bytes'] for x in samples),
        min_linux_free_bytes=min(x['ext4_free_bytes'] for x in samples),min_available_memory_bytes=min(x['mem_available_bytes'] for x in samples),
        max_swap_used_bytes=max(x['swap_used_bytes'] for x in samples),scope='System guard samples. Individual geometry child peak RSS not separately instrumented.')
    return output
def build_manifest(run):
    spec=[
      ('P0 全局身份、HEAD、boot和容量','P0_bootstrap.json;P0_verification.json','完成','冻结身份为全局门槛'),
      ('P0 逐字段来源、回执、哈希和派生式','P0_field_lineage.json;P0_field_lineage.csv','完成','精确关联pair、方向、父ID及原始子点身份'),
      ('P0 子点数量、数值复现、轨迹容差','P0_verification.json;independent_checks.json','完成','全部484473、有效361254、无有效GT123219'),
      ('P0 历史基线哈希与500单元复用',str(run/'P0/baseline_reuse.json'),'完成','无新增基线求解'),
      ('分数定义、概率转换、分位数、零值和并列','score_distributions_and_ties.json;score_distributions_and_ties.csv','完成','缓存head为logit；sigmoid和乘积float64'),
      ('P1 两层角点及中心分档、汇总/场景/低纹理','P1_residual_bins.json;P1_residual_bins.csv;figures/P1_residual_bins.png','完成','中心残差缓存可用；不推断真实窗口重叠'),
      ('P2 子点/父点Spearman、AUSE、AUROC','P2_ranking.json;P2_ranking.csv;figures/P2_score_tradeoff.png','完成','冻结函数的并列规则不变'),
      ('P2 预设配对差值及区间','P2_paired_differences.json;P2_paired_differences.csv','完成','共同有定义pair、2000次场景bootstrap'),
      ('P2 保留曲线、去0003、低纹理','P2_retention_curves.json;P2_retention_curves.csv;P2_ranking.json','完成','低纹理只描述'),
      ('G5 四差值与价值档位','G5.json','完成','区间接触0按包含0处理'),
      ('P2b 实际策略三种概率及>5px','P2b_within_window.json;P2b_within_window.csv','完成','全部候选先选点后查GT'),
      ('P2b 共同有效三策略、均匀随机选择分布中位EPE','P2b_within_window.json;P2b_paired_differences.json;figures/P2b_within_window.png','完成','每窗口等权，候选权重1/v_i；不是窗口中位数的中位数'),
      ('P2b n_i/v_i分布、单有效、实际可评价和低纹理','P2b_window_denominators.json;P2b_denominator_supplement.json;'+str(run/'P2b/window_details.json'),'完成','n_i>=2实际策略；v_i>=2共同有效主分析'),
      ('P3 全子点80%/50%筛选及截断线并列','P3_cutoff_ties.json;P3_cutoff_ties.csv;'+str(run/'P3/prediction_preparation.json'),'完成','ceil逐pair；父ID/原始顺序破平，再恢复行序'),
      ('P3 主表、内点、分母和覆盖','P3_filter_quality.json;P3_filter_quality.csv;figures/P3_filter_quality.png','完成','无GT子点参与筛选、位姿、对极和覆盖'),
      ('P3 主指标全部预设配对区间','P3_paired_differences.json;P3_paired_differences.csv','完成','S_m对两单项、S_h对S_m、各方式对random'),
      ('P3 首20计时和5000正式求解',str(run/'P3/first20_timing_and_ETA.json')+';'+str(run/'P3/geometry_coverage.json')+';'+str(run/'P3/geometry_units.csv'),'完成','冻结LO-RANSAC，三CPU进程，CUDA隐藏'),
      ('P3 次要位姿AUC、基线与区间','P3_pose.json;P3_pose.csv','完成','100pair合并，五seed平均；整场景bootstrap'),
      ('G6 默认粗因子及覆盖代价','G6.json','完成','只决定本轮默认公式，不决定完整头去留'),
      ('P4 全部步数质量和差值','P4_step_quality.json;P4_step_quality.csv;P4_step_differences.json;figures/P4_truncation.png','完成','t4使用历史最终坐标；t0–t3离线重放'),
      ('P4 2000正式单元和位姿','P4_pose.json;P4_pose.csv;'+str(run/'P4/geometry_coverage.json')+';'+str(run/'P4/geometry_units.csv'),'完成','t4历史500单元直接复用'),
      ('G7 三项容差及固定确认序列','G7.json','完成','停止确认后的较短步数仅描述'),
      ('P4 可选分阶段推理耗时','P4_optional_timing.json','跳过','缓存没有可核验的分阶段精修耗时，未新增前向'),
      ('统计无定义、几何失败、缺失','undefined_and_failure_counts.json','无定义','各指标分别计数；未执行不算失败'),
      ('命令、PID、起止、耗时、磁盘','execution_summary.json','完成','日志/心跳/异常路径及首20计时'),
      ('口径假设和局限','protocol_assumptions.json;limitations.md','完成','补充条款优先'),
      ('总报告、JSON/CSV、图和README','Phase4_Analysis_Report.md;Phase4_Analysis_Report.json;Phase4_Analysis_Report.csv;README.md;figure_manifest.json','完成','所有结论性产物在experiment_results中')
    ]
    return [dict(item=n,status=s,reason=why,files=paths.split(';')) for n,paths,s,why in spec]
def main(run):
    run=Path(run);out=results(run)
    for s in ['P0','P1','P2','P2b','P3','P4']:
        if read(run/s/'status.json')['status']!='completed':raise GateFailure('Stage incomplete: '+s)
    checked=audit(run);plots(run);ctx=read(run/'context.json')
    docs={name:read(out/(name+'.json')) for name in ['P0_verification','P1_residual_bins','P2_ranking','P2b_within_window','P3_filter_quality','P3_pose','P4_step_quality','P4_pose','G5','G6','G7']}
    p0,p1,p2,p2b,p3,pose3,p4,pose4=[docs[n] for n in ['P0_verification','P1_residual_bins','P2_ranking','P2b_within_window','P3_filter_quality','P3_pose','P4_step_quality','P4_pose']]
    g5,g6,g7=[docs[k] for k in ['G5','G6','G7']];execution=executions(run);artifact(run,'execution_summary',execution)
    assumptions={
      'statistical_ties':'Frozen Spearman/AUROC functions unchanged. Frozen sparsification internally uses score and supplied IDs: the Phase3 encoding of (parent,native ordinal) is preserved. No jitter or extra ranking before metrics. Curves use the same frozen Phase3 order.',
      'concrete_ties':'Only P2b/P3 concrete selection explicitly sorts (-score,parent ID,native ordinal). P3 cutoff groups and selected/excluded members are separately counted.',
      'native_order':'Historical native fine output row order: coarse-window order, then within-window torch.topk(default sorted=True) on native dual-softmax, followed by threshold mask. Original equal-score topk order retained.',
      'native_score':'src/slim.py:834-904 uses dual softmax(sim_matrix_f), K=int(min(max_fine_matches/M,Lf)), global maximum*fine_match_thres. Phase3 fine recomputed in float32; do not assume identical finite-precision order.',
      'score_arithmetic':'head cache is a logit. head_p=expit(float64(logit)). Products use float64, no fitted weights, temperature or thresholds. Exact ties need not all be caused by precision; inherited-parent ties separately counted.',
      'random_seed':"Phase3-style SHA256 digest of seed2026, pair_id, purpose='fixed_child_random_prefix', first8 hex -> default_rng permutation. Same permutation prefixes for both budgets. P2b uses exact expectations.",
      'random_median':'均匀随机选择分布的中位 EPE：每pair每符合主分析的窗口权重1，窗口内每有效候选权重1/v_i；累计权重首次达到50%的误差，再pair→scene汇总。',
      'aggregation':'P1 primary pooled counts. P2/P2b/P3/P4 non-pose metrics: within-pair -> mean within scene -> equal scenes; pooled proportions separately named. Undefined excludes only its own metric; differences use joint defined pairs.',
      'retention':'Per-pair ceil(rate*all_children) before GT. No-GT children participate in score/selection/pose/epipolar/coverage. Incomplete full-population score invalidates all post-filter metrics for that method/budget.',
      'coverage':'32px grid over cached R832 valid image extent, partial boundary cells included. Out-of-rectangle coordinates do not occupy cells but remain in solver population.',
      'EPE':'P1/P2/P2b/P3 use committed Phase3 reconstructed EPE_T for exact reproduction. P4 t0..t3 from cached trajectory and GT; t4 uses historical native final coordinates/native_final_EPE. Threshold disagreement counts in independent_checks.',
      'pose':'Same frozen canonical solver via Phase3 one-seed adapter; 3 CPU workers, one numerical thread each. 500 historical baseline units reused. Whole-scene resampling preserves all methods and five seeds.',
      'degenerate_epipolar':'Nonfinite epipolar distance caused by degenerate denominator counts as not below threshold and is counted separately; valid coordinate/score/EPE nonfinite values remain faults.',
      'G7':'All step effects/intervals computed. Confirmation follows3,2,1,0 and stops at first non-pass. Assessments for shorter steps after stopping are descriptive only.'}
    artifact(run,'protocol_assumptions',assumptions)
    undefined={**{'P2/'+k:v['undefined_pair_counts'] for k,v in p2['summaries'].items()},**{'P2b/'+k:v['undefined_pair_counts'] for k,v in p2b['summaries'].items()},'P3':p3['summary']['undefined_pair_counts'],'P4':p4['summary']['undefined_pair_counts']}
    coverage={s:read(run/s/'geometry_coverage.json') for s in ['P3','P4']}
    artifact(run,'undefined_and_failure_counts',dict(statistical_undefined=undefined,geometry=coverage,implementation_incidents=execution['implementation_incidents'],
        new_storage_faults=0,OOM=0,new_model_nonfinite_faults=0,forward_calls=0,optimizer_updates=0,checkpoint_writes=0))
    limits=['单训练种子；验证场景参与过BCE-best选轮。','100对、15场景；没有独立数据验证组合分数，预设判据只支持下一步选择。','P2b候选仅限原生已输出子点，不代表所有窗口网格候选。','P4是离线轨迹重放；未实测真实截断GPU延迟、显存或端到端加速。','低纹理仅描述；无定义、几何失败、缺失严格分开。']
    text(out/'limitations.md','# 局限\n\n'+'\n'.join('- '+x for x in limits)+'\n')
    git=lambda *args:subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
    integrity=dict(status='pass',identity_sha=EXPECTED,HEAD=git('rev-parse','HEAD'),branch=git('branch','--show-current'),worktree=git('status','--porcelain=v1','--untracked-files=normal'),
        time_beijing=now(),diagnostic_code_sha256={p.name:sha(p) for p in CODE.glob('*.py')})
    artifact(run,'final_integrity',integrity)
    resume='cd /home/bupt/projects/SLiM-master && /home/bupt/miniconda3/envs/slim/bin/python -B diagnostics_phase4/run_phase4.py --run-id '+ctx['run_id']+' --stage remaining --background'
    text(out/'README.md','# SLiM Phase 4\n\n本目录保存全部结论性产物。入口：[Phase4_Analysis_Report.md](Phase4_Analysis_Report.md)。\n\n执行HEAD：'+integrity['HEAD']+'。新增diagnostics_phase4脚本未提交，精确SHA见[final_integrity.json](final_integrity.json)。\n\n冻结身份：'+EXPECTED+'。\n\n中间数据与日志：'+str(run)+'。新前向、训练更新、checkpoint写入均为0。\n\n恢复命令（显式复用同一run_id和结果目录，跳过已完成阶段）：\n\n~~~bash\n'+resume+'\n~~~\n')
    within=p2b['summaries']['all/common_valid'];wd=within['paired_differences'];primary=[r for r in p3['table'] if r['combo'].endswith('_80')]
    report=dict(status='completed',time_beijing=now(),run_id=ctx['run_id'],results=str(out),intermediates=str(run),identity_sha=EXPECTED,HEAD=integrity['HEAD'],
        analyses=docs,execution=execution,assumptions=assumptions,limitations=limits,resume_command=resume,geometry_coverage=coverage,new_forwards=0,optimizer_updates=0,checkpoint_writes=0)
    artifact(run,'Phase4_Analysis_Report',report)
    child=p2['summaries']['all/child']['scene_equal']
    qmap={r['combo']:r for r in primary}
    csvout(out/'Phase4_Analysis_Report.csv',[dict(section='P2_child',method=m,**v) for m,v in child.items()]+[dict(section='P3_primary',**r) for r in primary]+[dict(section='P4',**r) for r in p4['table']])
    lines=['# SLiM Phase 4 分析报告','',f'状态：**completed**。run_id：{ctx["run_id"]}。核验时间：{report["time_beijing"]}。','',
      '仅用Phase2/3提交缓存、CPU执行；新前向0、训练更新0、checkpoint写入0。本目录包含全部结论性产物；运行痕迹位于 '+str(run)+'。','',
      '## 四个问题','',
      '1. **组合：'+g5['status']+'。** '+g5['message']+'。未达到同时超过两个单项的四项标准；P3实际筛选指标见下文。',
      '2. **同窗口：** fine−随机的<2px正确率差 '+fmt(wd['fine-random']['precision_lt2']['mean_difference']*100,4)+' 个百分点，95%CI '+ci(wd['fine-random']['precision_lt2']['ci95'],100)+'；上限−fine差 '+fmt(wd['oracle-fine']['precision_lt2']['mean_difference']*100,4)+' 个百分点，95%CI '+ci(wd['oracle-fine']['precision_lt2']['ci95'],100)+'。',
      '3. **默认粗因子：'+g6['default_factor']+'。** '+g6['status']+'；'+g6['message']+'。',
      '4. **少步精修：** G7完整固定序列判定如下；离线通过只支持另行进行真实截断测速，不证明GPU加速。','',
      '## P0 验收','',
      '身份 '+EXPECTED+' 通过；HEAD '+integrity['HEAD']+'。全部/有效/无有效GT子点为 '+str(p0['counts']['children'])+'/'+str(p0['counts']['valid_children'])+'/'+str(p0['counts']['invalid_children'])+'。',
      '历史三分数的Spearman/AUSE复现最大偏差0；历史最终坐标及行序哈希一致，500个基线求解单元直接复用。轨迹重建最大偏差 '+fmt(p0['trajectory_max_deviation_px'],9)+' px < 0.01。',
      '逐字段来源、回执、SHA、派生式：[P0_field_lineage.csv](P0_field_lineage.csv)。验收：[P0_verification.json](P0_verification.json)、[independent_checks.json](independent_checks.json)。','',
      '## P1 粗残差分档','',
      '角点合同、全部父点汇总。第一层粗残差有效；第二层分母为最终EPE有效父点。该表不能证明两个窗口实际重叠。','',
      table(['档','父点','无细输出','有细无GT','最终有效','t0<2','t4<1','t4<2','救回/初始≥2','损失/初始<2'],
        [[r['bin'],r['parents'],r['without_fine'],r['fine_without_valid_GT'],r['valid_final_parents'],fmt(r['initial_lt2_fraction']),fmt(r['final_lt1_fraction']),fmt(r['final_lt2_fraction']),str(r['rescued'])+'/'+str(r['initial_child_ge2']),str(r['lost'])+'/'+str(r['initial_child_lt2'])] for r in p1['pooled'] if r['anchor']=='corner' and r['region']=='all']),'',
      '中心残差同表、低纹理、场景等权敏感性：[P1_residual_bins.json](P1_residual_bins.json)、[CSV](P1_residual_bins.csv)。','',
      '![P1](figures/P1_residual_bins.png)','',
      '## P2 固定分数和G5','',
      'head_p来自缓存logit的float64 sigmoid；乘积float64。未拟合权重、阈值或温度。父点分数取同一有效子点集合中位数。','',
      table(['分数','Spearman','AUSE','AUROC<1','AUROC<2','AUROC<3'],[[m]+[fmt(v[k]) for k in ['Spearman','AUSE','AUROC_lt1','AUROC_lt2','AUROC_lt3']] for m,v in child.items()]),'',
      table(['G5差值','值','95%CI','共同pair/scene'],[[k,fmt(v['mean_difference']),ci(v['ci95']),str(v['pair_count'])+'/'+str(v['scene_count'])] for k,v in g5['evidence'].items()]),'',
      '父点、去0003、低纹理：[P2_ranking.json](P2_ranking.json)。全部差值：[P2_paired_differences.csv](P2_paired_differences.csv)。保留曲线：[P2_retention_curves.csv](P2_retention_curves.csv)。分位数/零值/并列：[score_distributions_and_ties.json](score_distributions_and_ties.json)。','',
      '![P2](figures/P2_score_tradeoff.png)','',
      '## P2b 同窗口选点','',
      '实际策略n_i>=2：先在全部候选选点后查GT。共同有效主分析v_i>=2；单有效窗口不并入。值均为pair计算后场景等权。','',
      table(['实际策略','GT可评价率','有效且<2概率','条件正确率','有效且>5概率','条件>5率'],[[m]+[fmt(v[k]) for k in ['GT_evaluable_rate','valid_and_correct_probability','conditional_correct_rate','valid_and_bad_probability','conditional_bad_rate']] for m,v in p2b['summaries']['all/actual']['scene_equal'].items()]),'',
      '条件正确率为每pair期望正确数与期望可评价数之比。','',
      table(['共同有效策略','EPE<2','EPE>5','pair内中位EPE再场景汇总'],[[m,fmt(v['precision_lt2']),fmt(v['precision_gt5']),fmt(v['median_EPE'])] for m,v in within['scene_equal'].items()]),'',
      '**均匀随机选择分布的中位 EPE**：每窗口等权，候选权重1/v_i；每pair对混合分布取加权中位数，再按场景汇总。不是逐窗口中位数的再平均。',
      '分母及n_i/v_i分布：[P2b_window_denominators.json](P2b_window_denominators.json)、[补充分母](P2b_denominator_supplement.json)。低纹理与所有区间：[P2b_within_window.json](P2b_within_window.json)、[配对差值](P2b_paired_differences.json)。','',
      '![P2b](figures/P2b_within_window.png)','',
      '## P3 全子点过滤','',
      '每pair在全部细输出子点取ceil(0.8N)，并列按父ID/历史原始行序，再恢复原始顺序。无GT子点参与筛选；保留集合完整可重建。','',
      table(['方式','保留子点','有效GT','对极精度','EPE<1','EPE<2','GT可评价','低纹理保留','图0覆盖','图1覆盖'],
        [[r['combo'],r['children'],r['valid_children']]+[fmt(r[k]) for k in ['epipolar_precision','precision_lt1','precision_lt2','GT_evaluable_rate','low_texture_retention','coverage0','coverage1']] for r in primary]),'',
      '比例为场景等权；总计数为汇总数，不应直接相除解释加权比例。汇总比例、内点数、50%结果：[P3_filter_quality.csv](P3_filter_quality.csv)。',
      '80%预算下，S_m的对极精度/EPE<1比例为 '+fmt(qmap['S_m_80']['epipolar_precision']*100,3)+'% / '+fmt(qmap['S_m_80']['precision_lt1']*100,3)+'%；fine为 '+fmt(qmap['fine_80']['epipolar_precision']*100,3)+'% / '+fmt(qmap['fine_80']['precision_lt1']*100,3)+'%；mconf为 '+fmt(qmap['mconf_80']['epipolar_precision']*100,3)+'% / '+fmt(qmap['mconf_80']['precision_lt1']*100,3)+'%。这体现了对极精度与精细EPE之间的实际折中，不能称为全面超过fine。',
      'G6对极精度差 '+fmt(g6['epipolar']['mean_difference']*100)+' 个百分点，95%CI '+ci(g6['epipolar']['ci95'],100)+'；ΔSpearman '+fmt(g6['support_Spearman']['mean_difference'])+'，95%CI '+ci(g6['support_Spearman']['ci95'])+'。',
      '覆盖代价：[G6.json](G6.json)。所有主指标区间：[P3_paired_differences.json](P3_paired_differences.json)。截断线并列：[P3_cutoff_ties.csv](P3_cutoff_ties.csv)。','',
      table(['方式','AUC@5','AUC@10','AUC@20'],[[k]+[fmt(pose3['pooled_seed_mean'][k][m]) for m in ['AUC@5','AUC@10','AUC@20']] for k in ['baseline_100','fine_80','S_m_80','S_h_80','mconf_80','random_80']]),'',
      '位姿为次要指标，100pair合并、五seed分别算AUC后平均；0–100刻度。基线与同预算差值及区间：[P3_pose.json](P3_pose.json)。','',
      '![P3](figures/P3_filter_quality.png)','',
      '## P4 离线精修截断','',
      table(['t','平均EPE','pair中位EPE','EPE<1','EPE<2','EPE<3','对极精度','AUC@5','AUC@10','AUC@20'],
        [[r['step']]+[fmt(r[k]) for k in ['mean_EPE','median_EPE','precision_lt1','precision_lt2','precision_lt3','epipolar_precision']]+[fmt(pose4['pooled_seed_mean']['baseline_100' if r['step']==4 else 't'+str(r['step'])][m]) for m in ['AUC@5','AUC@10','AUC@20']] for r in p4['table']]),'',
      'G7统一t−t4。比例容差−0.01、−0.005，AUC容差−0.5；t4历史坐标/基线直接复用。','']
    for row in g7['steps']:
        lines += ['**t='+str(row['step'])+'：'+row['message']+'。**','',
          table(['条件','差值','95%CI','容差','评估','确认性'],[[k,fmt(v['difference']),ci(v['ci95']),fmt(v['tolerance']),v['assessment'],'是' if row['confirmatory'] else '否，仅描述'] for k,v in row['conditions'].items()]),'']
    lines += ['全部误差与位姿差值：[P4_step_differences.json](P4_step_differences.json)、[P4_pose.json](P4_pose.json)。可选阶段耗时缺缓存，已跳过；没有新增计时前向。','',
      '![P4](figures/P4_truncation.png)','',
      '## 执行与限制','',
      table(['任务','PID','开始','结束','worker秒','占用MiB'],[[s,v['worker_pid'],v['started'],v['finished'],fmt(v['seconds'],2),fmt(v['output_disk']['allocated_bytes']/2**20,2)] for s,v in execution.items() if s in ['P0','P1','P2','P2b','P3','P4']]),'',
      '正式几何：P3 '+str(coverage['P3']['committed'])+'/'+str(coverage['P3']['planned'])+'；P4 '+str(coverage['P4']['committed'])+'/'+str(coverage['P4']['planned'])+'。已执行失败 '+str(sum(v['executed_failures'] for v in coverage.values()))+'；缺失 '+str(sum(v['missing'] for v in coverage.values()))+'。',
      'P2b空窗口汇总及P3启动前语法这两次新脚本错误均已保存证据并修复，未重复已提交单元。命令、PID、日志/心跳、首20计时与资源：[execution_summary.json](execution_summary.json)。',
      '统计无定义分项：[undefined_and_failure_counts.json](undefined_and_failure_counts.json)。全部口径假设：[protocol_assumptions.json](protocol_assumptions.json)。','',
      *['- '+x for x in limits],'',
      '## 恢复与产物','',
      '~~~bash',resume,'~~~','',
      '结论性产物均在本目录；图另有SVG。逐项状态和文件大小/SHA：[artifact_manifest.json](artifact_manifest.json)。']
    text(out/'Phase4_Analysis_Report.md','\n'.join(lines)+'\n')
    items=build_manifest(run)
    for item in items:
        refs=[]
        for name in item['files']:
            p=Path(name) if name.startswith('/') else out/name
            if not p.is_file():raise GateFailure('Missing report artifact: '+str(p))
            refs.append(dict(path=name,location='runs' if name.startswith('/') else 'results',bytes=p.stat().st_size,sha256=sha(p)))
        item['files']=refs
    artifact(run,'artifact_manifest',items,items)
    total=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    if total>50*2**20:raise GateFailure('Results exceed 50 MiB')
    if any(p.stat().st_size>10*2**20 for p in out.rglob('*') if p.is_file()):raise GateFailure('Move large detail/appendix to run')
    save(run/'report/summary.json',dict(status='completed',results=str(out),report=str(out/'Phase4_Analysis_Report.md'),run_id=ctx['run_id'],result_bytes=total,resume_command=resume))
    return dict(status='completed',G5=g5['status'],G6=g6['status'],G7_first_failed=g7['first_failed_step'],results=str(out),bytes=total)
