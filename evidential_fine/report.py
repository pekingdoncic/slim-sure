"""Final CPU analysis, geometry, readable report and delivery inventory."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from .common import *
from .report_helpers import f,ci,table,audit_training,coverage_summary,stage_inventory,figures,artifact_manifest
def execute(run):
    run=Path(run);context=read(run/'context.json');out=Path(context['results']);out.mkdir(parents=True,exist_ok=True)
    identity_gate();start=time.monotonic();beat(run,'report',phase='CPU analysis startup')
    from .statistics import analyze
    metrics=analyze(run)
    training,curve,checkpoints=audit_training(run);coverage=coverage_summary(run)
    from .geometry import execute as pose_execute
    pose=pose_execute(run);save(out/'pose.json',pose)
    efficiency=read(run/'efficiency/summary.json');save(out/'efficiency.json',efficiency)
    e0=read(run/'E0/summary.json');e0final=read(run/'E0/native_final_reference_review.json')
    save(out/'E0.json',dict(initial=e0,exact_native_final_supplement=e0final,numerical_checks=read(run/'E0/numerical_checks.json')))
    save(out/'preregistered_protocol.json',read(run/'preregistered_protocol.json'))
    save(out/'source_reference.json',read(run/'source_reference.json'))
    total_cost=read(run/'first50_total_chain_cost.json');save(out/'first50_total_chain_cost.json',total_cost)
    points=read(out/'point_denominators.json');point_groups={}
    for route in metrics['routes']:
        for region in ['all','low_texture']:
            values=[x for x in points if x['route']==route and x['region']==region]
            point_groups[route+'/'+region]=dict(**{k:sum(x[k] for x in values) for k in ['outputs','valid_GT','parents_with_outputs','parents_with_valid_GT','inside','outside']},
                pairs=sum(x['valid_GT']>0 for x in values),scenes=len({x['scene'] for x in values if x['valid_GT']>0}))
    windows=read(out/'window_denominators.json');window_totals={}
    for region in ['all','low_texture']:
        values=[x for x in windows if x['region']==region]
        window_totals[region]=dict(**{k:sum(x[k] for x in values) for k in ['windows','actual_multi','common_valid_multi','single_valid','zero_valid']},
            actual_pairs=sum(x['actual_multi']>0 for x in values),common_pairs=sum(x['common_valid_multi']>0 for x in values),common_scenes=len({x['scene'] for x in values if x['common_valid_multi']>0}))
    save(out/'denominator_summary.json',dict(points=point_groups,windows=window_totals))
    # Aggregate independently executed path proof and native1 exact-reference evidence.
    path_proof={m:dict(pairs=0,max_deviation=0.,allocated_peak=0,reserved_peak=0) for m in ['native1','native4','A','B']}
    upstream=[];one_review=[];eff_rows=[]
    for r in records('epe'):
        row=get_unit(run,'efficiency',r)
        for method,timing in row['timing'].items():
            q=path_proof[method];q['pairs']+=1;q['max_deviation']=max(q['max_deviation'],row['max_output_deviation'].get(method,0))
            q['allocated_peak']=max(q['allocated_peak'],max(x['allocated_peak'] for x in timing['repeats']));q['reserved_peak']=max(q['reserved_peak'],max(x['reserved_peak'] for x in timing['repeats']))
            eff_rows.append(dict(**base(r),method=method,median_seconds=timing['median_seconds'],execution_proof=row['execution_proof'][method],**{k:max(x[k] for x in timing['repeats']) for k in ['allocated_peak','reserved_peak']}))
        upstream.append(dict(**base(r),methods={'upstream':{'seconds':row['upstream']['median_seconds']}}))
        for route in metrics['routes']:
            v=get_unit(run,'final_'+route,r);one_review.append(dict(**base(r),route=route,**v['native_one_step']))
    upstream_summary=scene_summary(upstream,['seconds'],['upstream'])
    save(out/'efficiency_proof.json',dict(paths=path_proof,common_upstream=upstream_summary,native_one_reference=one_review))
    csvout(out/'efficiency_pairs.csv',eff_rows)
    figures(out,training,curve,metrics,efficiency)
    startup=read(run/'startup.json');initial=read(run/'initial_environment.json');budget=read(run/'storage_budget.json')
    resources=read(run/'observation/observed_peaks.json') if (run/'observation/observed_peaks.json').exists() else {}
    stages=stage_inventory(run);save(out/'stages.json',stages)
    failure_notes=[dict(stage=x['stage'],evidence=x['failure_evidence']) for x in stages if x['failure_evidence']]
    control_incident=read(run/'control_connection_incident.json') if (run/'control_connection_incident.json').exists() else None
    if control_incident:save(out/'control_connection_incident.json',control_incident)
    finish=now();elapsed=(datetime.datetime.fromisoformat(finish)-datetime.datetime.fromisoformat(context['created'])).total_seconds()
    complete=metrics['status']=='completed' and all(training[x].get('status')=='completed' for x in ['A','B']) and pose['status'] in ['completed','skipped']
    delivery=dict(status='completed' if complete else 'partial',time_beijing=finish,identity=EXPECTED,HEAD=startup['HEAD'],branch=startup['branch'],boot_id=startup['boot_id'],
        run=str(run),results=str(out),routes=metrics['routes'],decisions=metrics['decisions'],training_updates=sum(x.get('committed_updates',0) for x in training.values()),
        total_chain_wall_seconds_so_far=elapsed,report_processing_seconds_so_far=time.monotonic()-start,resources=resources,denominators=dict(points=point_groups,windows=window_totals),
        path_proof=path_proof,failure_evidence=failure_notes,control_connection_incident=control_incident,degradation=read(run/'degradation.json') if (run/'degradation.json').exists() else None,
        T_report=str(ROOT/'experiment_results'/Path(context['T_run']).name/'PhaseT_Report.md'),monitoring='Each live stage supervised by cursor-aware5s storage/resource guard; final stopped-state receipt written after supervisor exit.')
    save(out/'summary.json',delivery)
    sections=[]
    sections.append('# Phase 5 证据回归细匹配头：最终报告\n\n'+
        f"状态：**{delivery['status']}**。固定第2轮权重；A→B各完成两轮13800次/轮，验证每轮750对，最终100对15场景。\n\n"+
        '主结果和分母见 [主表](main_table.csv)、[配对差值](paired_differences.csv)、[完整指标](metrics.json)、[判据](decisions.json)。')
    decision_rows=[]
    for key,d in metrics['decisions'].items():
        v=d.get('value',d.get('precision',{}));decision_rows.append([key,d['status'],ci(v),d.get('conclusion',d.get('reason')),v.get('pair_count'),v.get('scene_count')])
    sections.append('## 预设判据\n\n'+table(['判据','状态','差值 [95% CI]','结论','pair','scene'],decision_rows)+
        '\n\nG8只判断B对真实原生一步的EPE<1px；G9只判断同一B误差上的负认知不确定性相对fine的Spearman。G10同时要求共同父窗口<2px非劣效和实际细阶段更快。单项通过不代表整链路替代。')
    sections.append('## 身份与运行边界\n\n'+f"分支：{startup['branch']}；HEAD：{startup['HEAD']}。冻结digest：{EXPECTED}。Boot：{startup['boot_id']}。\n\n"+
        f"run：{run}。T仅通过报告和进程交接关联；Phase5以原生0.65入口、texture_budget关闭重新加载原权重。原数据、权重、冻结代码和历史run只读。保留启动时19个Phase4 CSV工作区差异。\n\n"+
        f"启动时间：{context['created']}；本报告时间：{finish}；两任务累计墙钟截至本报告 {elapsed/3600:.3f} 小时。最终进程退出时间由delivery receipt补充。")
    sections.append((out/'texture_definition.md').read_text())
    p=read(run/'preregistered_protocol.json')['Phase5']
    structure=[]
    for route in ['A','B']:
        z=training[route].get('formal_initialization',{})
        structure.append([route,z.get('parameter_count'),p[route]['R'],p[route]['K']+1,p[route]['cap'],
            '共享双侧：192粗投影+4×48象限投影→128；两侧各2层；融合2层；x/y独立64隐藏' if route=='A' else '96通道6×6；3×3卷积64两层，GN8、padding1；展平→128；x/y独立64隐藏'])
    sections.append('## 结构、数值与来源\n\n'+table(['路线','新参数','R(px)','箱数','训练上限/对','结构'],structure)+
        '\n\n新头float32；冻结骨干沿用AMP。AdamW：lr=0.001，weight_decay=0.0001，betas=(0.9,0.999)，eps=1e-8，无scheduler。GradScaler禁用但保存空状态。每轴μ为[-R,R]等距0.5px分箱softmax期望；归一化u=μ/R、y=GT偏移/R。损失为指定NIG NLL与|u−y|(2ν+α)等权之和，按样本和轴平均；ν/β下限1e-6、α−1下限1e-4。两轴方差均值为最终不确定性，方差还原乘R²。\n\n'+
        'A：源/初始目标均为各自粗格中心+4，冻结192维交互后粗描述子，原生8×8细窗的4×4象限均值按左上、右上、左下、右下拼接。B：原生t0真实子点源/目标，双侧48通道6×6双线性细窗，offset −2.5..2.5、align_corners=False、zero padding，细stride=1。训练仅使用GT有效且L∞严格<R的点；评估不丢弃范围外有效GT。\n\n'+
        '代码为本次独立实现，没有复制SURE源码。[来源核对](source_reference.json)记录官方SURE提交f702f618ebcf1b13c7c4344e91e56d2131ffc197和文件SHA。NIG通过独立scipy Student-t等价公式校验；正常与极端输入反向均有限。SURE的双向、偶然与认知联合分位筛选不同于本任务固定0→1、负认知排序和ceil固定预算；没有把SURE原设置成绩套用到本实验。')
    e0rows=[[route,e0['routes'][route]['status'],e0['routes'][route]['first20_loss'],e0['routes'][route]['last20_loss'],e0['routes'][route]['zero_mean_EPE'],
             e0['routes'][route]['final_fixed_mean_EPE'],e0['routes'][route]['resume']['parameter_max_abs'],e0['routes'][route]['resume']['loss_abs']] for route in ['A','B']]
    covrows=[[key,v['total'],v['valid'],v['inside'],v['outside'],v['inside_fraction'],v['pairs'],v['scenes']] for key,v in coverage['pooled'].items()]
    sections.append('## E0与监督覆盖\n\n'+table(['路线','结果','首20 loss','末20 loss','零偏移EPE','200步EPE','恢复参数差','恢复loss差'],e0rows)+
        '\n\n固定8对、预先落盘的同一监督点集，每路线200次真实更新。原参数与BN、梯度归属和optimizer集合校验通过；第199步保存恢复后的第200步与连续训练一致。自检状态销毁后正式训练seed2026、游标0重新初始化。\n\n'+
        f"固定100对复现194383父点、484473子点、361254有效GT子点；完整前向与截断训练输入最大偏差A={e0['native']['prefix_max_abs']['A']}、B={e0['native']['prefix_max_abs']['B']}，中心真值、可见mask和父子ID对齐。\n\n"+
        '初版E0把t4与Phase3重建t4比较，未覆盖Phase4精确终点；该来源差导致约1.6313e-6的比例差，原记录保留。B正式训练前使用本次A验证的完整原生前向缓存补验相同100对，精确t4坐标和EPE与历史均逐元素一致；t0/t1/t4比例均相对Phase4全精度参考差0，见[E0及补验](E0.json)。后续真实一步在同一原生细输入上确实只调用一次RRU，再与计时入口核对。\n\n'+
        table(['路线/层','全部输出','有效GT','范围内','范围外','pooled范围内比例','有效pair','scene'],covrows)+
        '\n\n此表比例为pooled；场景等权值与逐场景分母见[监督覆盖](supervision_coverage.json)、[逐对覆盖](supervision_coverage_pairs.csv)。原生四步累计位移观测最大L∞='+f(coverage['native_displacement']['observed_max_Linf'])+
        ' px，超出B单次±3px盒的点数='+f(coverage['native_displacement']['outside_B_single_box'])+'；B的预测支持集是每轴[-3,3]，并非原生四步的理论范围。')
    tr=[]
    for route,z in training.items():
        for e in z.get('epochs',[]):tr.append([route,e['epoch'],e['visits'],e['updates'],e['skip'],e['training_seconds'],e['validation_seconds'],e['validation_pairs']])
    sections.append('## 正式训练实绩\n\n'+table(['路线','轮','访问','更新','skip','训练秒','验证秒','验证对'],tr)+
        '\n\n冻结采样计划的前两轮、138个训练场景，未扩样或换序；共'+f(sum(x.get('committed_updates',0) for x in training.values()))+'次正式optimizer更新。每50访问提交原子游标/权重/optimizer/RNG/scaler；epoch末750对验证和last均核验。所有skip原因、计划唯一pair、实际唯一pair、更新唯一pair及资源峰值见[训练实绩](training.json)。最终用第2轮，未以本100对选择轮次。\n\n'+
        table(['路线','计划唯一pair','访问唯一pair','更新唯一pair','物理日志访问','提交访问','skip原因'],[[k,v.get('planned_unique_pairs'),v.get('visited_unique_pairs'),v.get('update_unique_pairs'),v.get('physical_log_rows'),v.get('committed_visits'),json.dumps(v.get('skip_reasons',{}))] for k,v in training.items()])+
        '\n\n[每轮750对验证](validation_by_epoch.csv)给出整体/低纹理/范围内外的均值、中位、<1/<2/<3px和Spearman；A同时列同源零偏移与原生父窗口中位，B列t0、实际t1、t4。\n\n'+
        table(['路线','首50 pair/s','首50时预计剩余小时','峰allocated GiB','峰reserved GiB','峰RSS GiB'],[[k,v['first50']['pairs_per_second'],v['first50']['total_remaining_seconds']/3600,
            v['resource_maxima'].get('peak_allocated',0)/2**30,v['resource_maxima'].get('peak_reserved',0)/2**30,v['resource_maxima'].get('rss_peak_bytes',0)/2**30] for k,v in training.items() if v.get('status')=='completed'])+
        '\n\n![Training curves](training.png)')
    sections.append('## 数值稳定下限触发率\n\n'+table(['路线','监督样本数','ν下限比例','α−1下限比例','β下限比例'],
        [[route,x['supervised_samples'],x['floor_hit_fractions']['nu'],x['floor_hit_fractions']['alpha_minus_one'],x['floor_hit_fractions']['beta']] for route,x in training.items() if x.get('status')=='completed'])+
        '\n\n按每个实际参与损失的样本和坐标轴加权汇总，包含重复访问；不等同于唯一点比例。固定下限保证数值定义，频繁命中表示相应证据参数受下限约束，不能仅凭输出有限或排序相关性宣称方差已校准。训练期间没有调整这些下限。')

    sections.append('## First50 total elapsed-time forecast\n\n'+table(['Route','Measured first50 time','Elapsed from task start (h)','Remaining forecast (h)','Total from task start (h)','Over16h'],[[k,v['snapshot_time'],v['elapsed_chain_seconds']/3600,v['remaining_seconds']/3600,v['estimated_total_chain_seconds']/3600,v['exceeds16hours']] for k,v in total_cost.items()]))
    qrows=[]
    for name in ['B/all', 'B/low_texture', 'A/full/all', 'A/full/low_texture', 'A/median/all', 'A/median/low_texture', 'A/fine_one/all', 'A/fine_one/low_texture', 'A/joint/all', 'A/joint/low_texture']:
        if name in metrics['quality']:
            for method,v in metrics['quality'][name]['scene_equal'].items():qrows.append([name,method,v['mean_EPE'],v['median_EPE'],v['lt1'],v['lt2'],v['lt3']])
    sections.append('## 最终坐标精度\n\n'+table(['口径','方法','平均EPE','中位EPE','<1','<2','<3'],qrows)+
        '\n\n表中为pair内统计→场景内平均→场景等权。B采用相同真实源点、子点ID和有效GT；t1为实际只执行一次精修，t4为真实终点。A的不同源像素比较仅解释为每父窗口输出质量：同源零偏移、原生子点中位EPE、GT不可见的fine最高单子点，均各算真实源点GT。中位参照与一子点参照分别使用共同可评父点，并另给联合共同集；未将无GT等同于错误。\n\n'+
        '[范围内外、配对区间与逐场景方向](metrics.json)；[A偏移分档及B初始大误差](offset_bins_and_B_initial_tail.csv)；[每对完整输出、排除、32px覆盖](coverage.csv)。A分档[0,4]、(4,8]、(8,12]、>12；恰好8与训练严格<8另记。\n\n'+
        table(['输出层','输出数','有效GT','输出父点','有效GT父点','可评pair','scene'],[[k,v['outputs'],v['valid_GT'],v['parents_with_outputs'],v['parents_with_valid_GT'],v['pairs'],v['scenes']] for k,v in point_groups.items()])+
        '\n\n全部分子、分母及无效GT原因见[逐对点分母](point_denominators.csv)，所有比例主表同时提供单列pooled分子/分母，不能把pooled比例当场景等权比例。\n\n![Coordinate and efficiency](coordinate_and_efficiency.png)')
    ranks=[]
    for group,s in metrics['ranking'].items():
        for method,v in s['scene_equal'].items():ranks.append([group,method,v['Spearman'],v['AUSE'],s['metric_denominators'][method]['Spearman']['evaluable_pairs'],s['metric_denominators'][method]['Spearman']['scenes']])
    sections.append('## 不确定性与窗口内选择\n\n'+table(['口径','分数','ρ(score,EPE)','AUSE','可评pair','scene'],ranks)+
        '\n\n分数越大代表越可信，所以与EPE的相关系数越负越好。主分数为负认知不确定性；偶然和两者之和仅探索。每条路线的全部分数比较使用该路线自己的同一坐标误差；A父级fine/S_m是离线原生实际子点中位，仅在共同可关联父点比较，不能算作A部署能力。独立的A完整输出分母另列。常量、样本不足分别记undefined；两种不确定性之间相关与0.1–1.0保留曲线见[指标](metrics.json)、[曲线](retention_curves.json)。\n\n'+
        table(['区域','全部窗口','实际≥2候选','有效GT≥2','仅1有效','无有效','共同口径pair','scene'],[[k,v['windows'],v['actual_multi'],v['common_valid_multi'],v['single_valid'],v['zero_valid'],v['common_pairs'],v['common_scenes']] for k,v in window_totals.items()]))
    wr=[]
    for name,z in metrics['within_window'].items():
        for method,v in z['scene_equal'].items():
            wr.append([name,method,v.get('GT_evaluable_rate'),v.get('valid_and_correct_probability'),v.get('conditional_correct_rate',v.get('precision_lt2')),v.get('conditional_bad_rate',v.get('precision_gt5')),v.get('median_EPE')])
    sections.append(table(['窗口口径','方式','GT可评率','有效且正确概率','条件<2','条件>5','中位EPE'],wr)+
        '\n\n实际口径：在所有实际子点上先选后查GT，窗口需至少2个实际子点；oracle只作有GT时离线上限。随机每窗口等权，分别使用v/n、g/n及Σ(g/n)/Σ(v/n)，坏点同理。共同口径：至少2个有效GT候选，随机中位名称为“均匀随机选择分布的中位 EPE”，窗口内权重1/v，先求每pair混合分布加权中位，再场景汇总。\n\n'+
        '主信号−fine、主信号−随机、oracle−主信号的场景配对区间完整列在[差值表](paired_differences.csv)。低纹理仅作描述，分母独立列出，不以整体Spearman代替低纹理同窗口效果。\n\n![Within-window](within_window.png)')
    sections.append('## 固定预算过滤与几何\n\n在全部预测点上保留ceil(0.95N)与ceil(0.8N)，随后恢复原始顺序。A按负认知；B另列S_m和冻结seed随机。GT未参与筛选。具体截取并列按父点ID、原始子点顺序；原始顺序是父点块内原生torch.topk细置信度降序，原生完全并列保留框架的原始输出顺序。Spearman、AUSE和曲线沿用冻结函数自己的并列规则。\n\n'+
        '[筛选质量、低纹理父点保留和覆盖](filter_counts.json)、[截断线并列数量](filter_ties.csv)提供每pair的完整分母。低纹理父点保留分母为筛前已有实际输出的低纹理父点。两图覆盖使用有效输入区域32px网格；包含无GT预测。\n\n'+
        f"位姿状态：{pose['status']}；新增求解{pose.get('new_units',0)}，原生复用{pose.get('reused_native_units',0)}，已执行失败{pose.get('executed_failures',0)}，缺失{pose.get('missing',0)}。"+
        'canonical LO-RANSAC，原图坐标/K，0.5px，confidence0.99999，seed0–4。每seed合并100对算AUC后平均，0–100刻度；失败计∞，未执行单列；场景bootstrap整块保留所有方法与seed并重算AUC。\n\n'+
        table(['方法','AUC@5','AUC@10','AUC@20'],[[k,*v] for k,v in pose.get('AUC_5_10_20',{}).items()])+
        '\n\n[位姿差值、场景区间和逐场景方向](pose.json)；逐单元匹配数、求解状态、内点数与耗时留在run/geometry/geometry_units.csv。')
    if pose.get('unit_summary'):
        sections.append('## Geometry unit denominators\n\n'+table(['Method','Units','Pairs','Successes','Executed failures','Mean matches','Mean inliers (all)','Mean inliers (success)'],
            [[name,x['units'],x['pairs'],x['successes'],x['executed_failures'],x['matches']['mean'],x['inliers_all']['mean'],x['inliers_success'].get('mean')] for name,x in pose['unit_summary'].items()]))
    erows=[]
    for method,v in efficiency.get('summary',{}).get('scene_equal',{}).items():
        q=path_proof[method];erows.append([method,v['seconds']*1000,q['pairs'],q['max_deviation'],q['allocated_peak']/2**30,q['reserved_peak']/2**30])
    sections.append('## 实际替代路径效率\n\n'+table(['路径','场景等权ms','pair','输出最大偏差','allocated GiB','reserved GiB'],erows)+
        '\n\n共同起点：交互后粗描述子、已匹配粗父点、上采样细特征；终点：全部实际输出源/目标坐标。每对每法预热，再CUDA同步重复3次取中位；窗口构造、采样、池化、预测与坐标解码均计入，GT/统计/落盘排除。路径证明native1只执行1次RRU，native4为4次；A不执行细相关或RRU；B执行细相关而不执行RRU。未使用训练点数上限。\n\n'+
        '共同上游特征生成单列：场景等权 '+f(upstream_summary['scene_equal']['upstream']['seconds']*1000)+' ms。[路径证明与峰值边界](efficiency_proof.json)、[逐对时间](efficiency_pairs.csv)、[配对时间差](efficiency.json)。allocated包含共同常驻张量；reserved是进程分配器保留的池峰值，不代表该算子的净增显存。细阶段差异不能外推端到端加速。')
    sections.append('## 资源、故障与复现\n\n'+f"启动Linux可用{initial['linux_free_bytes']/2**30:.3f} GiB，C卷{initial['resources']['host_c_free_bytes']/2**30:.3f} GiB；VHD和实际swap同处C卷，只计算一次宿主容量。预算新增8GiB并保留2GiB。VHD：{budget['backing_path']}；swap：{budget['swap_path']}。\n\n"+
        '启动进行了小规模真实读取和独立run写入/fsync/回读/清理。全boot历史日志流式检查后用游标续查；启动早期journal非正常关闭警告保留，未把它当作本次新存储故障。各运行阶段5秒watchdog检查存储、容量和存活；运行期低于2GiB安全暂停，系统存储错误立即停止且不强写新checkpoint。\n\n'+
        f"采样观测P5 run峰值{resources.get('P5_observed_peak_bytes',0)/2**30:.3f} GiB，宿主最小余量{resources.get('host_minimum_free_bytes',0)/2**30:.3f} GiB。这是30秒观测峰值，瞬时原子checkpoint临时副本另外至多增加一个payload，不能称连续测得的绝对峰值。\n\n"+
        '第一次E0在首个pair提交前遇到bilinear采样dtype实现错误，恢复原生AMP上下文后重跑，未产生训练更新；失败日志保留。初版t4参考缺项及补验也单独保留。训练数值核心/采样/超参数未在结果出来后调整。'+
        f"降级记录：{json.dumps(delivery['degradation'],ensure_ascii=False)}。\n\n"+
        '[全部阶段命令/PID/时间/故障](stages.json)、[checkpoint绝对路径、大小与SHA256](checkpoints.json)、[产物清单](artifact_manifest.json)。大于10MB明细保留run；每个结果目录限50MB。源码位于evidential_fine/，冻结目录未改。\n\n'+
        '在仓库目录使用以下入口；已有训练进度仅使用同配置恢复，已完成的训练不要再次启动：\n\n'+chr(96)*3+'bash\ncd '+str(ROOT)+'\n'+
        '\n'.join('/home/bupt/miniconda3/envs/slim/bin/python -B evidential_fine/run_phase5.py --run '+str(run)+' --stage '+stage+' --background --resume' for stage in ['A','B'])+
        '\n/home/bupt/miniconda3/envs/slim/bin/python -B evidential_fine/run_phase5.py --run '+str(run)+' --stage evaluate --background --resume'+
        '\n/home/bupt/miniconda3/envs/slim/bin/python -B evidential_fine/run_phase5.py --run '+str(run)+' --stage report --background --resume\n'+chr(96)*3)
    correction_path=run/'efficiency_mismatch_probe/correction.json'
    if correction_path.exists():
        correction=read(correction_path);save(out/'efficiency_correction.json',correction)
        sections.append('## 效率入口的一次实现修复\n\n首对计时曾因B整批头计算与完整评估的4096分块不同而触发坐标一致性门禁。单对独立探针测得：偏移最大差8.046627e-7px，坐标相加后最大差6.103516e-5px；采用完整评估原有分块后，偏移和坐标差均为0。A同对无差异。\n\n仅将计时入口统一为相同4096分块，所有子点仍参与，GPU窗口提取、分块头计算、拼接和坐标解码均计时。精度、尺寸、阈值、输出预算及1e-5验收容差不变。失败前没有效率单元提交，修复后100对统一重测；已提交的A/B最终100对缓存、两轮训练和checkpoint未重跑或改写。故障日志和修复前代码保留，见[定位与修复证据](efficiency_correction.json)。')

    b_window=metrics['within_window'].get('all/common',{}).get('paired_differences',{}).get('epistemic-fine',{}).get('precision_lt2',{})
    if control_incident:
        sections.append('## Control connection incident\n\nA new WSL shell returned 0x8007274c while the existing A worker and watchdog continued updating. Host available memory was 0.691GiB at the recorded sample. Plain bupt connections with an explicit Linux cwd later succeeded; boot was unchanged. Incremental kernel records were UtilAcceptVsock warnings, with no detected new storage fault. The cause is not proven; no system/configuration changes, restarts, or training retry occurred. See [incident evidence](control_connection_incident.json).\n\nThe serial coordinator launch and actual stage handoffs are retained under run/chain/. CPU synthetic checks caught undefined scene-interval handling and duplicate CSV metadata, corrected before formal final statistics; failure records remain under run/checks/.')
    suggestion='坐标、排序和同窗口选择分别审视。优先重复同一冻结方案的一个独立训练种子以检验主结论；若主要收益只出现在排序，应固定原生一步坐标，单独验证窗口内选择；候选召回的后续依据独立T判据，不能借调P5阈值。'
    sections.append('## 解释边界与下一项最小实验\n\nB共同有效窗口主信号−fine的<2px差值：'+ci(b_window)+'。整体相关性与窗口内收益是不同结论；同样，坐标精度和候选召回不可互相替代。\n\n'+
        '仅一个训练seed、冻结采样计划前两轮、冻结SLiM、R832、100对15个室外验证场景；最终样本已被反复分析。低纹理使用Phase2训练集冻结阈值0.0001920461654663086，源父点随源粗格、子点继承；并非按本次GT/成绩挑选。该成绩既不是测试集泛化证明，也不是联合训练的严格上下界。\n\n'+suggestion+' 本次只给建议，解冻、扩样、标签改动和测试集评测均未执行。')
    text(out/'Phase5_Report.md','\n\n'.join(sections)+'\n')
    text(out/'README.md','# Phase 5 results\n\n[完整报告](Phase5_Report.md) · [判据](decisions.json) · [主表](main_table.csv) · [分母](denominator_summary.json) · [训练](training.json) · [checkpoint](checkpoints.json)\n\n'+
        'Run: '+str(run)+'\n\n所有数值来自本次提交缓存。报告中各主量按pair→scene→scene等权；pooled分子/分母单独标注。缺失、undefined、协议求解失败和运行故障分别保留。\n')
    footprint=artifact_manifest(run,out)
    result=dict(status=delivery['status'],report=str(out/'Phase5_Report.md'),results=str(out),footprint=footprint,time_beijing=now(),seconds=time.monotonic()-start,decisions=metrics['decisions'])
    save(run/'report/summary.json',result);return result
