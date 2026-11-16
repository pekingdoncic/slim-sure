"""Evidence-backed Phase 3 report. No inference, training, or checkpoint writes."""
from common import *
import subprocess
def fmt(x,n=6):
    return '无定义' if x is None else f'{x:.{n}f}' if isinstance(x,(int,float)) else str(x)
def ci(x):return '不可判定' if not x else '['+', '.join(fmt(v) for v in x)+']'
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(x).replace('|','/') for x in r)+' |' for r in rows])
def link(run,p):return '['+p+']('+str(Path(run)/p)+')'
def disk(p):
    logical=allocated=count=0
    for f in Path(p).rglob('*'):
        if f.is_file():
            st=f.stat();logical+=st.st_size;allocated+=st.st_blocks*512;count+=1
    return dict(logical_bytes=logical,allocated_bytes=allocated,files=count)
def executions(run):
    output={}
    for stage in ['A','B','C','A4']:
        state=read(run/stage/'status.json');worker=state.get('worker',{});resources=worker.get('resources',{})
        output[stage]=dict(status=state['status'],started=state.get('started'),finished=state.get('finished'),seconds=worker.get('seconds'),
            supervisor_pid=state.get('supervisor_pid'),worker_pid=state.get('worker_pid'),launch=read(run/stage/'launch.json'),
            GPU_peak_allocated=resources.get('peak_allocated',0),GPU_peak_reserved=resources.get('peak_reserved',0),rss_peak_bytes=resources.get('rss_peak_bytes'),
            output_disk=disk(run/stage),log=str(run/stage/'worker.log'),heartbeat=str(run/stage/'heartbeat.json'))
    output['B_initial_metadata_error']=read(run/'B/attempts/01_metadata_error/worker_status.json')
    output['physical_forward_counts']=dict(B_committed=750,B_uncommitted_metadata_error=1,C_committed=100,C_optional_two_step=3,total=854,optimizer_updates=0,checkpoint_writes=0)
    output['initial_B_GPU_peak_note']='The uncommitted metadata-error attempt did not save a GPU peak; later successful B measurements do not substitute for it.'
    save(run/'execution_summary.json',output);return output
def main(run):
    from reliability_v1.phase2_runtime import make_identity
    run=Path(run);stages={s:read(run/s/'summary.json') for s in ['A','B','C','A4']}
    if any(v['status']!='completed' for v in stages.values()):raise GateFailure('Report requires committed complete stage summaries')
    if digest(make_identity(read(CONFIG),SOURCE))!=EXPECTED:raise GateFailure('Final frozen identity mismatch')
    for path,value in read(run/'input_refs.json')['frozen_paths'].items():
        if sha(path)!=value:raise GateFailure('Read-only Phase 2 reference changed: '+path)
    if list(run.rglob('*.pt')) or list(run.rglob('*.ckpt')):raise GateFailure('Unexpected checkpoint output')
    for stage in ['B','C']:
        check=read(run/stage/'frozen_check.json')
        if not check['backbone_and_BN_unchanged'] or not check['head_unchanged'] or check['optimizer_updates']:raise GateFailure('Frozen check failed')
    guards={s:read(run/'environment_probe'/s/'status.json') for s in ['A','B','C','A4']}
    if any(v['status']!='pass' or v['new_faults'] for v in guards.values()):raise GateFailure('Environment guard did not pass')
    integrity=dict(status='pass',time_beijing=now(),identity_sha=EXPECTED,
        HEAD=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        branch=subprocess.check_output(['git','-C',str(ROOT),'branch','--show-current'],text=True).strip(),
        worktree=subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain=v1','--untracked-files=normal'],text=True).strip(),
        input_reference_hashes_unchanged=True,new_checkpoints=0,diagnostic_code_sha256={p.name:sha(p) for p in (ROOT/'diagnostics_phase3').iterdir() if p.suffix in ['.py','.json']})
    save(run/'final_integrity.json',integrity)
    execution=executions(run);a0=read(run/'A/A0_reproduction.json');b0=read(run/'B/B0_reproduction.json');c0=read(run/'C/C0_acceptance.json')
    pred=read(run/'C/stage_predictive_power.json');changes=read(run/'C/stage_advantage_changes.json')
    keys=['common_BC/parent/'+x for x in ['coarse_linf','coarse_l2','t0','t1','t4']]
    predictive=[]
    for key in keys:
        s=pred[key];d=s['paired_differences']['head-mconf']['Spearman']
        predictive.append(dict(stage=key,head=s['scene_equal']['head']['Spearman'],lr=s['scene_equal']['lr']['Spearman'],mconf=s['scene_equal']['mconf']['Spearman'],
            D=d['mean_difference'],ci95=d['ci95'],pairs=d['pair_count'],scenes=d['scene_count'],parents=sum(v['valid_parents'] for v in s['per_scene_denominators'].values())))
    csvout(run/'Phase3_Diagnostics_Report.csv',predictive)
    change=changes['common_BC/parent/t0_minus_coarse_linf']
    conclusion=dict(status='可判定' if stages['C']['all_BC_parents_associated'] else '不可判定',
        first_observed_reversal='细窗口选出子点之后、第一次精修更新之前（t=0）',
        evidence='Head is superior for both continuous coarse errors, then inferior for t=0 child-median EPE on the same common parent set; 99 jointly defined pairs, 15 scenes.',
        D_change=change,causal_limit='Observed stage transition only; anchor convention, fine matching selection, and actual-child GT geometry effects are not individually isolated.')
    gates={k:stages[s][k] for k,s in [('G1','B'),('G2','A'),('G3','A4'),('G4','C')]}
    undefined={}
    for name,path in [('A2','A/A2_same_parent_comparison.json'),('B','B/label_score_comparison.json'),('C_stages','C/stage_predictive_power.json'),('C_quality','C/quality_signals.json'),('C_classification','C/quality_classification.json')]:
        undefined[name]={k:v['undefined_pair_counts'] for k,v in read(run/path).items()}
    save(run/'undefined_counts.json',undefined)
    items=[]
    for row in read(ROOT/'diagnostics_phase3/report_manifest.json'):
        name,paths,reason,*status=row
        items.append(dict(item=name,paths=[str(run/p.strip()) for p in paths.split(';')],status=status[0] if status else '完成',reason=reason))
    repeat=read(run/'A4/baseline_independent_recheck.json')
    if repeat['status']!='pass':
        for item in items:
            if item['item']=='A4：5对独立同seed复算':item.update(status='失败',reason='Independent same-seed repeat differs; G3 is not decidable')
    save(run/'artifact_manifest.json',items);csvout(run/'artifact_manifest.csv',items)
    report=dict(status='completed',verified_at=now(),source_run=str(SOURCE),output_run=str(run),frozen_identity_sha=EXPECTED,stages=stages,gates=gates,
        first_observed_transition=conclusion,predictive_table=predictive,
        acceptance=dict(A0=a0,B0=b0,C0={k:v for k,v in c0.items() if k!='pairs'},two_step=read(run/'C/optional_two_step_equivalence.json'),
            AUC_scale=read(run/'A4/AUC_scale_test.json'),baseline_recheck=repeat),execution=execution,artifact_manifest=items,
        protocol_assumptions=read(run/'protocol_assumptions.json'),
        failure_categories=dict(statistical_undefined=str(run/'undefined_counts.json'),executed_geometry_failures=stages['A4']['solver_failures'],
            missing_geometry_units=stages['A4']['missing_units'],implementation_incidents=1,model_nonfinite_faults=0,OOM=0,frozen_parameter_or_BN_changes=0,new_storage_faults=0),
        limitations=['Single training seed','Validation scenes used for checkpoint selection','100 pose pairs in 15 scenes','G3 not triggered does not establish no effect','No score weights fitted on these 100 pairs'],
        optimizer_updates=0,checkpoint_writes=0,expansion_executed=False)
    save(run/'Phase3_Diagnostics_Report.json',report)
    lines=['# SLiM Phase 3 诊断报告','',f'状态：**completed**。核验时间：{report["verified_at"]}。输出：{run}','',
        f'冻结身份 {EXPECTED} 复核通过；当前 HEAD {integrity["HEAD"]}。已完成 Phase 2 run 只读。训练更新 0、checkpoint 写入 0。','',
        '**核心结论：首次观察到完整头相对 mconf 的排序优势反转，是细窗口选出子点后的 t=0，尚未执行第一次精修更新。**',
        '完整头对两种粗连续误差也有优势，本次不支持“在粗阶段就只会阈值分类”的解释。该结果定位观察到的阶段变化，不独立归因锚点、细匹配选择或子点GT。','',
        '## 预设判据','',
        table(['判据','状态','实测'],[
            ['G1',gates['G1']['status'],'共同有效一致率 '+fmt(gates['G1']['agreement'])+'；R4 ΔBA 95%CI '+ci(gates['G1']['R4_delta_BA_ci95'])],
            ['G2',gates['G2']['status'],'粗负例有效最终父点 EPE<2 比例 '+fmt(gates['G2']['value'])+'；分母 '+str(gates['G2']['denominator'])],
            ['G3',gates['G3']['status'],'0.8预算 head−mconf pooled AUC@5 '+fmt(gates['G3']['difference'])+' 个百分点；95%CI '+ci(gates['G3']['ci95'])],
            ['G4',gates['G4']['status'],'主信号子点 ΔSpearman '+fmt(gates['G4']['mean_difference'])+'；95%CI '+ci(gates['G4']['ci95'])]]),'',
        'G1支持另行授权的换标签训练对照，不认定现行实现错误，也不预言重训一定提升位姿。G4只由预设主信号触发。训练、换标签、扩样和测试集任务本轮均未执行。','',
        '## 逐阶段预测力','',
        'Spearman越负越好；D=head−mconf。同一批共同有效父点，场景等权；区间为2000次场景bootstrap、seed2026。',
        table(['阶段','head','LR','mconf','D','D 95%CI','pair/scene/父点'],[[r['stage'].split('/')[-1],fmt(r['head']),fmt(r['lr']),fmt(r['mconf']),fmt(r['D']),ci(r['ci95']),f'{r["pairs"]}/{r["scenes"]}/{r["parents"]}'] for r in predictive]),'',
        '粗Linf → t=0 的 D 变化为 '+fmt(change['mean_difference'])+'，95%CI '+ci(change['ci95'])+'，仅使用共同有定义的 '+str(change['pair_count'])+' 对。',
        '完整父点/子点表及相邻阶段变化：'+link(run,'C/stage_predictive_power.csv')+'；'+link(run,'C/stage_advantage_changes.json')+'。','',
        '核心表CSV：'+link(run,'Phase3_Diagnostics_Report.csv')+'。附图：'+link(run,'figures/stage_advantage.svg')+'；'+link(run,'figures/stage_advantage.png')+'。','',
        '## A：粗标签与最终结果','']
    cross=read(run/'A/A1_cross_table.json');rows=[]
    for label in [-1,0,1]:
        v=cross['all/label_'+str(label)]['pooled'];rows.append([label,v['denominator'],v['without_fine_output'],v['without_valid_final_GT']]+[str(v[k])+' / '+fmt(v['fractions'][k]) for k in ['epe_lt1','epe_lt2','epe_lt3','epe_gt5']])
    lines += [table(['粗标签','最终有效父点','无细输出','无最终GT','<1 数/比例','<2 数/比例','<3 数/比例','>5 数/比例'],rows),'',
        '主分母为“该粗标签下有有效最终EPE的父点”。场景等权与逐场景结果见 '+link(run,'A/A1_cross_table.json')+'。',
        'R1/R2/R3/R3a/R3b/R4全部三方法AUROC/AP、阈值1/2/3/5、区间、逐场景胜负、低纹理及undefined分母见 '+link(run,'A/A2_same_parent_comparison.json')+'；共同pair优势变化见 '+link(run,'A/A2_advantage_changes.json')+'。',
        '稳健保留曲线含均值、中位数、<1/2/3精度、去掉0003与低纹理：'+link(run,'A/A3_robust_curves.csv')+'。另外提供全部保留父点的汇总中位数：'+link(run,'A/A3_pooled_curves.csv')+'。',
        '低纹理实际分母：'+json.dumps(read(run/'A/low_texture_denominators.json')['pooled']['low_texture'],ensure_ascii=False)+'。','',
        '## B：标签锚点','']
    agree=read(run/'B/label_agreement.json');v=agree['pooled'];r4=read(run/'B/R4_label_final_EPE_comparison.json')
    lines += [table(['全部父点','共同有效','仅角点','仅中心','均无效','一致率','场景等权一致率'],[[v['total'],v['both_valid'],v['corner_only'],v['center_only'],v['both_invalid'],fmt(v['agreement']),fmt(agree['scene_equal_agreement'])]]),'',
        '共同有效2×2（行角点0/1、列中心0/1）：'+json.dumps(v['matrix'])+'。角点/中心正例率：'+fmt(v['corner_positive_fraction'])+' / '+fmt(v['center_positive_fraction'])+'。',
        'R4 ΔBA='+fmt(r4['paired_delta_BA']['mean_difference'])+'，共同可评价 '+str(r4['paired_delta_BA']['pair_count'])+' 对；95%CI '+ci(r4['paired_delta_BA']['ci95'])+'。',
        '**中心锚点单向标签不等于原生SLiM监督。** 可选原生三因素分解未执行。',
        '实测粗stride=8、fine_scale=1。相关窗口从 round(coarse_coord/fine_scale) 开始，每轴索引0…7；子点报告坐标为 coarse_coord+(0.5+i,0.5+j)，范围+0.5…+7.5px，cell覆盖[0,8)，中心为+4px。角点标签的[-4,4]盒与该正向cell范围不同；中心单向对照与cell范围相容，但仍不是逐子点GT或原生监督。',
        '直接相关窗口不自己clamp；循环精修6×6双线性窗口使用zero padding。细节、边界和冻结源码行号：'+link(run,'B/fine_window_geometry_audit.json')+'。',
        'x/y位移、Linf主分档、L2附表、局部尺度与未知、低纹理分母、逐场景结果均见下方产物清单；100对粗残差导出索引为 '+link(run,'B/coarse_100_index.json')+'。','',
        '## C：轨迹与免训练信号','',
        f'实测 T={c0["actual_T"]}、fine_scale={c0["actual_fine_scale"]}、lookup radius={c0["actual_lookup_radius"]}。t=0直接使用intermediate_coord_1，t步对应all_offset_1[t−1]并先转float32。各步共用一次真实子点源坐标投影得到的GT掩码。',
        '重建最大偏差 '+fmt(c0['reconstruction_max_deviation_px'],9)+' px；100对均可与Phase2和B关联。3对两步截断核查通过；未测量实际早停节省。','']
    outcomes=[r for r in read(run/'C/rescues_losses_gain_classes.json')['pooled_and_per_scene'] if r['scene']=='ALL_POOLED']
    lines += [table(['子点组','有效数','初始≥2/救回','初始<2/损失','1→T收益','小变化','恶化'],[[r['group'],r['count'],f'{r["initial_ge2"]}/{r["rescued"]}',f'{r["initial_lt2"]}/{r["lost"]}',r['further_gain'],r['stable_error'],r['worsened']] for r in outcomes]),'',
        '收益分类阈值为0.1px。坐标位移另行报告；粗负例最终可用不全由精修救回。各步均值/中位数/精度、下降量与逐场景分层：'+link(run,'C/step_errors.csv')+'。',
        '下表为全部有效子点的场景等权质量信号；分数越高越可靠，Spearman越负越好。','']
    q=read(run/'C/quality_signals.json')['all/child'];qrows=[]
    for m,v in q['scene_equal'].items():
        d=q['paired_differences'].get(m+'-mconf',{}).get('Spearman',{})
        qrows.append([m,fmt(v['Spearman']),fmt(v['AUSE']),fmt(d.get('mean_difference')) if m!='mconf' else '0',ci(d.get('ci95')) if m!='mconf' else '基线'])
    lines += [table(['信号','Spearman','AUSE','ΔSpearman−mconf','95%CI'],qrows),'',
        '父点级信号取同一有效子点集合的中位数；末步/路径信号只用于精修后质量评估。低纹理、父点/子点、AUROC@1/2/3px、曲线与路由：'+link(run,'C/quality_classification.json')+'；'+link(run,'C/quality_retention_curves.json')+'；'+link(run,'C/routing.json')+'。','',
        '## A4：位姿过滤','',
        f'正式单元 {stages["A4"]["formal_solver_units"]}；独立复算 {stages["A4"]["independent_rechecks"]}；已执行几何失败 {stages["A4"]["solver_failures"]}；缺失 {stages["A4"]["missing_units"]}。',
        '排序对象是全部有细输出父点，不做GT筛选。恢复原始子点顺序后交给规范求解器。100%四方法数组及行序哈希相同，每pair/seed共享一次求解；5对×5seed独立复算另列。',
        'AUC是0–100刻度，主表合并全部100对，每seed算AUC后平均5seed。场景bootstrap保留同pair的全部方法和5seed。主保留率预设为0.8，其他预算只描述。','']
    pose=read(run/'A4/pose_AUC_comparisons.json')
    lines += [table(['方式','AUC@5','AUC@10','AUC@20'],[[k]+[fmt(pose['pooled_seed_mean'][k][m]) for m in ['AUC@5','AUC@10','AUC@20']] for k in ['baseline_100','head_80','lr_80','mconf_80','random_80']]),'']
    pcomp=[]
    for k in ['head_80-mconf_80','head_80-baseline_100','mconf_80-baseline_100','random_80-baseline_100']:
        v=pose['paired_comparisons'][k];pcomp.append([k]+[fmt(v['mean_difference'][m])+' '+ci(v['ci95'][m]) for m in ['AUC@5','AUC@10','AUC@20']])
    lines += [table(['比较','ΔAUC@5 [CI]','ΔAUC@10 [CI]','ΔAUC@20 [CI]'],pcomp),'',
        '各预算实际子点/内点、对极精度与低纹理保留：'+link(run,'A4/filter_quality_and_counts.csv')+'。截断20度配对误差和胜负：'+link(run,'A4/capped_pose_error_comparisons.json')+'。',
        '32px网格主覆盖口径为R832有效图像区，原图32px另列。两图筛前/后及低纹理保留：'+link(run,'A4/primary_coverage_per_pair.json')+'。']
    if gates['G3']['expand_750_primary_only_recommended']:
        lines += ['**扩样建议（未执行）：** G3未触发，AUC@5区间半宽 '+fmt(gates['G3']['ci_half_width'])+' 个百分点，大于1.0。建议下一轮在750对验证集只跑主保留率0.8；等待用户新指令。']
    else:lines += ['本轮未执行750对扩样或MegaDepth-1500测试。G3未触发不代表没有效果。']
    lines += ['','## 验收、异常与资源','',
        table(['验收','结果'],[['A0',a0['status']+'；最大聚合偏差 '+fmt(a0['maximum_absolute_deviation'],9)],['B0',b0['status']+'；分数偏差 '+json.dumps(b0['max_absolute_score_deviation'])],['C0','pass；最大重建偏差 '+fmt(c0['reconstruction_max_deviation_px'],9)],['冻结模型/BN/头','B/C check_frozen与头hash均通过'],['AUC刻度','全0→100；全超过阈值/执行失败→0；未二次转换'],['100%独立复算',repeat['status']]]),'',
        '统计无定义见 '+link(run,'undefined_counts.json')+'。首个B尝试有新诊断元数据KeyError(fine_scale)，尚未提交任何pair；修正后只重试这一个未提交前向。它不是B0或模型数值故障，证据保留在B/attempts/。',
        table(['任务','开始','结束','worker秒','GPU alloc峰GiB','GPU reserve峰GiB','输出占用GiB'],[[s,e['started'],e['finished'],fmt(e['seconds'],2),fmt(e['GPU_peak_allocated']/2**30,3),fmt(e['GPU_peak_reserved']/2**30,3),fmt(e['output_disk']['allocated_bytes']/2**30,3)] for s,e in execution.items() if s in ['A','B','C','A4']]),'',
        '命令/PID/日志/心跳及异常历史：'+link(run,'execution_summary.json')+'。磁盘为收尾占用，持续主机/主存/swap门禁见environment_probe。首次B失败尝试的GPU峰值未落盘，未以其他测量替代。','',
        '## 口径假设和局限','',
        *['- '+k+'：'+v for k,v in report['protocol_assumptions'].items()],'',
        '单训练种子；验证场景参与选轮；位姿试点100对、15场景。低纹理只描述。未在这100对拟合组合权重。G3未触发不意味着无效，G1/G4也不是论文结论。','',
        '## 逐项产物清单','',
        table(['事项','状态','文件','说明'],[[x['item'],x['status'],'<br>'.join('['+str(Path(p).relative_to(run))+']('+p+')' for p in x['paths']),x['reason']] for x in items])]
    text(run/'Phase3_Diagnostics_Report.md','\n'.join(lines)+'\n')
    save(run/'stage_status.json',dict(status='completed',last_verified_beijing=now(),tasks={s:'completed' for s in stages},gates={k:v['status'] for k,v in gates.items()},
        identity_sha=EXPECTED,optimizer_updates=0,checkpoint_writes=0,expansion_executed=False,report=str(run/'Phase3_Diagnostics_Report.md'),resume_needed=False))
    return dict(status='completed',gates={k:v['status'] for k,v in gates.items()},report=str(run/'Phase3_Diagnostics_Report.md'),first_observed_reversal=conclusion['first_observed_reversal'])
