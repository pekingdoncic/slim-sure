"""CPU geometry, verified baseline reuse and T report."""
from shared import *
import argparse,signal
def old_json(rel):
    index=read(P3/'delivery_artifact_sha256.json');p=P3/rel
    if rel not in index or sha(p)!=index[rel]['sha256']:raise IdentityFailure('Historical checksum: '+rel)
    return read(p)
def pose(run,g):
    sys.path.insert(0,str(ROOT/'diagnostics_phase3'))
    from pose_statistics import verified_unit,mean_auc
    from reliability_v1.geometry_worker import prediction_identity
    protocol=geometry_protocol()
    if protocol!=old_json('A4/protocol.json'):raise IdentityFailure('Canonical protocol differs')
    oldplan=old_json('A4/geometry_plan.json');prep={x['pair_id']:x for x in old_json('A4/prediction_preparation.json')['rows']}
    out=run/'pose';out.mkdir(exist_ok=True);save(out/'protocol.json',protocol)
    recs=records('epe');baseline=[];units={};combos=[];skips=read(run/'degradation.json')['skipped'] if (run/'degradation.json').exists() else []
    for s in ['S1','S2','S3','S4','S5']:
        if s in ['S3','S4'] and 'T S3/S4 pose' in skips:continue
        if all(receipt(run,s,r) for r in recs):combos.append(s)
    for r in recs:
        a=get_unit(run,'S0',r,True)
        if a is None:return dict(status='不可判定',reason='S0 incomplete')
        pred=dict(points0_orig=a['source']*a['scale0'],points1_orig=a['target']*a['scale1'],K0=a['K0'],K1=a['K1'],T_0to1=a['T'])
        if prediction_identity(pred,protocol)!=prep[r['pair_id']]['predictions']['baseline_100']['identity']:raise IdentityFailure('S0 geometry coordinate/order hash differs')
        for seed in range(5):
            x=verified_unit(oldplan,pairname(r['pair_id'])+'/baseline_100/'+str(seed))
            if x is None:raise IdentityFailure('Missing baseline geometry')
            baseline.append(dict(**base(r),seed=seed,error=x['pose_error_deg'],status=x['solver_status'],inliers=x['num_inliers']))
        for s in combos:
            a=get_unit(run,s,r,True);p=out/'predictions'/s/(pairname(r['pair_id'])+'.npz')
            values=dict(points0_orig=a['source']*a['scale0'],points1_orig=a['target']*a['scale1'],K0=a['K0'],K1=a['K1'],T_0to1=a['T'],pair_id=np.array(r['pair_id']))
            if not p.exists():npzout(p,values)
            elif prediction_identity(npzload(p),protocol)!=prediction_identity(values,protocol):raise IdentityFailure('Prepared prediction changed')
            for seed in range(5):
                key=pairname(r['pair_id'])+'/'+s+'/'+str(seed)
                units[key]=dict(**base(r),combo=s,method=s,seed=seed,prediction=str(p),prediction_sha256=sha(p),directory=pairname(r['pair_id'])+'/'+s,independent_repeat=False)
    plan=dict(output=str(out/'geometry'),protocol=str(out/'protocol.json'),protocol_sha256=digest(protocol),units=units,formal_unit_order=list(units),formal_units=len(units),combos=combos)
    if (out/'plan.json').exists() and read(out/'plan.json')!=plan:raise IdentityFailure('Pose plan changed')
    save(out/'plan.json',plan);save(out/'baseline_reuse.json',dict(status='pass',reused=500,new_solves=0,rows=baseline))
    jobs=[]
    try:
        for idx in range(3):
            keys=list(units)[idx::3];f=out/f'jobs_{idx}.json'
            if f.exists() and read(f)!=keys:raise IdentityFailure('Pose partition changed')
            save(f,keys);cmd=['/home/bupt/miniconda3/envs/hybrid-eval/bin/python','-B','-u',str(ROOT/'diagnostics_phase3/pose_units.py'),'--plan',str(out/'plan.json'),'--jobs',str(f),'--worker-name',f'cpu_{idx}']
            log=(out/f'worker_{idx}.log').open('ab');c=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1'),start_new_session=True)
            save(out/f'launch_{idx}.json',dict(command=cmd,pid=c.pid,started=now()));jobs.append((c,log))
        while any(c.poll() is None for c,f in jobs):
            g.check();beat(run,'pose',phase='canonical geometry',workers=[dict(pid=c.pid,returncode=c.poll()) for c,f in jobs])
            if any(c.poll() not in [None,0] for c,f in jobs):raise RuntimeError('Geometry worker failure')
            time.sleep(5)
        if any(c.returncode for c,f in jobs):raise RuntimeError('Geometry worker failed')
    finally:
        for c,f in jobs:
            if c.poll() is None:c.terminate()
            try:c.wait(10)
            except subprocess.TimeoutExpired:c.kill();c.wait()
            f.close()
    errors={s:np.full((100,5),np.nan) for s in ['S0']+combos};lookup={r['pair_id']:i for i,r in enumerate(recs)}
    for x in baseline:errors['S0'][lookup[x['pair_id']],x['seed']]=x['error'] if x['status']=='success' else np.inf
    rows=[]
    for key,u in units.items():
        x=verified_unit(plan,key)
        if x is None:raise RuntimeError('Missing geometry receipt')
        errors[u['combo']][lookup[u['pair_id']],u['seed']]=x['pose_error_deg'] if x['solver_status']=='success' else np.inf
        rows.append(dict(**base(u),setting=u['combo'],seed=u['seed'],error=x['pose_error_deg'],status=x['solver_status'],inliers=x['num_inliers'],matches=x['num_points'],seconds=x['geometry_seconds']))
    auc={s:mean_auc(e).tolist() for s,e in errors.items()};scenes=sorted({r['scene'] for r in recs});groups=[np.array([i for i,r in enumerate(recs) if r['scene']==s]) for s in scenes]
    rng=np.random.default_rng(2026);boot={s:[] for s in combos}
    for _ in range(2000):
        ix=np.concatenate([groups[j] for j in rng.integers(len(scenes),size=len(scenes))]);b=mean_auc(errors['S0'][ix])
        for s in combos:boot[s].append(mean_auc(errors[s][ix])-b)
    differences={s:dict(mean_difference=(np.array(auc[s])-auc['S0']).tolist(),ci95=np.quantile(np.asarray(v),[.025,.975],axis=0).T.tolist(),pairs=100,scenes=15) for s,v in boot.items()}
    result=dict(status='completed',new_units=len(rows),reused_baseline_units=500,executed_failures=sum(x['status']!='success' for x in rows),missing=0,AUC_5_10_20=auc,paired_differences=differences,
        per_scene={s:{k:mean_auc(e[ix]).tolist() for k,e in errors.items()} for s,ix in zip(scenes,groups)},protocol=protocol)
    save(out/'summary.json',result);csvout(out/'units.csv',rows);return result
def analyze(run):
    recs=records('epe');output=Path(read(run/'context.json')['results']);outcomes={};main=[];funnel=[];quality=[];aucrows=[];budgetrows=[];normrows=[];comparisons={};by={}
    for s in ['S0','S1','S2','S3','S4','S5']:
        rows=[get_unit(run,s,r) for r in recs if receipt(run,s,r)];by[s]=rows
        outcomes[s]=dict(completed=len(rows),planned=100,status=read(run/s/'status.json') if (run/s/'status.json').exists() else {'status':'not_executed'})
        for r in rows:
            for region,v in r['funnel'].items():
                for stage,x in v.items():
                    if isinstance(x,dict) and 'numerator' in x:funnel.append(dict(**base(r),setting=s,region=region,stage=stage,**x))
            for region,v in r['quality'].items():quality.append(dict(**base(r),setting=s,region=region,**v))
            for region,v in r['added_AUROC'].items():aucrows.append(dict(**base(r),setting=s,region=region,**v))
            budgetrows.append(dict(**base(r),setting=s,**r['budget'],**r['counts']))
        if rows:
            metrics=['source_pass','target_pass','both_pass','correct_parent','correct_parent_fine','correct_parent_good','any_parent','any_good']
            sums={}
            for reg in ['low_texture','other_texture']:
                rr=[dict(**base(r),methods={s:{k:r['funnel'][reg][k]['ratio'] for k in metrics}}) for r in rows]
                sums[reg]=scene_summary(rr,metrics,[s])
            pooled={k:sum(r['counts'][k] for r in rows) for k in ['parents','children','valid_children','low_parents','low_children']}
            times=[dict(**base(r),methods={s:{'seconds':r['timing']['median_seconds']}}) for r in rows]
            latency=scene_summary(times,['seconds'],[s])['scene_equal'][s]['seconds']
            outcomes[s].update(pooled=pooled,recall=sums,scene_equal_seconds=latency,
                peak_allocated=max(v['allocated_peak'] for r in rows for v in r['timing']['repeats']),peak_reserved=max(v['reserved_peak'] for r in rows for v in r['timing']['repeats']))
            main.append(dict(setting=s,complete=len(rows),**pooled,low_texture_recall=sums['low_texture']['scene_equal'][s]['any_good'],other_texture_recall=sums['other_texture']['scene_equal'][s]['any_good'],seconds=latency,low_grid_denominator=sum(r['funnel']['low_texture']['denominator'] for r in rows)))
        for r in recs:
            a=get_unit(run,s,r,True)
            if a is None:continue
            for side in [0,1]:
                ids=a[f'candidate{side}'];budgetrows.append(dict(**base(r),setting=s,side=side,candidates=len(ids),low_candidates=int(a[f'grid_low{side}'][ids].sum()),low_fraction=ratio(int(a[f'grid_low{side}'][ids].sum()),len(ids))))
                for reg,mask in [('low_texture',a[f'grid_legal{side}']&a[f'grid_low{side}']),('other_texture',a[f'grid_legal{side}']&~a[f'grid_low{side}'])]:
                    normrows.append(dict(**base(r),setting=s,side=side,region=reg,legal=int(mask.sum()),passed=int(mask[ids].sum()),pass_rate=ratio(int(mask[ids].sum()),int(mask.sum())),norm=distributions(a[f'grid_norm{side}'][mask])))
    baseline={r['pair_id']:r for r in by['S0']}
    for s in ['S1','S2','S3','S4','S5']:
        paired=[]
        for r in by[s]:
            b=baseline.get(r['pair_id'])
            if b is None:continue
            values={name:{'low_recall':x['funnel']['low_texture']['any_good']['ratio'],'other_recall':x['funnel']['other_texture']['any_good']['ratio'],'both_pass':x['funnel']['low_texture']['both_pass']['ratio'],'latency':x['timing']['median_seconds'],'latency_ratio':x['timing']['median_seconds']/b['timing']['median_seconds']} for name,x in [('S0',b),(s,r)]}
            paired.append(dict(**base(r),methods=values))
        if paired:comparisons[s]=scene_summary(paired,['low_recall','other_recall','both_pass','latency','latency_ratio'],[s,'S0'],[(s,'S0')])
    decisions={x:dict(status='不可判定',reason='required complete setting unavailable') for x in ['GT1','GT2','GT3']}
    if outcomes['S0']['completed']==100 and outcomes['S2']['completed']==100:
        v=comparisons['S2']['paired_differences']['S2-S0']['low_recall'];lo,hi=v['ci95']
        decisions['GT1']=dict(status='completed',value=v,conclusion='在这100对上，从0.65放宽到0.45明显提高可用低纹理召回' if lo>.05 else '该放宽幅度的召回增量小于2个百分点'+('，且召回下降' if hi<0 else '') if hi<.02 else '增量有限或证据不足')
        q=[r['quality']['added_low_texture'] for r in by['S2']];den=sum(x['parent']['n'] for x in q);good=sum(x['parent']['n_lt2'] for x in q);total=sum(x['parents'] for x in q);p=ratio(good,den)
        decisions['GT2']=dict(status='completed' if den else '不可判定',good=good,evaluable=den,added_total=total,proportion=p,GT_evaluable_rate=ratio(den,total),conclusion='可评价新增匹配多数正确' if den and p>=.6 else '多数未达标' if den and p<=.4 else '好坏参半' if den else '无有效分母')
    if outcomes['S0']['completed']==100 and outcomes['S5']['completed']==100:
        eq=all(r['budget']['delta0']==0 and r['budget']['delta1']==0 and r['budget']['delta_product']==0 for r in by['S5'])
        v=comparisons['S5']['paired_differences']['S5-S0']['low_recall'];t=comparisons['S5']['scene_equal']['S5']['latency_ratio']
        decisions['GT3']=dict(status='completed' if eq else '不可判定',budget_equal=eq,recall=v,time_ratio=t,passed=bool(eq and v['ci95'][0]>0 and t<=1.1),unmet=[x for x,ok in [('candidate_budget_equal',eq),('recall_lower_gt0',v['ci95'][0]>0),('time_ratio_le1.10',t<=1.1)] if not ok])
    extra={}
    for s,rs in by.items():
        for region in ['all','low_texture','added','added_low_texture']:
            rows=[]
            for r in rs:
                q=r['quality'][region];m={f'parent_{k}':q['parent'][k] for k in ['lt1','lt2','gt5','mean_EPE','median_EPE']}
                m.update({f'child_{k}':q['child'][k] for k in ['lt1','lt2','mean_EPE','median_EPE']});m.update(center_positive=q['center_positive_rate'],epipolar=q['epipolar_precision'],GT_evaluable=q['GT_evaluable_parent_rate'])
                rows.append(dict(**base(r),methods={s:m}))
            if rows:extra[s+'/'+region]=scene_summary(rows,list(rows[0]['methods'][s]),[s])
    result=dict(status='completed' if all(outcomes[s]['completed']==100 for s in ['S0','S2','S5']) else 'partial',settings=outcomes,comparisons=comparisons,decisions=decisions,quality_scene_equal=extra,aggregation='pair -> scene -> equal scene; paired common units',degradation=read(run/'degradation.json') if (run/'degradation.json').exists() else None)
    for name,val in [('summary',result),('decisions',decisions),('outcomes',outcomes),('quality_scene_equal',extra)]:save(output/(name+'.json'),val)
    for name,rows in [('main_table',main),('grid_funnel',funnel),('quality',quality),('new_parent_AUROC',aucrows),('candidate_budget',budgetrows),('norm_pass',normrows)]:csvout(output/(name+'.csv'),rows)
    for name,key in [('filter80','filter80'),('source_grid_changes','changes'),('score_quantiles','score_quantiles')]:save(output/(name+'.json'),{s:[dict(**base(r),**r[key]) for r in rs] for s,rs in by.items()})
    os.environ['MPLCONFIGDIR']=str(run/'matplotlib');import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4));names=[x['setting'] for x in main]
    axes[0].bar(names,[x['low_texture_recall'] for x in main]);axes[0].set_ylabel('Usable low-texture source-grid recall')
    axes[1].bar(names,[1000*x['seconds'] for x in main]);axes[1].set_ylabel('Scene-equal forward latency (ms)');fig.tight_layout();fig.savefig(output/'recall_latency.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for s in ['S1','S2','S3','S4','S5']:
        vals=[]
        for r in recs:
            a=get_unit(run,s,r,True)
            if a is not None:vals.extend(a['parent_epe'][a['added']&a['low']&np.isfinite(a['parent_epe'])].tolist())
        if vals:ax.hist(np.clip(vals,0,20),bins=np.linspace(0,20,41),histtype='step',density=True,label=s)
    ax.set_xlabel('New low-texture parent median EPE (px; clipped20)');ax.legend();fig.tight_layout();fig.savefig(output/'new_quality_distribution.png',dpi=160);plt.close(fig)
    return result,main
def report(run,result,main,geometry):
    out=Path(read(run/'context.json')['results']);save(out/'pose.json',geometry)
    lines=['# Phase T 范数阈值与低纹理候选扫描','',f"Run: {run.name}；冻结身份 {EXPECTED}。",f"状态：{result['status']}。完整统计 [summary.json](summary.json)，判据 [decisions.json](decisions.json)。",'','|设置|完成/100|父点|有效GT子点|低纹理主召回|前向秒|','|---|---:|---:|---:|---:|---:|']
    for x in main:lines.append(f"|{x['setting']}|{x['complete']}|{x['parents']}|{x['valid_children']}|{x['low_texture_recall']:.6f}|{x['seconds']:.6f}|")
    lines+=['','## 预设判据']
    for k,x in result['decisions'].items():lines.append(k+': '+json.dumps(clean(x),ensure_ascii=False))
    lines+=['','## 分母和质量','源格中心+4，GT可见且映射到有效目标粗格才进入固定分母；目标格用半开区间。每源格每阶段一票。主召回允许任意父目标格产生EPE<2px有效子点。误差用子点自身真实源坐标。',
        '[逐阶段去向](grid_funnel.csv)、[质量](quality.csv)、[预算](candidate_budget.csv)、[范数通过率](norm_pass.csv)、[净增](source_grid_changes.json)、[新增AUROC](new_parent_AUROC.csv)、[80%筛选](filter80.json)。',
        '父级fine取全部实际子点分数中位数，S_m=mconf×fine；先筛后查GT。无细输出、无GT、单类undefined分别保留。源低纹理跟随冻结源粗格，两图候选纹理分别计算。',
        '','## 计时与几何','每对完整预热1次，同步计时3次取中位数；GT、统计及保存不进计时。粗上限6000；细预算按父点分配top-k后再应用全局相对阈值。触顶和截断计数见预算表。段时间来自原生CUDA timer。',
        f"位姿状态：{geometry.get('status')}；[pose.json](pose.json)。原生500单元经坐标、行序、协议和提交哈希核验后复用；求解失败计无限误差，未执行不进分母。",
        '![Recall and latency](recall_latency.png)','![New quality](new_quality_distribution.png)','',
        '## 范围与交接','仅适用于该R832室外、冻结纹理和可见性、反复分析的100对15场景验证样本，不外推测试集或联合训练。T没有checkpoint、新头或optimizer。各worker退出前恢复入口并check_frozen。Phase5新进程重载原权重，独立E0。',
        f"完整命令、PID、耗时、资源、故障、回执和大明细：{run}",
        f"续扫：/home/bupt/miniconda3/envs/slim/bin/python -B diagnostics_threshold/run_threshold.py --run {run} --background --resume",
        f"续几何/报告：/home/bupt/miniconda3/envs/slim/bin/python -B diagnostics_threshold/finish_T.py --run {run}",
        '已有提交先验证再复用。降级记录：'+json.dumps(result.get('degradation'),ensure_ascii=False)]
    text(out/'PhaseT_Report.md','\n\n'.join(lines)+'\n')
    text(out/'README.md','# Phase T\n\n入口：[PhaseT_Report.md](PhaseT_Report.md)。预登记：[preregistered_protocol.json](preregistered_protocol.json)。主表：[main_table.csv](main_table.csv)。大明细路径与SHA256：[artifact_manifest.json](artifact_manifest.json)。\n')
    manifest=[]
    for folder in [out,run]:
        for p in sorted(folder.rglob('*')):
            if p.is_file() and p.name not in ['artifact_manifest.json','artifact_manifest.csv'] and 'cache' not in p.parts:manifest.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),location='result' if p.is_relative_to(out) else 'run'))
    save(out/'artifact_manifest.json',manifest);assert sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<=50*2**20
    save(run/'handoff.json',dict(time_beijing=now(),T_scan_workers_exited=True,model_overrides_process_local=True,phase5_requires_new_process_identity_and_E0=True,report=str(out/'PhaseT_Report.md'),manifest_sha=sha(out/'artifact_manifest.json')))
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=Path(a.run);identity_gate()
    from run_threshold import guard
    from reliability_v1.runtime import run_lock
    with run_lock(run):
        g=guard(run,'T_report');save(run/'report_launch.json',dict(pid=os.getpid(),command=sys.argv,started=now()));result,rows=analyze(run);g.check()
        geometry=pose(run,g) if result['settings']['S0']['completed']==100 else dict(status='不可判定',reason='S0 incomplete')
        report(run,result,rows,geometry);g.check();save(run/'stage_status.json',dict(status=result['status'],report='completed',time_beijing=now(),pid=os.getpid()))
if __name__=='__main__':main()
