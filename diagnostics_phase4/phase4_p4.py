from common4 import *
from geometry4 import *
METRICS=['mean_EPE','median_EPE','precision_lt1','precision_lt2','precision_lt3','epipolar_precision']
def gate7(quality,pose):
    output=[];active=True;first_failed=None
    tolerances={'AUC@5':-.5,'precision_lt1':-.01,'epipolar_precision':-.005}
    for t in [3,2,1,0]:
        conditions={};allpass=True;anydecline=False
        for metric,tolerance in tolerances.items():
            if metric=='AUC@5':
                comp=pose['paired_comparisons'].get('t'+str(t)+'-baseline_100',{});ci=comp.get('ci95',{}).get(metric);value=comp.get('mean_difference',{}).get(metric)
            else:
                comp=quality['summary']['paired_differences'].get('t'+str(t)+'-t4',{}).get(metric,{});ci=comp.get('ci95');value=comp.get('mean_difference')
            if ci is None:state='不可判定';allpass=False
            elif ci[0]>=tolerance:state='通过'
            elif ci[1]<tolerance:state='下降超过预设容差';allpass=False;anydecline=True
            else:state='非劣效证据不足';allpass=False
            conditions[metric]=dict(difference=value,ci95=ci,tolerance=tolerance,assessment=state,confirmatory=active)
        if not active:message='仅描述：固定序列已在更长步数停止，未进行本档确认性判定'
        elif allpass:message='离线轨迹重放中，该步数满足预设精度容差，值得进行真实截断测速'
        elif conditions['precision_lt1']['assessment']=='通过' and conditions['epipolar_precision']['assessment']=='通过' and conditions['AUC@5']['assessment']=='非劣效证据不足':message='精度满足容差，位姿不可确认'
        elif anydecline:message='未通过：'+','.join(k for k,v in conditions.items() if v['assessment']=='下降超过预设容差')+' 下降超过预设容差'
        else:message='未通过：非劣效证据不足或依赖缺失'
        output.append(dict(step=t,confirmatory=active,all_three_pass=allpass,conditions=conditions,message=message))
        if active and not allpass:first_failed=t;active=False
    return dict(status='completed',sequence='3 -> 2 -> 1 -> 0, stop confirmation at first non-pass',first_failed_step=first_failed,steps=output,
                limitation='Offline replay only; no claim about actual truncated GPU latency or memory.')
def quality_summary(run):
    prep=read(Path(run)/'P4/prediction_preparation.json')['rows'];rows=[]
    for r in prep:
        methods={k:{m:v.get(m) for m in METRICS} for k,v in r['metrics'].items()}
        methods['t4']={m:r['baseline_metrics'].get(m) for m in METRICS}
        rows.append(dict(**base(r),methods=methods))
    s=metric_rows_summary(rows,METRICS,['t0','t1','t2','t3','t4'],[('t'+str(t),'t4') for t in range(4)])
    for comp,vals in s['paired_differences'].items():
        for k,x in vals.items():
            lower=k in ['mean_EPE','median_EPE'];v=np.asarray(list(x['per_scene_difference'].values()))*(-1 if lower else 1)
            x.update(win=int((v>1e-12).sum()),tie=int((np.abs(v)<=1e-12).sum()),loss=int((v< -1e-12).sum()),higher_is_better=not lower)
    table=[]
    for t in range(5):
        name='t'+str(t);vs=[r['baseline_metrics'] if t==4 else r['metrics'][name] for r in prep]
        n=sum(x['children'] for x in vs);v=sum(x['valid_children'] for x in vs)
        table.append(dict(step=t,**s['scene_equal'][name],children=n,valid_children=v,pooled_precision_lt1=ratio(sum(x['epe_lt1'] for x in vs),v),
            pooled_precision_lt2=ratio(sum(x['epe_lt2'] for x in vs),v),pooled_precision_lt3=ratio(sum(x['epe_lt3'] for x in vs),v),
            pooled_epipolar_precision=ratio(sum(x['epipolar_correct'] for x in vs),n)))
    obj=dict(status='completed',summary=s,table=table,
        error_contract='t0..t3 reconstructed in float32 from cached initial+pixel updates; GT from committed actual child sources. t4 exact historical final coordinates and native final EPE.',
        aggregation='Pair metrics -> within-scene means -> scene equal; pooled proportions separately supplied.')
    artifact(run,'P4_step_quality',obj,table);artifact(run,'P4_step_differences',s['paired_differences'],flat_diffs(s,'step'))
    save(Path(run)/'P4/pair_quality_rows.json',rows);return obj
def main(run):
    if not read(Path(run)/'P0/summary.json')['capabilities']['P4']:
        obj=dict(status='不可判定',reason='All-child trajectory or historical baseline verification unavailable');artifact(run,'G7',obj);save(Path(run)/'P4/summary.json',obj);return obj
    plan=prepare(run,'P4');q=quality_summary(run);execute(run,'P4',plan)
    pose,units=summarize_pose(run,'P4',plan,[('t'+str(t),'baseline_100') for t in range(4)])
    g=gate7(q,pose);artifact(run,'G7',g)
    # Existing C resources are whole-forward samples, not staged inference timings.
    timing=dict(status='跳过',reason='Committed caches contain task/whole-forward timing, but no verified per-refinement-stage latency. No new timed forward is allowed.')
    artifact(run,'P4_optional_timing',timing)
    result=dict(status='completed',G7=g,geometry=read(Path(run)/'P4/geometry_coverage.json'),optional_timing=timing)
    save(Path(run)/'P4/summary.json',result);return result
