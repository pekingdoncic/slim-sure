from common4 import *
from geometry4 import *
METRICS=['epipolar_precision','precision_lt1','precision_lt2','GT_evaluable_rate','low_texture_retention','coverage0','coverage1']
def quality_summary(run,plan,units=None):
    prep=read(Path(run)/'P3/prediction_preparation.json')['rows'];rows=[];tie_rows=[]
    for r in prep:
        rows.append(dict(**base(r),methods={k:{m:v.get(m) for m in METRICS} for k,v in r['metrics'].items()}))
        for combo,v in r['cutoff_ties'].items():tie_rows.append(dict(**base(r),combo=combo,**v))
    comps=[]
    for suffix in ['80','50']:
        comps += [(a+'_'+suffix,b+'_'+suffix) for a,b in [('S_m','mconf'),('S_m','fine'),('S_h','S_m')]]+[(m+'_'+suffix,'random_'+suffix) for m in FILTERS if m!='random']
    summary=metric_rows_summary(rows,METRICS,plan['combos'],[(a,b) for a,b in comps if a in plan['combos'] and b in plan['combos']])
    table=[]
    for combo in plan['combos']:
        ms=[r['metrics'][combo] for r in prep if combo in r['metrics']]
        count=sum(x['children'] for x in ms);gt=sum(x['valid_children'] for x in ms)
        row=dict(combo=combo,pairs=len(ms),children=count,valid_children=gt,**summary['scene_equal'][combo],
            pooled_epipolar_precision=ratio(sum(x['epipolar_correct'] for x in ms),count),pooled_GT_evaluable_rate=ratio(gt,count),
            pooled_precision_lt1=ratio(sum(x['epe_lt1'] for x in ms),gt),pooled_precision_lt2=ratio(sum(x['epe_lt2'] for x in ms),gt),
            low_texture_parents_before=sum(x['low_texture_before'] for x in ms),low_texture_parents_retained=sum(x['low_texture_retained'] for x in ms),
            pooled_low_texture_retention=ratio(sum(x['low_texture_retained'] for x in ms),sum(x['low_texture_before'] for x in ms)))
        if units is not None:
            u=[x for x in units if x['combo']==combo];row.update(solver_units=len(u),inliers_total_over_5_seeds=sum(x['inliers'] for x in u),
                inliers_mean_per_pair_seed=float(np.mean([x['inliers'] for x in u])) if u else None)
        table.append(row)
    obj=dict(status='completed' if units is not None else 'metrics_complete_geometry_pending',summary=summary,table=table,
        primary_retention=.8,main_metric_aggregation='Pair ratios -> scene pair mean -> equal mean across 15 scenes. Pooled count-weighted ratios separately supplied.',
        denominator_contract='Retention selected on ALL children before GT; EPE ratios only selected GT-valid children; epipolar/GT rate all selected; low-texture parent denominator original output parents; coverage R832 valid area.',
        rounding='ceil(rate*N_children) independently per pair',ties='Concrete selection only: descending score then raw parent ID then historical native child ordinal; restore native child order before solver')
    artifact(run,'P3_filter_quality',obj,table)
    artifact(run,'P3_paired_differences',summary['paired_differences'],flat_diffs(summary,'filter'))
    artifact(run,'P3_cutoff_ties',dict(rows=tie_rows,pooled_by_combo={combo:{k:sum(x.get(k,0) for x in tie_rows if x['combo']==combo) for k in ['cutoff_equal_score_count','selected_at_cutoff','excluded_at_cutoff','tie_broken_selected','tie_broken_excluded','tie_straddles_cutoff']} for combo in plan['combos']}),tie_rows)
    save(Path(run)/'P3/pair_quality_rows.json',rows)
    return obj
def g6(run,q):
    main=q['summary']['paired_differences'].get('S_h_80-S_m_80',{}).get('epipolar_precision')
    support=read(results(run)/'P2_ranking.json')['summaries']['all/child']['paired_differences'].get('S_h-S_m',{}).get('Spearman')
    if not main or not support or not main['ci95'] or not support['ci95']:return dict(status='不可判定',reason='Required full-population filter or valid-GT ranking comparison unavailable')
    lo,hi=main['ci95'];bad=support['ci95'][0]>0
    if lo>0 and not bad:state='S_h 更优';default='S_h';message='进入部署候选；本轮只判断公式，不决定完整头去留'
    elif lo>0 and bad:state='指标冲突';default='S_m';message='对极精度改善但 Spearman 显著不利，本轮默认 S_m'
    elif hi<0:state='S_h 更差';default='S_m';message='放弃本轮 head_p × fine 公式，冻结头和权重仍保留'
    else:state='优势不明确';default='S_m';message='默认额外开销更低的 S_m，冻结完整头保留作对照'
    return dict(status=state,default_factor=default,epipolar=main,support_Spearman=support,
        low_texture=q['summary']['paired_differences']['S_h_80-S_m_80']['low_texture_retention'],
        coverage0=q['summary']['paired_differences']['S_h_80-S_m_80']['coverage0'],coverage1=q['summary']['paired_differences']['S_h_80-S_m_80']['coverage1'],message=message)
def main(run):
    plan=prepare(run,'P3');q=quality_summary(run,plan);gate=g6(run,q);artifact(run,'G6',gate)
    execute(run,'P3',plan)
    comps=[]
    for suffix in ['80','50']:
        comps += [(m+'_'+suffix,'baseline_100') for m in FILTERS]+[('S_m_'+suffix,'mconf_'+suffix),('S_h_'+suffix,'S_m_'+suffix)]
    pose,units=summarize_pose(run,'P3',plan,comps);q=quality_summary(run,plan,units)
    coverage=read(Path(run)/'P3/geometry_coverage.json')
    result=dict(status='completed',G6=gate,geometry=coverage,primary_metrics=[x for x in q['table'] if x['combo'].endswith('_80')])
    save(Path(run)/'P3/summary.json',result);return result
