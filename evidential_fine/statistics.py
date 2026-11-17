"""CPU-only fixed analyses; frozen ranking/bootstrap and Phase4 random mixture median."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from .common import *
from reliability_v1.comparison import spearman,sparsification
from diagnostics_phase3.common import curve_stats,curve_summary
sys.path.insert(0,str(ROOT/'diagnostics_phase4'))
from phase4_p2b import weighted_median,actual_stats,common_stats
from common4 import metric_rows_summary
sys.path.insert(0,str(ROOT/'diagnostics_phase3'))
from pose_prepare import occupancy
def support(value,metric,fallback=0):
    """Additive pooled counts are supplemental; main estimate remains scene-equal."""
    n=value.get('n',fallback);num=None;den=n
    if metric.startswith('lt') and metric[2:].isdigit():
        num=value.get('n_'+metric,value.get('epe_'+metric));den=value.get('valid',n)
    elif metric=='gt5':num=value.get('n_gt5')
    elif metric=='mean_EPE':num=value.get('sum_EPE')
    elif metric=='epipolar_precision':
        num=value.get('n_epi_correct',value.get('epipolar_correct'));den=value.get('n_epi',n)
    elif metric=='GT_evaluable_rate':
        num=value.get('expected_valid',value.get('valid'));den=value.get('windows',n)
    elif metric=='valid_and_correct_probability':num=value.get('expected_good');den=value.get('windows',n)
    elif metric=='valid_and_bad_probability':num=value.get('expected_bad');den=value.get('windows',n)
    elif metric=='conditional_correct_rate':num=value.get('expected_good');den=value.get('expected_valid')
    elif metric=='conditional_bad_rate':num=value.get('expected_bad');den=value.get('expected_valid')
    elif metric=='precision_lt2':num=value.get('n_lt2',value.get('expected_correct'))
    elif metric=='precision_gt5':num=value.get('n_gt5',value.get('expected_bad'))
    elif metric=='low_parent_retention':num=value.get('low_parents_kept');den=value.get('low_parents_before')
    elif metric in ['coverage0','coverage1']:
        detail=value.get(metric+'_detail',{});num=detail.get('occupied_cells');den=detail.get('valid_grid_cells')
    return num,den
def stat(rows,metrics,methods,comparisons=()):
    x=metric_rows_summary(rows,metrics,methods,comparisons);x['metric_denominators']={}
    for m in methods:
        x['metric_denominators'][m]={}
        for k in metrics:
            usable=[r for r in rows if r['methods'].get(m,{}).get(k) is not None]
            vals=[support(r['methods'][m],k,r.get('count',0)) for r in usable]
            dens=[d for n,d in vals if d is not None];nums=[n for n,d in vals if n is not None]
            den=sum(dens) if dens else 0;num=sum(nums) if len(nums)==len(vals) and vals else None
            x['metric_denominators'][m][k]=dict(evaluable_pairs=len(usable),scenes=len({r['scene'] for r in usable}),observations=den,
                pooled_numerator=num,pooled_denominator=den,pooled_value=ratio(num,den) if num is not None else None,
                aggregation='Main value: pair mean within scene, then equal scenes. Pooled ratio is separately labeled; median/correlation/AUSE are non-additive.')
    return x
def order_ids(cp,ordinal):
    order=np.lexsort((ordinal,cp));ids=np.empty(len(cp),np.int64);ids[order]=np.arange(len(cp));return ids
def select(a,score,fraction):
    if len(score)!=len(a['cp']) or not np.isfinite(score).all():raise FloatingPointError('All-output selection score absent/nonfinite')
    n=len(score);k=int(np.ceil(fraction*n));order=np.lexsort((a['ordinal'],a['cp'],-score));chosen=np.sort(order[:k])
    if not k:return chosen,dict(n=n,k=0,tie_straddles=False)
    cut=score[order[k-1]];equal=int((score==cut).sum());above=int((score>cut).sum());selected=k-above;excluded=equal-selected
    return chosen,dict(n=n,k=k,cutoff=float(cut),equal_at_cutoff=equal,selected_at_cutoff=selected,excluded_at_cutoff=excluded,tie_straddles=bool(excluded>0 and selected>0))
def array_epipolar(a,source,target):
    protocol=geometry_protocol();sys.path.insert(0,str(Path(protocol['hybrid_root']).parent))
    from hybrid_match.evaluation.metrics import symmetric_epipolar_distance
    value=symmetric_epipolar_distance(source*a['scale0'],target*a['scale1'],a['K0'],a['K1'],a['T'])
    finite(value,'all-prediction epipolar distances');return value
def coverage(a,ix,source=None,target=None,low=None):
    source=a['source'] if source is None else source;target=a['target'] if target is None else target
    low=a['low'] if low is None else low
    c0=occupancy(source[ix],a['hw0']);c1=occupancy(target[ix],a['hw1'])
    before=np.unique(a['cp'][low]);after=np.unique(a['cp'][ix]);kept=int(np.isin(before,after).sum())
    v=a['valid'][ix];e=a['epe'][ix][v];ep=a['epipolar'][ix]
    return dict(n=len(ix),valid=int(v.sum()),epe_lt1=int((e<1).sum()),epe_lt2=int((e<2).sum()),epipolar_correct=int((ep<1e-4).sum()),
        GT_evaluable_rate=ratio(int(v.sum()),len(ix)),lt1=ratio(int((e<1).sum()),int(v.sum())),lt2=ratio(int((e<2).sum()),int(v.sum())),
        epipolar_precision=ratio(int((ep<1e-4).sum()),len(ix)),low_parents_before=len(before),low_parents_kept=kept,low_parent_retention=ratio(kept,len(before)),
        coverage0=c0['occupancy'],coverage1=c1['occupancy'],coverage0_detail=c0,coverage1_detail=c1)
def ranking(r,a,route,mask,include_offline=True):
    scores={'epistemic':-a['epistemic'],'aleatoric':-a['aleatoric'],'total':-(a['epistemic']+a['aleatoric']),'mconf':a['mconf'],'fine':a['fine'],'S_m':a['S_m']}
    if not include_offline:scores={k:v for k,v in scores.items() if k not in ['fine','S_m']}
    joint=mask&a['valid']
    for s in scores.values():joint&=np.isfinite(s)
    e=a['epe'][joint];ids=order_ids(a['cp'],a['ordinal'])[joint]
    methods={m:dict(n=len(e),Spearman=spearman(s[joint],e),**sparsification(s[joint],e,ids)) for m,s in scores.items()}
    curves={m:curve_stats(e,s[joint],ids) for m,s in scores.items()}
    undefined={}
    for m,s in scores.items():
        undefined[m]='insufficient_samples' if len(e)<3 else 'constant_score' if len(np.unique(s[joint]))<2 else 'constant_error' if len(np.unique(e))<2 else None
    return dict(**base(r),comparison_scope='common_with_offline_scores' if include_offline else 'full_output_own_scores_and_mconf',count=int(joint.sum()),all_valid=int((mask&a['valid']).sum()),missing_offline_fine=int((mask&a['valid']&~np.isfinite(a['fine'])).sum()),
        methods=methods,curves=curves,undefined=undefined,uncertainty_Spearman=spearman(a['epistemic'][joint],a['aleatoric'][joint]))
def window_rows(r,a,low=False):
    cp=a['cp'];ordinal=a['ordinal'];valid=a['valid'];e=a['epe'];scores={'epistemic':-a['epistemic'],'fine':a['fine']}
    ordered=np.argsort(cp,kind='stable');windows=np.split(ordered,np.flatnonzero(np.diff(cp[ordered]))+1) if len(cp) else []
    if low:windows=[g for g in windows if a['low'][g[0]]]
    multi=[g for g in windows if len(g)>=2];joint=[g for g in windows if valid[g].sum()>=2]
    act={m:[[],[],[]] for m in ['epistemic','fine','random','oracle']}
    for g in multi:
        for m,score in scores.items():
            idx=g[np.lexsort((ordinal[g],cp[g],-score[g]))[0]]
            act[m][0].append(float(valid[idx] and e[idx]<2));act[m][1].append(float(valid[idx]));act[m][2].append(float(valid[idx] and e[idx]>5))
        v=valid[g];act['random'][0].append(float(((e[g]<2)&v).sum()/len(g)));act['random'][1].append(float(v.sum()/len(g)));act['random'][2].append(float(((e[g]>5)&v).sum()/len(g)))
        ev=e[g][v];best=float(ev.min()) if len(ev) else None
        act['oracle'][0].append(float(best is not None and best<2));act['oracle'][1].append(float(best is not None));act['oracle'][2].append(float(best is not None and best>5))
    actual=dict(**base(r),count=len(multi),methods={m:dict(**actual_stats(*v),windows=len(multi),expected_good=float(np.sum(v[0])),expected_valid=float(np.sum(v[1])),expected_bad=float(np.sum(v[2]))) for m,v in act.items()})
    chosen={m:[] for m in ['epistemic','fine','oracle']};rg=[];rb=[];vals=[];weights=[]
    for g in joint:
        ix=g[valid[g]]
        for m,score in scores.items():
            k=ix[np.lexsort((ordinal[ix],cp[ix],-score[ix]))[0]];chosen[m].append(float(e[k]))
        chosen['oracle'].append(float(e[ix].min()));rg.append(float((e[ix]<2).mean()));rb.append(float((e[ix]>5).mean()))
        vals.extend(e[ix].tolist());weights.extend([1/len(ix)]*len(ix))
    methods={}
    for m,v in chosen.items():
        v=np.asarray(v);methods[m]=dict(**common_stats(v,v<2,v>5),n=len(v),n_lt2=int((v<2).sum()),n_gt5=int((v>5).sum()))
    methods['random']=dict(precision_lt2=float(np.mean(rg)) if rg else None,precision_gt5=float(np.mean(rb)) if rb else None,
        median_EPE=weighted_median(vals,weights),n=len(joint),expected_correct=float(np.sum(rg)),expected_bad=float(np.sum(rb)),mixture_weight=float(np.sum(weights)),median_name='均匀随机选择分布的中位 EPE')
    common=dict(**base(r),count=len(joint),methods=methods)
    den=dict(**base(r),region='low_texture' if low else 'all',windows=len(windows),actual_multi=len(multi),common_valid_multi=len(joint),
        single_valid=sum(int(valid[g].sum())==1 for g in windows),zero_valid=sum(int(valid[g].sum())==0 for g in windows),
        candidate_counts=distributions(np.asarray([len(g) for g in windows])),valid_candidate_counts=distributions(np.asarray([int(valid[g].sum()) for g in windows])))
    return actual,common,den
def analyze(run):
    run=Path(run);out=Path(read(run/'context.json')['results']);groups={};ranking_groups={};window_groups={};denoms=[];filters={};ties=[];bins=[];coverage_rows=[];counts={};curves={};point_denominators=[]
    routes=[s for s in ['A','B'] if (run/('final_'+s)/'summary.json').exists() and read(run/('final_'+s)/'summary.json')['status']=='committed']
    for route in routes:
        counts[route]=dict(outputs=0,valid_GT=0,inside=0,low_outputs=0,low_valid_GT=0,pairs=0)
        for idx,r in enumerate(records('epe')):
            a=get_unit(run,'final_'+route,r,True);n=len(a['cp']);v=a['valid'];e=a['epe'];low=a['low'];zero=np.linalg.norm(a['offset'],axis=1)
            epimethods={'B':a['epipolar'],'t0':array_epipolar(a,a['source'],a['trajectory'][0]),'t1':array_epipolar(a,a['source'],a['native_one_target']),'t4':array_epipolar(a,a['native_source'],a['native_target'])} if route=='B' else {}
            counts[route]['outputs']+=n;counts[route]['valid_GT']+=int(v.sum());counts[route]['inside']+=int(a['inside'].sum());counts[route]['low_outputs']+=int(low.sum());counts[route]['low_valid_GT']+=int((v&low).sum());counts[route]['pairs']+=1
            for region,mask in [('all',np.ones(n,bool)),('low_texture',low)]:
                invalid=a['gt_reasons'][mask&~v]
                reasons={str(k):int(c) for k,c in zip(*np.unique(invalid,return_counts=True))}
                point_denominators.append(dict(**base(r),route=route,region=region,outputs=int(mask.sum()),valid_GT=int((mask&v).sum()),
                    parents_with_outputs=len(np.unique(a['cp'][mask])),parents_with_valid_GT=len(np.unique(a['cp'][mask&v])),
                    inside=int((mask&a['inside']).sum()),outside=int((mask&v&~a['inside']).sum()),invalid_reasons=reasons))
            for region,mask in [('all',np.ones(n,bool)),('low_texture',low),('inside',a['inside']),('outside',v&~a['inside'])]:
                vm=v&mask
                if route=='B':
                    methods={route:error_metrics(e[vm]),'t0':error_metrics(a['trajectory_epe'][0,vm]),'t1':error_metrics(a['native_one_epe'][vm]),'t4':error_metrics(a['native_epe'][vm])}
                    for method,eps in epimethods.items():methods[method].update(epipolar_precision=ratio(int((eps[mask]<1e-4).sum()),int(mask.sum())),n_epi=int(mask.sum()),n_epi_correct=int((eps[mask]<1e-4).sum()))
                    groups.setdefault('B/'+region,[]).append(dict(**base(r),count=int(vm.sum()),methods=methods))
                else:
                    groups.setdefault('A/full/'+region,[]).append(dict(**base(r),count=int(vm.sum()),methods={'A':error_metrics(e[vm]),'zero':error_metrics(zero[vm])}))
                    pe=a['native_parent_epe'];choice=a['native_fine_choice'];one=np.full(n,np.nan);can=choice>=0;validone=np.zeros(n,bool);validone[can]=a['native_valid'][choice[can]]
                    one[validone]=a['native_epe'][choice[validone]]
                    for comp,common in [('median',vm&np.isfinite(pe)),('fine_one',vm&np.isfinite(one)),('joint',vm&np.isfinite(pe)&np.isfinite(one))]:
                        methods={'A':error_metrics(e[common]),'zero':error_metrics(zero[common])}
                        if comp in ['median','joint']:methods['native_median']=error_metrics(pe[common])
                        if comp in ['fine_one','joint']:methods['native_fine_one']=error_metrics(one[common])
                        groups.setdefault('A/'+comp+'/'+region,[]).append(dict(**base(r),count=int(common.sum()),methods=methods))
                if region in ['all','low_texture']:ranking_groups.setdefault(route+'/'+region,[]).append(ranking(r,a,route,mask))
                if route=='A' and region in ['all','low_texture']:ranking_groups.setdefault('A/full_'+region,[]).append(ranking(r,a,route,mask,include_offline=False))
            if route=='A':
                linf=np.max(np.abs(a['offset']),axis=1);edges=[(0,4),(4,8),(8,12),(12,float('inf'))]
                for region,base_mask in [('all',v),('low_texture',v&low)]:
                    for bi,(lo,hi) in enumerate(edges):
                        mask=base_mask&(linf>=lo if bi==0 else linf>lo)&(linf<=hi)
                        bins.append(dict(**base(r),region=region,bin=['[0,4]','(4,8]','(8,12]','>12'][bi],train_boundary_equal8=int((mask&(linf==8)).sum()),**error_metrics(e[mask])))
                choice=a['native_fine_choice'];keep=choice>=0;ix=choice[keep]
                coverage_rows.append(dict(**base(r),route='A',method='native_fine_one',matches=len(ix),GT_evaluable=int(a['native_valid'][ix].sum()),coverage0=occupancy(a['native_source'][ix],a['hw0']),coverage1=occupancy(a['native_target'][ix],a['hw1']),
                    no_fine_parent=int((~keep).sum()),fine_but_selected_GT_missing=int((~a['native_valid'][ix]).sum())))
            else:
                for islow in [False,True]:
                    actual,common,den=window_rows(r,a,islow);name='low_texture' if islow else 'all'
                    window_groups.setdefault(name+'/actual',[]).append(actual);window_groups.setdefault(name+'/common',[]).append(common);denoms.append(den)
                bins.append(dict(**base(r),route='B',valid=int(v.sum()),initial_EPE_gt3=int((v&(zero>3)).sum()),initial_EPE_gt5=int((v&(zero>5)).sum()),true_offset_Linf_ge3=int((v&~a['inside']).sum())))
            seed=int(digest(dict(seed=2026,pair_id=r['pair_id'],purpose='fixed_child_random_prefix'))[:8],16)
            permutation=np.random.default_rng(seed).permutation(n);randomscore=np.empty(n,float);randomscore[permutation]=np.arange(n,0,-1,dtype=float)
            scores={'epistemic':-a['epistemic']}
            if route=='B':scores.update(S_m=a['S_m'],random=randomscore)
            combos={'all':np.arange(n)}
            for name,score in scores.items():
                for rate in [.95,.8]:
                    ix,tie=select(a,score,rate);key=name+'_'+str(int(rate*100));combos[key]=ix;ties.append(dict(**base(r),route=route,method=name,rate=rate,random_seed=seed,**tie))
            for key,ix in combos.items():
                for region in ['all','low_texture']:
                    sub=ix if region=='all' else ix[low[ix]]
                    row=coverage(a,sub)
                    filters.setdefault(route+'/'+region,[]).append(dict(**base(r),method=key,**row))
            coverage_rows.append(dict(**base(r),route=route,method='full',**coverage(a,np.arange(n))))
            beat(run,'report',phase='CPU fixed comparisons',route=route,pair=idx+1)
    quality={}
    for name,rs in groups.items():
        methods=list(rs[0]['methods']);primary='B' if name.startswith('B') else 'A'
        comps=[(primary,m) for m in methods if m!=primary]
        quality[name]=stat(rs,['mean_EPE','median_EPE','lt1','lt2','lt3','gt5']+(['epipolar_precision'] if name.startswith('B/') else []),methods,comps)
    rankings={}
    for name,rs in ranking_groups.items():
        methods=list(rs[0]['methods'])
        rankings[name]=stat(rs,['Spearman','AUSE'],methods,[('epistemic',m) for m in ['fine','S_m','mconf'] if m in methods])
        rankings[name]['undefined_reasons']={m:{reason:sum(r['undefined'][m]==reason for r in rs) for reason in ['insufficient_samples','constant_score','constant_error']} for m in methods}
        rankings[name]['uncertainty_correlation']=stat([dict(**base(r),methods={'ua_ue':dict(Spearman=r['uncertainty_Spearman'],n=r['count'])}) for r in rs],['Spearman'],['ua_ue'])
        curves[name]=curve_summary(rs,methods)
    windows={}
    for name,rs in window_groups.items():
        metrics=['GT_evaluable_rate','valid_and_correct_probability','conditional_correct_rate','valid_and_bad_probability','conditional_bad_rate'] if name.endswith('actual') else ['precision_lt2','precision_gt5','median_EPE']
        windows[name]=stat(rs,metrics,['epistemic','fine','random','oracle'],[('epistemic','fine'),('epistemic','random'),('oracle','epistemic')])
    filter_summary={}
    for name,rs in filters.items():
        bypair={}
        for r in rs:bypair.setdefault(r['pair_id'],dict(**base(r),methods={}))['methods'][r['method']]=r
        ms=list(next(iter(bypair.values()))['methods'])
        filter_summary[name]=stat(list(bypair.values()),['GT_evaluable_rate','lt1','lt2','epipolar_precision','low_parent_retention','coverage0','coverage1'],ms,[(m,'all') for m in ms if m!='all'])
    decisions={g:dict(status='不可判定',reason='Required completed route unavailable') for g in ['G8','G9','G10']}
    def interval_defined(v):
        return v.get('ci95') is not None and len(v['ci95'])==2 and all(x is not None and np.isfinite(x) for x in v['ci95'])
    if 'B/all' in quality:
        v=quality['B/all']['paired_differences']['B-t1']['lt1']
        if interval_defined(v):
            lo,hi=v['ci95']
            conclusion='该指标上优于一步' if lo>0 else '该指标上达到一步的预设非劣效容差' if lo>=-.01 else '下降超过容差' if hi<-.01 else '证据不足'
            decisions['G8']=dict(status='completed',value=v,conclusion=conclusion,tolerance=-.01)
        else:decisions['G8']=dict(status='undefined',value=v,reason='No defined paired scene interval; insufficient common support')
        v=rankings['B/all']['paired_differences']['epistemic-fine']['Spearman']
        if interval_defined(v):
            lo,hi=v['ci95']
            decisions['G9']=dict(status='completed',value=v,conclusion='排序优于fine' if hi<0 else '排序弱于fine' if lo>0 else '证据不足')
        else:decisions['G9']=dict(status='undefined',value=v,reason='No defined paired scene interval; insufficient common support or constant score/error')
    if 'A/median/all' in quality:
        v=quality['A/median/all']['paired_differences']['A-native_median']['lt2'];precision=v['ci95'][0]>=-.02 if interval_defined(v) else None
        ep=run/'efficiency/summary.json';eff=read(ep) if ep.exists() else {};lat=eff.get('summary',{}).get('scene_equal',{})
        available='A' in lat and 'native4' in lat;speed=lat['A']['seconds']<lat['native4']['seconds'] if available else None
        decisions['G10']=dict(status='completed' if available and precision is not None else 'undefined',precision=v,precision_pass=precision,latency_pass=speed,latency=lat,
            conclusion='本验证集共同父窗口质量在容差内，实测细阶段更快' if precision and speed else '精度部分未满足容差' if precision is False else '精度区间无定义' if precision is None else '时间依赖不可判定' if not available else '实测细阶段未更快')
    result=dict(status='completed' if len(routes)==2 else 'partial',routes=routes,counts=counts,quality=quality,ranking=rankings,within_window=windows,filter=filter_summary,decisions=decisions,
        random_median_name='均匀随机选择分布的中位 EPE',random_median_definition='每窗口等权，窗口内有效候选权重1/v；每pair混合分布加权中位数，再场景等权',
        original_ordinal='Native coarse-parent block order; within each block original torch.topk fine-confidence descending order. Exact native ties retain framework output ordinal. Concrete selection breaks ties by parent ID then original ordinal.',
        random_budget_seed='Same Phase4 digest(seed=2026,pair_id,purpose=fixed_child_random_prefix), unique random prefix scores',
        interpretation='Single training seed, first two epochs of frozen scene-balanced plan, frozen SLiM, R832; repeated-analysis 100-pair15-scene validation. A comparisons with different source pixels concern parent-window output quality.')
    save(out/'metrics.json',result);save(out/'decisions.json',decisions);save(out/'retention_curves.json',curves)
    save(run/'report/coordinate_pair_rows.json',groups);save(run/'report/ranking_pair_rows.json',ranking_groups);save(run/'report/window_pair_rows.json',window_groups);save(run/'report/filter_pair_rows.json',filters)
    csvout(out/'point_denominators.csv',point_denominators);save(out/'point_denominators.json',point_denominators)
    csvout(out/'window_denominators.csv',denoms);csvout(out/'offset_bins_and_B_initial_tail.csv',bins);csvout(out/'filter_ties.csv',ties);csvout(out/'coverage.csv',coverage_rows)
    main=[];deltas=[]
    for family,collection in [('coordinate',quality),('ranking',rankings),('window',windows),('filter',filter_summary)]:
        for name,s in collection.items():
            for method,vs in s['scene_equal'].items():
                for metric,value in vs.items():main.append(dict(family=family,analysis=name,method=method,metric=metric,value=value,**s['metric_denominators'][method][metric]))
            for comp,vs in s['paired_differences'].items():
                for metric,v in vs.items():deltas.append(dict(v,family=family,analysis=name,comparison=comp,metric=metric))
    csvout(out/'main_table.csv',main);csvout(out/'paired_differences.csv',deltas);save(out/'window_denominators.json',denoms);save(out/'filter_counts.json',filters)
    return result
