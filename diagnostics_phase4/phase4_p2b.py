from common4 import *
def weighted_median(values,weights):
    values=np.asarray(values,float);weights=np.asarray(weights,float)
    if not len(values) or weights.sum()<=0:return None
    order=np.argsort(values,kind='stable');c=np.cumsum(weights[order])
    return float(values[order[min(int(np.searchsorted(c,0.5*c[-1],side='left')),len(order)-1)]])
def actual_stats(selected_good,selected_valid,selected_bad):
    n=len(selected_good)
    g=float(np.sum(selected_good));v=float(np.sum(selected_valid));b=float(np.sum(selected_bad))
    return dict(GT_evaluable_rate=ratio(v,n),valid_and_correct_probability=ratio(g,n),conditional_correct_rate=ratio(g,v),
                valid_and_bad_probability=ratio(b,n),conditional_bad_rate=ratio(b,v))
def common_stats(selected,good,bad):
    return dict(precision_lt2=float(np.mean(good)) if len(good) else None,precision_gt5=float(np.mean(bad)) if len(bad) else None,
                median_EPE=float(np.median(selected)) if len(selected) else None)
def main(run):
    caps=read(Path(run)/'P0/summary.json')['capabilities']
    if not caps['P2'].get('fine'):
        result=dict(status='不可判定',reason='Verified fine score on GT-valid candidates missing');artifact(run,'P2b_within_window',result);return result
    groups={};denoms=[];window_details=[]
    for i,r in enumerate(records('epe')):
        saved=own_read(run,'P2b',r)
        if saved is None:
            a=data(run,r);cp=a['cp'];valid=a['valid'];e=a['EPE'][-1];fine=a['fine']
            order=np.argsort(cp,kind='stable');windows=np.split(order,np.flatnonzero(np.diff(cp[order]))+1)
            outputs={};details=[]
            for region in ['all','low_texture']:
                pool=[g for g in windows if region=='all' or a['low'][g[0]]]
                ns=[len(g) for g in pool];vs=[int(valid[g].sum()) for g in pool]
                multi=[g for g in pool if len(g)>=2];joint=[g for g in pool if int(valid[g].sum())>=2]
                actual_f=[[],[],[]];actual_r=[[],[],[]]
                for g in multi:
                    candidate=g[np.lexsort((a['ordinal'][g],a['cp'][g],-fine[g]))[0]]
                    actual_f[0].append(float(valid[candidate] and e[candidate]<2));actual_f[1].append(float(valid[candidate]));actual_f[2].append(float(valid[candidate] and e[candidate]>5))
                    v=valid[g];actual_r[0].append(float(((e[g]<2)&v).sum()/len(g)));actual_r[1].append(float(v.sum()/len(g)));actual_r[2].append(float(((e[g]>5)&v).sum()/len(g)))
                act=dict(**base(r),count=len(multi),methods=dict(fine=actual_stats(*actual_f),random=actual_stats(*actual_r)))
                selected=[];oracle=[];random_good=[];random_bad=[];vals=[];weights=[]
                for g in joint:
                    ok=g[valid[g]];winner=ok[np.lexsort((a['ordinal'][ok],a['cp'][ok],-fine[ok]))[0]]
                    selected.append(float(e[winner]));oracle.append(float(e[ok].min()))
                    random_good.append(float((e[ok]<2).mean()));random_bad.append(float((e[ok]>5).mean()))
                    vals.extend(e[ok].tolist());weights.extend([1/len(ok)]*len(ok))
                selected=np.asarray(selected);oracle=np.asarray(oracle)
                co=dict(**base(r),count=len(joint),methods=dict(fine=common_stats(selected,selected<2,selected>5),oracle=common_stats(oracle,oracle<2,oracle>5),
                    random=dict(precision_lt2=float(np.mean(random_good)) if random_good else None,precision_gt5=float(np.mean(random_bad)) if random_bad else None,
                                median_EPE=weighted_median(vals,weights))))
                outputs[region+'/actual']=act;outputs[region+'/common_valid']=co
                details.append(dict(**base(r),region=region,total_windows=len(pool),actual_windows=len(multi),common_valid_windows=len(joint),
                    windows_one_valid=sum(v==1 for v in vs),windows_no_valid=sum(v==0 for v in vs),windows_any_valid=sum(v>0 for v in vs),
                    n_distribution=describe(ns),v_distribution=describe(vs),n_histogram=dict(collections.Counter(ns)),v_histogram=dict(collections.Counter(vs))))
                for g in pool:
                    details[-1].setdefault('window_counts',[]).append(dict(parent_id=int(cp[g[0]]),n=len(g),v=int(valid[g].sum())))
            saved=dict(rows=outputs,denominators=details);own_commit(run,'P2b',r,saved)
        for k,x in saved['rows'].items():groups.setdefault(k,[]).append(x)
        denoms.extend(saved['denominators'])
        heartbeat(run,'P2b',committed_pairs=i+1,total_pairs=100)
    summaries={};flat=[];differences=[]
    for k,rows in groups.items():
        if k.endswith('/actual'):
            metrics=['GT_evaluable_rate','valid_and_correct_probability','conditional_correct_rate','valid_and_bad_probability','conditional_bad_rate'];methods=['fine','random'];comps=[('fine','random')]
        else:metrics=['precision_lt2','precision_gt5','median_EPE'];methods=['fine','random','oracle'];comps=[('fine','random'),('oracle','fine')]
        s=metric_rows_summary(rows,metrics,methods,comps)
        for comp,vals in s['paired_differences'].items():
            for metric,v in vals.items():
                lower=metric in ['precision_gt5','median_EPE','valid_and_bad_probability','conditional_bad_rate']
                signs=np.asarray(list(v['per_scene_difference'].values()))*(-1 if lower else 1)
                v.update(win=int((signs>1e-12).sum()),tie=int((np.abs(signs)<=1e-12).sum()),loss=int((signs< -1e-12).sum()),higher_is_better=not lower)
        summaries[k]=s;flat+=flat_metrics(s,k);differences+=flat_diffs(s,k)
    counts={}
    for region in ['all','low_texture']:
        d=[r for r in denoms if r['region']==region]
        keys=['total_windows','actual_windows','common_valid_windows','windows_one_valid','windows_no_valid','windows_any_valid']
        alln=[x['n'] for r in d for x in r.get('window_counts',[])];allv=[x['v'] for r in d for x in r.get('window_counts',[])]
        counts[region]=dict(**{k:sum(x[k] for x in d) for k in keys},n_distribution=describe(alln),v_distribution=describe(allv),
            n_histogram=dict(collections.Counter(alln)),v_histogram=dict(collections.Counter(allv)))
    summary=dict(status='completed',summaries=summaries,denominators=counts,
        random_median_name='均匀随机选择分布的中位 EPE',
        random_median_definition='Within each pair each eligible window has weight 1, each valid candidate weight 1/v_i. Weighted median is inverse empirical CDF at 0.5 (first cumulative weight >= half). Then average pairs within scene and scenes equally.',
        actual_conditional_definition='Within each image pair: ratio of expected correct (or bad) count to expected evaluable count; not a mean of per-window conditional ratios. Pair values then scene aggregated.',
        actual_status='完成' if caps['P2b_actual'] else '不可判定',candidate_scope='Only native already-output fine children, not all window lattice candidates.')
    artifact(run,'P2b_within_window',summary,flat);artifact(run,'P2b_paired_differences',dict(rows=differences),differences)
    artifact(run,'P2b_window_denominators',counts)
    save(Path(run)/'P2b/window_details.json',denoms);save(Path(run)/'P2b/pair_rows.json',groups)
    save(Path(run)/'P2b/summary.json',summary)
    return dict(status='completed',denominators=counts)
