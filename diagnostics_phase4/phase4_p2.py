from common4 import *
from c_statistics import parent_bundle
def one_row(r,e,scores,ids):
    methods={}
    for m,s in scores.items():
        stat=dict(Spearman=spearman(s,e),**sparsification(s,e,ids))
        for t in [1,2,3]:stat['AUROC_lt'+str(t)]=ranks((e<t).astype(np.int8),s)['score_AUROC']
        methods[m]=stat
    return dict(**base(r),count=len(e),methods=methods,curves={m:curve_stats(e,s,ids) for m,s in scores.items()})
def gate(summary):
    d=summary['paired_differences']
    vals={a+'-'+b+'/'+metric:d[a+'-'+b][metric] for a,b in [('S_m','fine'),('S_m','mconf')] for metric in ['Spearman','AUSE']}
    if any(v['ci95'] is None for v in vals.values()):return dict(status='不可判定',evidence=vals)
    if all(v['ci95'][1]<0 for v in vals.values()):status='全面更优';message='组合在两项指标上均优于两个单项'
    elif vals['S_m-mconf/Spearman']['ci95'][1]<0 and vals['S_m-fine/AUSE']['ci95'][1]<0:status='有折中价值';message='组合弥补了各单项的弱项：mconf 的 Spearman 与 fine 的 AUSE'
    else:status='未成立';message='未满足预设的全面更优或折中价值条件'
    return dict(status=status,message=message,evidence=vals)
def score_distribution(run):
    stores={};windows=collections.Counter();order_differences=0
    for r in records('epe'):
        a=data(run,r)
        for scope,mask in [('all_children',np.ones(len(a['cp']),bool)),('valid_GT',a['valid'])]:
            for m in ['head_p','mconf','fine']:
                if m in a:stores.setdefault((scope,m),[]).append(a[m][mask])
        for m in ['head_p','mconf']:
            _,ix=np.unique(a['cp'],return_index=True);stores.setdefault(('distinct_output_parents',m),[]).append(a[m][ix])
        inds=np.argsort(a['cp'],kind='stable');groups=np.split(inds,np.flatnonzero(np.diff(a['cp'][inds]))+1)
        for g in groups:
            windows['all_output_windows']+=1
            if np.any(np.diff(a['fine'][g])>0):order_differences+=1
            for m in ['mconf','head_p']:
                if a[m][g[0]]==0:windows[m+'_zero_parent_windows']+=1;windows[m+'_zero_children']+=len(g)
            if len(g)>1 and len(np.unique(a['fine'][g]))<len(g):windows['fine_tied_windows']+=1
    out=[]
    for (scope,m),arrays in stores.items():
        v=np.concatenate(arrays);unique,n=np.unique(v,return_counts=True)
        out.append(dict(scope=scope,method=m,**describe(v),zero_count=int((v==0).sum()),unique_values=len(unique),
            duplicate_excess_count=len(v)-len(unique),duplicate_excess_fraction=1-len(unique)/len(v),tied_point_count=int(n[n>1].sum()),tied_point_fraction=float(n[n>1].sum()/len(v))))
    obj=dict(rows=out,windows=dict(windows),windows_where_float32_fine_not_descending_in_native_order=order_differences,
        caveat='Counts describe exact ties in finite-precision cached values, not a proof every tie was caused by rounding. Child inherited parent-score ties are structural; distinct_output_parents removes that multiplicity.',
        original_order='Historical native fine output row order: coarse-window order then torch.topk(default sorted=True) on native dual-softmax within each window, followed by a global relative threshold. Native equal-score torch.topk order is retained. Phase3 recomputed fine in float32; no assumption of identical finite-precision ordering.',
        selection_score_source='src/slim.py:834,866-904; original sim_matrix_f dual softmax, top_k=min(max_fine_matches/M,Lf), threshold>global maximum*fine_match_thres.',
        within_window='For a positive parent factor, products preserve within-window fine ranking mathematically; zero factors produce full ties. Products are float64; no weights/temperature fitted.')
    artifact(run,'score_distributions_and_ties',obj,out)
def main(run):
    caps=read(Path(run)/'P0/summary.json')['capabilities']['P2'];methods=[m for m in SCORE_NAMES if caps[m]]
    if not methods:
        obj=dict(status='不可判定',reason='No verified valid-GT score population');artifact(run,'P2_ranking',obj);return obj
    comparisons=[(a,b) for a,b in PAIR_COMPS if a in methods and b in methods];groups={}
    for i,r in enumerate(records('epe')):
        stored=own_read(run,'P2',r)
        if stored is None:
            a=data(run,r);scores={m:a[m] for m in methods};err=a['EPE'][-1];per={}
            for scope,mask in [('all',a['valid']),('low_texture',a['valid']&a['low'].astype(bool))]:
                cr=one_row(r,err[mask],{m:s[mask] for m,s in scores.items()},a['tie_ids'][mask])
                ids=np.unique(a['cp'][mask]);ep=np.asarray([np.median(err[(a['cp']==pid)&mask]) for pid in ids])
                ps={m:np.asarray([np.median(s[(a['cp']==pid)&mask]) for pid in ids]) for m,s in scores.items()}
                pr=one_row(r,ep,ps,ids)
                per[scope+'/child']=cr;per[scope+'/parent']=pr
            stored=per;own_commit(run,'P2',r,stored)
        for k,v in stored.items():groups.setdefault(k,[]).append(v)
        heartbeat(run,'P2',committed_pairs=i+1,total_pairs=100)
    for k,rows in list(groups.items()):
        if k.startswith('all/'):groups['without_0003/'+k.split('/')[1]]=[r for r in rows if r['scene']!='0003']
    metrics=['Spearman','AUSE','AUROC_lt1','AUROC_lt2','AUROC_lt3'];summaries={};flat=[];diff=[];curves={}
    for k,rows in groups.items():
        s=metric_rows_summary(rows,metrics,methods,comparisons);summaries[k]=s;flat+=flat_metrics(s,k);diff+=flat_diffs(s,k)
        curves[k]=curve_summary(rows,methods)
    G5=gate(summaries['all/child']) if all(caps[m] for m in ['S_m','fine','mconf']) else dict(status='不可判定',reason='One required score unavailable')
    artifact(run,'P2_ranking',dict(status='completed',summaries=summaries),flat)
    artifact(run,'P2_paired_differences',dict(comparisons={k:v['paired_differences'] for k,v in summaries.items()}),diff)
    crflat=[dict(analysis=k,method=m,**row) for k,obj in curves.items() for m,d in obj['methods'].items() for row in d['scene_equal']]
    artifact(run,'P2_retention_curves',curves,crflat);artifact(run,'G5',G5)
    save(Path(run)/'P2/pair_rows.json',groups);score_distribution(run)
    result=dict(status='completed',pairs=100,methods=methods,G5=G5);save(Path(run)/'P2/summary.json',result);return result
