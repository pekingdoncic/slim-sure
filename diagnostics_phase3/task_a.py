"""A0-A3: CPU-only analysis of committed Phase 2 final EPE payloads."""
from common import *
def main(run):
    run=Path(run);out=run/'A';recs=records('epe');rows=[];allrows={};curverows={'all':[],'low_texture':[]};cross=[];denoms=[];total=collections.Counter()
    for index,r in enumerate(recs):
        receipt=committed(run,'A',r)
        if receipt is None:
            p,ref=cache(r);raw=p['raw_id'];labels=p['labels'];low=p['low'].astype(bool)
            valid=p['child_valid'].astype(bool);agg=parent_errors(raw,p['child_parent_id'],p['child_epe'],valid)
            ids=agg['parent_ids'];pos=positions(raw,ids);err=agg['error'];finite(err,'parent EPE')
            epemask=np.zeros(len(raw),bool);epemask[pos]=True
            errors=np.full(len(raw),np.nan);errors[pos]=err
            output=np.isin(raw,p['child_parent_id'])
            scores={m:p[m] for m in METHODS}
            row=errorrow(r,err,{m:p[m][pos] for m in METHODS},ids);row['counts']=agg['counts']
            cls={};curve={};xrows=[];drows=[]
            for region,regionmask in [('all',np.ones(len(raw),bool)),('low_texture',low)]:
                r1=regionmask&(labels>=0);r2=r1&epemask
                cls[region+'/R1']=classify(r,labels,scores,r1)
                cls[region+'/R2']=classify(r,labels,scores,r2)
                for threshold in [1,2,3,5]:
                    target=(errors<threshold).astype(np.int8)
                    for name,mask in [('R3',r2),('R3a',r2&(labels==0)),('R3b',r2&(labels==1))]:
                        cls[region+'/'+name+'_lt'+str(threshold)]=classify(r,target,scores,mask)
                for label in [-1,0,1]:
                    mask=regionmask&(labels==label);good=mask&epemask;e=errors[good]
                    xrows.append(dict(**base(r),region=region,coarse_label=label,original_parents=int(mask.sum()),with_fine_output=int((mask&output).sum()),
                        without_fine_output=int((mask&~output).sum()),without_valid_final_GT=int((mask&output&~epemask).sum()),denominator=int(good.sum()),
                        epe_lt1=int((e<1).sum()),epe_lt2=int((e<2).sum()),epe_lt3=int((e<3).sum()),epe_gt5=int((e>5).sum())))
                mask=regionmask&epemask;rp=np.flatnonzero(mask)
                curve[region]=dict(**base(r),curves={m:curve_stats(errors[rp],p[m][rp],raw[rp]) for m in METHODS})
                drows.append(dict(**base(r),region=region,original_parents=int(regionmask.sum()),with_fine_output=int((regionmask&output).sum()),
                    valid_final_GT=int(mask.sum()),evaluable_pair=bool(mask.any())))
            result=dict(A0=row,classification=cls,curves=curve,cross=xrows,denominators=drows)
            arrays=dict(raw_id=raw,labels=labels,low=low,has_fine_output=output,valid_final_parent=epemask,parent_epe=errors,**scores)
            receipt=commit(run,'A',r,result,arrays,ref)
        result=result_of(run,'A',receipt);rows.append(result['A0']);total.update(result['A0']['counts'])
        for name,one in result['classification'].items():allrows.setdefault(name,[]).append(one)
        for region,one in result['curves'].items():curverows[region].append(one)
        cross.extend(result['cross']);denoms.extend(result['denominators'])
        heartbeat(run,'A',committed_pairs=index+1,total_pairs=100,phase='A0-A3 pair analysis')
    reproduced=summary(rows,['Spearman','AUSE']);frozen=read(SOURCE/'final_epe/summary.json')
    deviations={m:{k:abs(reproduced['scene_equal'][m][k]-frozen['ranking']['scene_equal'][m][k]) for k in ['AUSE','Spearman']} for m in METHODS}
    a0=dict(status='pass' if total['evaluable_parents']==149102 and total['valid_fine_children']==361254 and max(v for d in deviations.values() for v in d.values())<=1e-6 else 'failed',
        counts=dict(total),expected_evaluable_parents=149102,expected_valid_fine_children=361254,maximum_absolute_deviation=max(v for d in deviations.values() for v in d.values()),metric_deviations=deviations,reproduced=reproduced,
        frozen_summary_sha256=sha(SOURCE/'final_epe/summary.json'),payloads_sha256_verified=100,scope='Frozen final EPE payload aggregate exact-count and full-precision metric reproduction')
    save(out/'A0_reproduction.json',a0)
    if a0['status']!='pass':raise GateFailure('A0 failed; A and A4 blocked')
    # A1 primary pooled counts; scene-equal proportions are sensitivity summaries.
    cross_summary={}
    countkeys=['original_parents','with_fine_output','without_fine_output','without_valid_final_GT','denominator','epe_lt1','epe_lt2','epe_lt3','epe_gt5']
    for region in ['all','low_texture']:
        for label in [-1,0,1]:
            group=[x for x in cross if x['region']==region and x['coarse_label']==label]
            pooled={k:sum(x[k] for x in group) for k in countkeys};scenes={}
            for scene in sorted({x['scene'] for x in group}):
                a=[x for x in group if x['scene']==scene];v={k:sum(x[k] for x in a) for k in countkeys}
                v['fractions']={k:v[k]/v['denominator'] if v['denominator'] else None for k in ['epe_lt1','epe_lt2','epe_lt3','epe_gt5']};scenes[scene]=v
            pooled['fractions']={k:pooled[k]/pooled['denominator'] if pooled['denominator'] else None for k in ['epe_lt1','epe_lt2','epe_lt3','epe_gt5']}
            average={k:float(np.mean(v)) if (v:=[s['fractions'][k] for s in scenes.values() if s['fractions'][k] is not None]) else None for k in pooled['fractions']}
            cross_summary[region+'/label_'+str(label)]=dict(pooled=pooled,scene_equal_fractions=average,per_scene=scenes,denominator_definition='Parents with valid final median EPE under this coarse label')
    save(out/'A1_cross_table.json',cross_summary);csvout(out/'A1_pair_counts.csv',cross)
    csvout(out/'A1_pooled.csv',[dict(group=k,**v['pooled']) for k,v in cross_summary.items()])
    summ={name:summary(rs,['score_AUROC','AP']) for name,rs in allrows.items()}
    save(out/'A2_rows.json',allrows);save(out/'A2_same_parent_comparison.json',summ);metric_csv(out/'A2_same_parent_comparison.csv',summ)
    changes={}
    for region in ['all','low_texture']:
        changes[region+'/R2_minus_R1']=advantage_change(allrows[region+'/R1'],allrows[region+'/R2'])
        for threshold in [1,2,3,5]:
            changes[region+'/R3_lt'+str(threshold)+'_minus_R2']=advantage_change(allrows[region+'/R2'],allrows[region+'/R3_lt'+str(threshold)])
    save(out/'A2_advantage_changes.json',changes)
    curves={region:{scope:curve_summary(rs,METHODS,exclude) for scope,exclude in [('all_scenes',None),('exclude_0003','0003')]} for region,rs in curverows.items()}
    save(out/'A3_pair_curves.json',curverows);save(out/'A3_robust_curves.json',curves)
    flat=[]
    for region,scopes in curves.items():
        for scope,s in scopes.items():
            for m,c in s['methods'].items():
                for v in c['scene_equal']:flat.append(dict(region=region,sensitivity=scope,method=m,**v))
    csvout(out/'A3_robust_curves.csv',flat)
    # A3 pooled median is also supplied to distinguish it from mean pair medians.
    pools={};poolcount=0
    for r in recs:
        a=arrays_of(run,'A',committed(run,'A',r))
        for region,mask in [('all',a['valid_final_parent']),('low_texture',a['valid_final_parent']&a['low'])]:
            for m in METHODS:
                p=np.flatnonzero(mask);order=p[np.lexsort((a['raw_id'][p],-a[m][p]))]
                for i,f in enumerate(FRACTIONS):
                    e=a['parent_epe'][order[:int(np.ceil(len(order)*f))]]
                    for scope in ['all_scenes']+(['exclude_0003'] if r['scene']!='0003' else []):
                        pools.setdefault((region,scope,m,i),[]).append(e)
    pooled_curves=[]
    for (region,scope,m,i),values in pools.items():
        e=np.concatenate(values);n=len(e)
        pooled_curves.append(dict(region=region,sensitivity=scope,method=m,fraction=float(FRACTIONS[i]),parents=n,mean=float(e.mean()) if n else None,median=float(np.median(e)) if n else None,
            **{'precision_lt_'+str(t):float((e<t).mean()) if n else None for t in [1,2,3]}))
    csvout(out/'A3_pooled_curves.csv',pooled_curves)
    denominator_summary={}
    for region in ['all','low_texture']:
        a=[x for x in denoms if x['region']==region]
        denominator_summary[region]={k:sum(x[k] for x in a) for k in ['original_parents','with_fine_output','valid_final_GT','evaluable_pair']}
    save(out/'low_texture_denominators.json',dict(pooled=denominator_summary,per_pair=denoms))
    g2=cross_summary['all/label_0']['pooled']['fractions']['epe_lt2']
    result=dict(status='completed',A0=a0['status'],pairs=100,counts=dict(total),G2=dict(status='触发' if g2 is not None and g2>=.3 else '未触发' if g2 is not None else '不可判定',value=g2,threshold=.3,denominator=cross_summary['all/label_0']['pooled']['denominator']),
        R4='pending B',new_forwards=0,optimizer_updates=0,checkpoint_writes=0)
    save(out/'summary.json',result);return result
