"""C statistical tables, all paired on explicit common identities."""
from common import *
SIGNALS=('fine_confidence','neg_first_step','neg_last_step','neg_path_length','mconf','head')
def parent_bundle(a,mask,errors,scores):
    agg=parent_errors(a['raw_id'],a['child_parent_id'],errors,mask)
    ids=agg['parent_ids'];pscores={}
    for method,score in scores.items():
        pscores[method]=np.asarray([np.median(score[(a['child_parent_id']==pid)&mask]) for pid in ids],float)
    return ids,agg['error'],pscores
def tied_child_ids(cp):
    order=np.lexsort((np.arange(len(cp)),cp));out=np.empty(len(cp),np.int64);out[order]=np.arange(len(cp));return out
def epe_stats(e):
    n=len(e)
    return dict(count=n,mean=float(np.mean(e)) if n else None,median=float(np.median(e)) if n else None,
        **{'precision_lt_'+str(t):float(np.mean(e<t)) if n else None for t in [1,2,3]})
def retention_summary(rows,methods):
    scenes=sorted({r['scene'] for r in rows});output={}
    for m in methods:
        per_scene={}
        for s in scenes:
            v=[r['methods'][m]['curve'] for r in rows if r['scene']==s and r['methods'][m]['curve'] is not None]
            per_scene[s]=np.mean(v,axis=0).tolist() if v else None
        values=[v for v in per_scene.values() if v is not None]
        output[m]=dict(fractions=FRACTIONS.tolist(),scene_equal=np.mean(values,axis=0).tolist() if values else None,per_scene=per_scene,
            defined_pairs=sum(r['methods'][m]['curve'] is not None for r in rows))
    return output
def analyze(run,recs):
    run=Path(run);out=run/'C';stages={};quality={};qualities_class={};routing=[];step_rows=[];outcome_rows=[];store={};allchecks=[]
    for number,r in enumerate(recs):
        receipt=committed(run,'C',r);result=result_of(run,'C',receipt);a=arrays_of(run,'C',receipt);allchecks.append(result)
        valid=a['child_valid'].astype(bool);cp=a['child_parent_id'];cid=tied_child_ids(cp);T=a['EPE'].shape[0]-1
        childscore=dict(head=a['signal_head'],lr=a['child_lr'],mconf=a['signal_mconf'])
        B_ok=result['C0']['B_parent_association']['status']=='pass'
        joint=valid&(a['B_labels'][a['child_parent_position']]>=0) if B_ok else None
        for scope,mask in [('all_valid_C',valid)]+([('common_BC',joint)] if B_ok else []):
            ids,ep,ps=parent_bundle(a,mask,a['EPE'][0],childscore)
            parentpos=positions(a['raw_id'],ids)
            if scope=='common_BC':
                residual=a['B_corner_residual'][parentpos];finite(residual,'common coarse residual')
                for name,e in [('coarse_linf',np.max(np.abs(residual),axis=1)),('coarse_l2',np.linalg.norm(residual,axis=1))]:
                    stages.setdefault(scope+'/parent/'+name,[]).append(errorrow(r,e,{m:a[m][parentpos] for m in METHODS},ids))
            for t in range(T+1):
                ids,ep,ps=parent_bundle(a,mask,a['EPE'][t],childscore)
                stages.setdefault(scope+'/parent/t'+str(t),[]).append(errorrow(r,ep,ps,ids))
                stages.setdefault(scope+'/child/t'+str(t),[]).append(errorrow(r,a['EPE'][t,mask],{m:v[mask] for m,v in childscore.items()},cid[mask]))
        regions=[('all',valid),('coarse_0',valid&(a['child_labels']==0)),('coarse_1',valid&(a['child_labels']==1)),('coarse_ignore',valid&(a['child_labels']<0)),('low_texture',valid&a['child_low'].astype(bool))]
        for group,mask in regions:
            for t in range(T+1):
                e=a['EPE'][t,mask];row=dict(**base(r),group=group,step=t,**epe_stats(e))
                if t:row['mean_error_drop_from_previous']=float(np.mean(a['EPE'][t-1,mask]-e)) if len(e) else None
                step_rows.append(row);store.setdefault((group,t,r['scene']),[]).append(e)
            initial=a['EPE'][0,mask];final=a['EPE'][T,mask];gain=a['EPE'][1,mask]-final
            move=np.linalg.norm(a['trajectory_coords'][T,mask]-a['trajectory_coords'][1,mask],axis=1)
            outcome_rows.append(dict(**base(r),group=group,count=int(mask.sum()),initial_ge2=int((initial>=2).sum()),rescued=int(((initial>=2)&(final<2)).sum()),
                initial_lt2=int((initial<2).sum()),lost=int(((initial<2)&(final>=2)).sum()),
                further_gain=int((gain>.1).sum()),stable_error=int((np.abs(gain)<=.1).sum()),worsened=int((gain<-.1).sum()),
                coordinate_move=distributions(move),gain_distribution=distributions(gain)))
            store.setdefault((group,'move_1toT',r['scene']),[]).append(move);store.setdefault((group,'gain_1toT',r['scene']),[]).append(gain)
        for region,mask in [('all',valid),('low_texture',valid&a['child_low'].astype(bool))]:
            signals={m:a['signal_'+m] for m in SIGNALS};error=a['EPE'][T]
            childr=errorrow(r,error[mask],{m:s[mask] for m,s in signals.items()},cid[mask])
            ids,perr,ps=parent_bundle(a,mask,error,signals)
            parentr=errorrow(r,perr,ps,ids)
            quality.setdefault(region+'/child',[]).append(childr);quality.setdefault(region+'/parent',[]).append(parentr)
            for threshold in [1,2,3]:
                qualities_class.setdefault(region+'/child/lt'+str(threshold),[]).append(classify(r,(error[mask]<threshold).astype(np.int8),{m:s[mask] for m,s in signals.items()}))
                qualities_class.setdefault(region+'/parent/lt'+str(threshold),[]).append(classify(r,(perr<threshold).astype(np.int8),ps))
        gain=a['EPE'][1]-a['EPE'][T];benefit=(gain[valid]>.1).astype(np.int8)
        routing.append(classify(r,benefit,dict(first_step_magnitude=-a['signal_neg_first_step'][valid],low_fine_confidence=-a['signal_fine_confidence'][valid],
            low_mconf=-a['signal_mconf'][valid],low_head=-a['signal_head'][valid])))
        heartbeat(run,'C',phase='CPU statistical tables after frozen GPU check',analyzed_pairs=number+1,total_pairs=100)
    # Common pair comparisons never subtract independently averaged stages.
    stage_summ={k:summary(v,['Spearman']) for k,v in stages.items()}
    save(out/'stage_predictive_rows.json',stages);save(out/'stage_predictive_power.json',stage_summ);metric_csv(out/'stage_predictive_power.csv',stage_summ)
    changes={}
    for scope in ['all_valid_C','common_BC']:
        for level in ['parent','child']:
            for before,after in [('t0','t1'),('t1','t'+str(T))]+[('t'+str(t),'t'+str(t+1)) for t in range(1,T)]:
                k0=scope+'/'+level+'/'+before;k1=scope+'/'+level+'/'+after
                if k0 in stages and k1 in stages:changes[k1+'_minus_'+before]=advantage_change(stages[k0],stages[k1],'Spearman')
    for coarse in ['coarse_linf','coarse_l2']:
        k0='common_BC/parent/'+coarse;k1='common_BC/parent/t0'
        if k0 in stages and k1 in stages:changes[k1+'_minus_'+coarse]=advantage_change(stages[k0],stages[k1],'Spearman')
    save(out/'stage_advantage_changes.json',changes)
    steps=[];dist=[]
    groups=sorted({x[0] for x in store});scenes=sorted({r['scene'] for r in recs})
    for group in groups:
        for scene in ['ALL_POOLED']+scenes:
            chosen=scenes if scene=='ALL_POOLED' else [scene]
            for t in range(T+1):
                arrays=[e for s in chosen for e in store.get((group,t,s),[])];e=np.concatenate(arrays) if arrays else np.array([])
                v=dict(group=group,scene=scene,step=t,**epe_stats(e))
                if t:
                    previous=np.concatenate([e for s in chosen for e in store.get((group,t-1,s),[])])
                    v['mean_error_drop_from_previous']=float(np.mean(previous-e)) if len(e) else None
                    v['median_error_drop_from_previous']=float(np.median(previous-e)) if len(e) else None
                steps.append(v)
            for kind in ['move_1toT','gain_1toT']:
                arrays=[e for s in chosen for e in store.get((group,kind,s),[])];e=np.concatenate(arrays) if arrays else np.array([])
                dist.append(dict(group=group,scene=scene,kind=kind,distribution=distributions(e)))
    save(out/'step_errors.json',dict(pair_rows=step_rows,pooled_and_per_scene=steps));csvout(out/'step_errors.csv',steps)
    save(out/'gain_and_coordinate_distributions.json',dist)
    outcomes=[];keys=['count','initial_ge2','rescued','initial_lt2','lost','further_gain','stable_error','worsened']
    for group in groups:
        for scene in ['ALL_POOLED']+scenes:
            rs=[r for r in outcome_rows if r['group']==group and (scene=='ALL_POOLED' or r['scene']==scene)]
            values={k:sum(r[k] for r in rs) for k in keys}
            values.update(rescue_fraction=values['rescued']/values['initial_ge2'] if values['initial_ge2'] else None,loss_fraction=values['lost']/values['initial_lt2'] if values['initial_lt2'] else None,
                gain_fraction=values['further_gain']/values['count'] if values['count'] else None,stable_fraction=values['stable_error']/values['count'] if values['count'] else None,worse_fraction=values['worsened']/values['count'] if values['count'] else None)
            outcomes.append(dict(group=group,scene=scene,**values))
    save(out/'rescues_losses_gain_classes.json',dict(pooled_and_per_scene=outcomes,per_pair=outcome_rows));csvout(out/'rescues_losses_gain_classes.csv',outcomes)
    qsum={k:summary(v,['Spearman','AUSE'],SIGNALS) for k,v in quality.items()}
    qcls={k:summary(v,['score_AUROC','AP'],SIGNALS) for k,v in qualities_class.items()}
    save(out/'quality_signal_rows.json',quality);save(out/'quality_signals.json',qsum);metric_csv(out/'quality_signals.csv',qsum)
    save(out/'quality_classification.json',qcls);metric_csv(out/'quality_classification.csv',qcls)
    save(out/'quality_retention_curves.json',{k:retention_summary(v,SIGNALS) for k,v in quality.items()})
    rm=('first_step_magnitude','low_fine_confidence','low_mconf','low_head')
    route=summary(routing,['score_AUROC','AP'],rm,comparisons=[(m,'low_mconf') for m in rm if m!='low_mconf'])
    save(out/'routing.json',dict(summary=route,rows=routing,positive='EPE_1 - EPE_T > 0.1 input pixels',score_directions='first step magnitude positive; fine confidence, mconf and head scores negated'))
    all_linked=all(x['C0']['B_parent_association']['status']=='pass' for x in allchecks)
    c0=dict(status='pass',reconstruction_max_deviation_px=max(x['C0']['reconstruction']['max_absolute_deviation_px'] for x in allchecks),
        phase2_comparable_pairs=sum(x['C0']['phase2_child_reproduction']['status']=='pass' for x in allchecks),
        B_associated_pairs=sum(x['C0']['B_parent_association']['status']=='pass' for x in allchecks),pairs=allchecks,
        actual_T=sorted({x['actual']['T'] for x in allchecks}),actual_fine_scale=sorted({x['actual']['fine_scale'] for x in allchecks}),
        actual_lookup_radius=sorted({x['actual']['refine_lookup_radius'] for x in allchecks}))
    save(out/'C0_acceptance.json',c0)
    g4=qsum['all/child']['paired_differences']['fine_confidence-mconf']['Spearman'];ci=g4['ci95']
    result=dict(status='completed',pairs=100,all_BC_parents_associated=all_linked,C0=c0.copy(),
        G4=dict(status='触发' if ci and ci[1]<0 else '未触发' if ci else '不可判定',**g4),
        cross_stage_conclusion_status='可判定' if all_linked else '不可判定',
        parent_quality_signal_aggregation='Median signal over the same valid children used for median parent EPE',
        stage_masks='all_valid_C: fixed current-forward child GT mask at all steps; common_BC: additionally valid B corner parent GT; each table explicitly names its mask',
        new_scores_fit=False,optimizer_updates=0,checkpoint_writes=0)
    result['C0'].pop('pairs')
    save(out/'summary.json',result);return result
