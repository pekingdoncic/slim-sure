"""Real frozen-model threshold scans and fixed-grid GT funnel."""
from shared import *
from reliability_data import project_visible
from reliability_v1.comparison import final_source_errors
def aggregate_children(n,m,e,valid,fine):
    counts=np.bincount(m,minlength=n);vc=np.bincount(m[valid],minlength=n)
    med=np.full(n,np.nan);fm=np.full(n,np.nan)
    for idx,first,num in zip(*np.unique(m[valid],return_index=True,return_counts=True)):
        med[idx]=np.median(e[valid][first:first+num])
    for idx,first,num in zip(*np.unique(m,return_index=True,return_counts=True)):
        fm[idx]=np.median(fine[first:first+num])
    return counts,vc,med,fm
def extract(rt,d,r,setting):
    import torch
    raw=d['coarse_match_id'].cpu().numpy();i=d['i_idx_c'].cpu().numpy();j=d['j_idx_c'].cpu().numpy()
    cp=d['fine_parent_id'].cpu().numpy();m=d['m_idx_it'].cpu().numpy()
    p0=d['coarse_coord_0'].cpu().numpy();p1=d['coarse_coord_1'].cpu().numpy()
    source=d['fine_coord_0'].cpu().numpy();target=d['fine_coord_1'].cpu().numpy()
    e,valid,gt=final_source_errors(d);valid=valid.astype(bool)
    sim=d['sim_matrix_f'].float();fc=sim.softmax(1)*sim.softmax(2)
    fine=fc[d['m_idx_it'],d['i_idx_it'],d['j_idx_it']].cpu().numpy();del sim,fc
    finite(fine,'fine confidence');finite(e[valid],'valid final EPE')
    conf=d['coarse_confidence'].float().cpu().numpy()
    grid=[low_grids(d,s,rt.model.texture_threshold) for s in [0,1]]
    norms=[]
    for side in [0,1]:
        f=d['feat'+str(side)+'_c'];norm=(f.permute(0,2,3,1).reshape(1,-1,f.shape[1])/f.shape[1]**.5).norm(dim=-1)[0].cpu().numpy()
        norm[~grid[side]['legal']]=0;norms.append(norm)
    candidates=[d['candidate_ids'+str(s)][0].cpu().numpy() for s in [0,1]]
    if setting=='S4':
        for s in [0,1]:
            if not np.array_equal(candidates[s],np.flatnonzero(grid[s]['legal'])):raise StopRoute('S4 legal candidate set mismatch')
    hw0=tuple(d['hw0_c']);hw1=tuple(d['hw1_c']);ids=np.arange(np.prod(hw0));xy=np.stack([ids%hw0[1],ids//hw0[1]],1)*8+4
    projected=project_visible(xy,d);gtxy=projected['points1_gt_input'];visible=projected['valid_gt'].astype(bool)
    correct=np.full(len(ids),-1,np.int64);finitegt=np.isfinite(gtxy).all(1)
    cell=np.zeros((len(ids),2),np.int64);cell[finitegt]=np.floor(gtxy[finitegt]/8).astype(np.int64)
    bounds=finitegt&(cell[:,0]>=0)&(cell[:,1]>=0)&(cell[:,0]<hw1[1])&(cell[:,1]<hw1[0])
    correct[bounds]=cell[bounds,1]*hw1[1]+cell[bounds,0]
    targetlegal=np.zeros(len(ids),bool);targetlegal[bounds]=grid[1]['legal'][correct[bounds]]
    denom=grid[0]['legal']&visible&targetlegal
    sp=np.isin(ids,candidates[0]);tp=np.isin(correct,candidates[1])&bounds
    stages=dict(source_pass=sp,target_pass=tp,both_pass=sp&tp)
    correct_parent=denom[i]&(j==correct[i])
    pc,vc,pe,pfine=aggregate_children(len(raw),m,e,valid,fine)
    labels=np.full(len(raw),-1,np.int8);labels[denom[i]]=(j[denom[i]]==correct[i[denom[i]]]).astype(np.int8)
    # Correct source-grid target uses half-open coarse cells; each source votes once.
    for name,parentmask in [('correct_parent',correct_parent),('correct_parent_fine',correct_parent&(pc>0))]:
        a=np.zeros(len(ids),bool);a[i[parentmask]]=True;stages[name]=a
    correctchild=correct_parent[m]&valid&(e<2)
    anygood=valid&(e<2)
    for name,srcids in [('correct_parent_good',i[m[correctchild]]),('any_parent',i),('any_good',i[m[anygood]]),('has_evaluable_child',i[m[valid]])]:
        a=np.zeros(len(ids),bool);a[srcids]=True;stages[name]=a
    old=get_unit(rt.task_run,'S0',r,True) if setting!='S0' else None
    added=np.ones(len(raw),bool) if old is None else ~np.isin(raw,old['raw_id'])
    if setting=='S0':added[:]=False
    parentlow=grid[0]['low'][i];childlow=parentlow[m];epi=epipolar(d,source,target)
    finite(epi,'all predicted epipolar errors')
    arrays=dict(raw_id=raw,source_grid=i,target_grid=j,coarse0=p0,coarse1=p1,cp=cp,parent_pos=m,
        source=source,target=target,valid=valid,epe=e,gt=gt['points1_gt_input'],gt_reasons=gt['invalid_reason'],
        mconf=conf,fine=fine,parent_fine=pfine,parent_S_m=conf.astype(float)*pfine,low=parentlow,child_low=childlow,
        child_counts=pc,valid_child_counts=vc,parent_epe=pe,center_labels=labels,added=added,epipolar=epi,
        grid_ids=ids,grid_center_gt=gtxy,grid_gt_reasons=projected['invalid_reason'],grid_denominator=denom,grid_correct_target=correct,
        scale0=np.asarray(d['_gt']['plan0']['scale_input_to_orig_xy']),scale1=np.asarray(d['_gt']['plan1']['scale_input_to_orig_xy']),
        K0=d['_gt']['K0'],K1=d['_gt']['K1'],T=d['_gt']['T_0to1'],
        hw0=np.asarray(d['_gt']['plan0']['resized_hw']),hw1=np.asarray(d['_gt']['plan1']['resized_hw']),
        **{'grid_'+k:v for k,v in stages.items()})
    for s in [0,1]:
        arrays.update({f'candidate{s}':candidates[s],f'grid_legal{s}':grid[s]['legal'],f'grid_low{s}':grid[s]['low'],
            f'grid_texture{s}':grid[s]['texture'],f'grid_norm{s}':norms[s],f'grid_relative_norm{s}':norms[s]/max(float(norms[s].max()),1e-30)})
    funnel={}
    order=['source_pass','both_pass','correct_parent','correct_parent_fine','correct_parent_good']
    for region,mask in [('low_texture',grid[0]['low']),('other_texture',~grid[0]['low'])]:
        mask=mask&denom;den=int(mask.sum());entry=dict(denominator=den)
        for name,a in stages.items():entry[name]=dict(numerator=int((a&mask).sum()),denominator=den,ratio=ratio(int((a&mask).sum()),den))
        prev=den
        for name in order:
            num=entry[name]['numerator'];entry[name]['previous_denominator']=prev;entry[name]['relative_previous']=ratio(num,prev);prev=num
        entry['no_evaluable_child']=int((mask&~stages['has_evaluable_child']).sum())
        entry['norm_distribution']=distributions(norms[0][mask]);funnel[region]=entry
    quality={}
    for name,mask in [('all',np.ones(len(raw),bool)),('low_texture',parentlow),('added',added),('added_low_texture',added&parentlow)]:
        pm=mask&np.isfinite(pe);cm=mask[m];ce=cm&valid;lm=mask&(labels>=0)
        quality[name]=dict(parents=int(mask.sum()),center_valid=int(lm.sum()),center_positive=int((labels[lm]==1).sum()),center_positive_rate=ratio(int((labels[lm]==1).sum()),int(lm.sum())),
            no_fine=int((mask&(pc==0)).sum()),fine_no_GT=int((mask&(pc>0)&(vc==0)).sum()),GT_evaluable_parent_rate=ratio(int(pm.sum()),int(mask.sum())),
            parent=error_metrics(pe[pm]),child=error_metrics(e[ce]),children=int(cm.sum()),epipolar_correct=int((epi[cm]<1e-4).sum()),epipolar_precision=ratio(int((epi[cm]<1e-4).sum()),int(cm.sum())))
    auc={}
    from reliability_v1.comparison import ranks
    for region,mask in [('added',added),('added_low_texture',added&parentlow)]:
        validparent=mask&np.isfinite(pe)
        auc[region]=dict(evaluable=int(validparent.sum()),methods={name:ranks((pe[validparent]<2).astype(np.int8),score[validparent]) for name,score in [('mconf',conf),('fine',pfine),('S_m',arrays['parent_S_m'])]})
    eligible=np.flatnonzero(added&parentlow&(pc>0));ranking=np.lexsort((raw[eligible],-arrays['parent_S_m'][eligible]));chosen=eligible[ranking[:math.ceil(.8*len(eligible))]]
    sel=np.zeros(len(raw),bool);sel[chosen]=True;pv=added&parentlow&np.isfinite(pe);good=pv&(pe<2);bad=pv&~(pe<2)
    filter80=dict(eligible=len(eligible),selected=len(chosen),good_before=int(good.sum()),good_kept=int((good&sel).sum()),bad_before=int(bad.sum()),bad_removed=int((bad&~sel).sum()),
        good_retention=ratio(int((good&sel).sum()),int(good.sum())),bad_removal=ratio(int((bad&~sel).sum()),int(bad.sum())),selected_GT_evaluable_rate=ratio(int((sel&pv).sum()),len(chosen)))
    arrays['filter80_selected']=sel
    changes={}
    if old is not None:
        before=old['grid_any_good']&denom;after=stages['any_good']&denom
        changes=dict(added_parents=int(added.sum()),lost_parents=int((~np.isin(old['raw_id'],raw)).sum()),added_from_previously_unmatched_source=int((added&~np.isin(i,old['source_grid'])).sum()),
            added_different_target_for_existing_source=int((added&np.isin(i,old['source_grid'])).sum()))
        for name,mask in [('all',denom),('low_texture',denom&grid[0]['low']),('other_texture',denom&~grid[0]['low'])]:
            changes[name]=dict(new=int((after&~before&mask).sum()),lost=int((before&~after&mask).sum()),retained=int((after&before&mask).sum()),net=int((after&mask).sum()-(before&mask).sum()))
        if not np.array_equal(denom,old['grid_denominator']) or not np.array_equal(correct,old['grid_correct_target']):raise IdentityFailure('Cross-setting denominator differs')
    budget=dict(N0=len(candidates[0]),N1=len(candidates[1]),product=len(candidates[0])*len(candidates[1]))
    if old is not None:budget.update(delta0=len(candidates[0])-len(old['candidate0']),delta1=len(candidates[1])-len(old['candidate1']),delta_product=budget['product']-len(old['candidate0'])*len(old['candidate1']))
    audit={}
    if setting=='S0':
        c=historical('C',r);b=historical('B',r)
        pairs=[('raw_id',raw,c['raw_id']),('cp',cp,c['child_parent_id']),('source',source,c['source_input']),('target',target,c['target_input']),('valid',valid,c['child_valid'])]
        audit={key:dict(equal=np.array_equal(x,y),actual_sha=array_sha(x),reference_sha=array_sha(y)) for key,x,y in pairs}
        if not all(v['equal'] for v in audit.values()):raise StopRoute('S0 coordinate/parent/row-order reproduction failed: '+r['pair_id'])
    counts=dict(parents=len(raw),children=len(cp),valid_children=int(valid.sum()),low_parents=int(parentlow.sum()),low_children=int(childlow.sum()),
        legal_source_grids=int(grid[0]['legal'].sum()),center_invisible=int((grid[0]['legal']&~visible).sum()),visible_unmapped_target=int((grid[0]['legal']&visible&~targetlegal).sum()),
        coarse_prethreshold_topk=min(rt.model.max_coarse_matches,min(budget['N0'],budget['N1'])),
        coarse_budget_cap_hit=min(budget['N0'],budget['N1'])>=rt.model.max_coarse_matches,
        fine_per_parent_topk=min(int(rt.model.max_fine_matches/len(raw)),int((d['coarse_scale']/d['fine_scale'])**2)) if len(raw) else 0)
    counts['fine_prethreshold_topk_total']=counts['fine_per_parent_topk']*len(raw)
    counts['fine_budget_cap_limited']=bool(len(raw) and rt.model.max_fine_matches/len(raw)<(d['coarse_scale']/d['fine_scale'])**2)
    return arrays,dict(**base(r),setting=setting,budget=budget,counts=counts,funnel=funnel,quality=quality,added_AUROC=auc,
        filter80=filter80,changes=changes,reproduction=audit,score_quantiles={g:{k:distributions(v[mask&np.isfinite(v)]) for k,v in [('mconf',conf),('S_m',arrays['parent_S_m'])]} for g,mask in [('added',added),('S0_existing',~added)]})
def run_setting(run,setting,start_limit=None):
    import torch
    run=Path(run);out=run/setting;out.mkdir(parents=True,exist_ok=True);rt=None;original=None;beg=time.monotonic()
    recs=records('epe')
    try:
        rt=NativeRuntime(out/'runtime');rt.task_run=run;original=setup_threshold(rt.model,setting,out)
        for index,r in enumerate(recs):
            if receipt(run,setting,r):continue
            beat(run,setting,phase='forward',pair_index=index,pair_id=r['pair_id'],committed=index)
            inputs=rt.load(r);d=dict(inputs)
            rt.forward(d);torch.cuda.synchronize() # full warmup on the same input
            times=[];segments=[]
            for repeat in range(3):
                del d;d=dict(inputs)
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter()
                rt.forward(d);torch.cuda.synchronize();elapsed=time.perf_counter()-t
                times.append(dict(seconds=elapsed,allocated_peak=torch.cuda.max_memory_allocated(),reserved_peak=torch.cuda.max_memory_reserved()))
                segments.append({k:float(d.get(k,0)) for k in ['feat_extract_time','upsample_time','coarse_match_time','fine_match_time','refine_time']})
            arrays,result=extract(rt,d,r,setting)
            result.update(timing=dict(repeats=times,median_seconds=float(np.median([x['seconds'] for x in times])),segments=segments),
                resources=resource_sample(),elapsed_seconds=time.monotonic()-beg)
            commit(run,setting,r,result,arrays)
            if index==19:
                median=float(np.mean([get_unit(run,setting,z)['timing']['median_seconds'] for z in recs[:20]]))
                observed=(time.monotonic()-beg)/20
                # All T unit costs measured; P5 first50 will replace this provisional upper bound.
                eta=dict(time_beijing=now(),measured_units=20,seconds_per_setting_pair=observed,forward_seconds=median,
                    remaining_T_units=600-(int(setting[1])*100+20),T_remaining_seconds=(600-(int(setting[1])*100+20))*observed,
                    P5_provisional_training_seconds=55200*median,P5_validation_final_seconds=3200*median,
                    note='P5 conservative full-forward proxy before route-specific first50; geometry/implementation overhead recorded separately')
                eta['total_remaining_seconds']=eta['T_remaining_seconds']+eta['P5_provisional_training_seconds']+eta['P5_validation_final_seconds']
                eta['exceeds_16_hours']=eta['total_remaining_seconds']>16*3600
                save(out/'first20_eta.json',eta);print(json.dumps(dict(event='first20',**eta)),flush=True)
                if eta['exceeds_16_hours'] and not (run/'degradation.json').exists():
                    save(run/'degradation.json',dict(time_beijing=now(),reason='measured provisional total >16 h',skipped=['T S3/S4 pose','T S1','Phase5 pose','T segmented timing','A efficiency timing'],required_main_samples_unchanged=True,ETA=eta))
            del d,inputs,arrays,result;gc.collect()
            beat(run,setting,phase='committed',committed=index+1,total=100,resources=resource_sample())
            if start_limit and index+1>=start_limit:break
        rt.check_frozen();completed=sum(receipt(run,setting,r) is not None for r in recs)
        if setting=='S0' and completed==100:
            counts={k:sum(get_unit(run,setting,r)['counts'][k] for r in recs) for k in ['parents','children','valid_children']}
            save(out/'reproduction.json',dict(status='pass' if counts==dict(parents=194383,children=484473,valid_children=361254) else 'failed',counts=counts,coordinates_and_order='exact hash per pair'))
            if counts!=dict(parents=194383,children=484473,valid_children=361254):raise StopRoute('S0 total counts differ')
        save(out/'worker_status.json',dict(status='completed' if completed==100 else 'partial',completed=completed,seconds=time.monotonic()-beg,resources=resource_sample(),frozen=True))
    finally:
        if rt is not None:
            if original is not None:rt.model._coarse_match_norm_clue=original
            rt.model.texture_budget_enabled=False
            save(out/'cleanup.json',dict(time_beijing=now(),default_norm_restored=True,texture_budget=False,check_frozen=rt.check_frozen()))
            del rt;gc.collect();torch.cuda.empty_cache()
