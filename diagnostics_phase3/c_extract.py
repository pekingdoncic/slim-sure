"""Full-forward extraction, explicit trajectory units, and C0 acceptance."""
from common import *
from reliability_v1.runtime import resource_sample
import gc
def max_abs(a,b):
    return float(np.max(np.abs(np.asarray(a,float)-np.asarray(b,float)),initial=0))
def child_alignment(new,old,tolerance=1e-3):
    """Match repeated parents by parent ID and actual source coordinate, never by EPE."""
    n=len(new['child_parent_id'])
    if n!=len(old['child_parent_id']):return None,'child count differs'
    if np.array_equal(new['child_parent_id'],old['child_parent_id']) and np.allclose(new['final_source_input'],old['final_source_input'],atol=tolerance,rtol=0):
        return np.arange(n),'native order'
    order=np.full(n,-1,np.int64);used=np.zeros(n,bool)
    lookup={}
    for j,p in enumerate(old['child_parent_id']):lookup.setdefault(int(p),[]).append(j)
    for i,(p,coord) in enumerate(zip(new['child_parent_id'],new['final_source_input'])):
        choices=[j for j in lookup.get(int(p),[]) if not used[j] and np.allclose(coord,old['final_source_input'][j],atol=tolerance,rtol=0)]
        if len(choices)!=1:return None,'ambiguous or missing child source identity'
        order[i]=choices[0];used[choices[0]]=True
    return order,'parent ID + source coordinate'
def extract(runtime,stats,fit,record,run):
    import torch
    data,labels=runtime.forward(record,full=True);p=forward_scores(runtime,data,stats,fit)
    n=len(data['fine_parent_id']);T=int(runtime.model.refine_iters);radius=int(runtime.model.refine_lookup_radius)
    fine_scale=float(data['fine_scale'])
    offset=data['all_offset_1'].detach().float()
    if tuple(offset.shape)!=(T,n,2):raise GateFailure('C displacement step shape differs from T,N,2')
    scale=data['fine_scale']
    if 'scale1' in data:scale=scale*data['scale1'][data['b_idx_it']].float()
    px=(offset*radius*scale).cpu().numpy()
    initial=data['intermediate_coord_1'].detach().float().cpu().numpy()
    coords=np.concatenate([initial[None],initial[None]+np.cumsum(px,axis=0,dtype=np.float32)],axis=0)
    final=data['fine_coord_1'].detach().float().cpu().numpy()
    difference=max_abs(np.sum(px,axis=0,dtype=np.float32),final-initial)
    finite(coords,'trajectory coordinates');finite(px,'pixel displacements')
    if difference>1e-2:
        save(Path(run)/'C/C0_failure.json',dict(pair_id=record['pair_id'],reason='displacement reconstruction exceeds 1e-2 px',max_deviation=difference))
        raise GateFailure('C trajectory reconstruction failed')
    error,valid,gt=final_source_errors(data);valid=np.asarray(valid,bool)
    truth=gt['points1_gt_input'];errors=np.linalg.norm(coords.astype(np.float64)-truth[None],axis=2)
    finite(errors[:,valid],'all-step valid EPE');finite(error[valid],'native final EPE')
    sim=data['sim_matrix_f'].float()
    conf=sim.softmax(dim=1)*sim.softmax(dim=2)
    fine=conf[data['m_idx_it'],data['i_idx_it'],data['j_idx_it']].cpu().numpy()
    norms=np.linalg.norm(px.astype(np.float64),axis=2)
    cp=data['fine_parent_id'].cpu().numpy();pos=positions(p['raw_id'],cp)
    current=dict(child_parent_id=cp,child_epe=error,child_valid=valid,final_source_input=data['fine_coord_0'].cpu().numpy(),final_target_input=final)
    old,ref=cache(record);alignment,reason=child_alignment(current,old)
    compare=dict(status='not_comparable',alignment=reason,child_count_current=n,child_count_phase2=len(old['child_parent_id']))
    if alignment is not None:
        gt_equal=np.array_equal(valid,old['child_valid'][alignment].astype(bool))
        deltas=dict(source=max_abs(current['final_source_input'],old['final_source_input'][alignment]),
            target=max_abs(final,old['final_target_input'][alignment]),
            native_final_EPE=max_abs(error[valid],old['child_epe'][alignment][valid]),
            reconstructed_final_EPE=max_abs(errors[T,valid],old['child_epe'][alignment][valid]))
        parent_equal=np.array_equal(cp,old['child_parent_id'][alignment])
        compare.update(status='pass' if gt_equal and parent_equal and max(deltas.values())<=1e-3 else 'not_comparable',
            parent_IDs_equal=parent_equal,valid_GT_mask_equal=gt_equal,max_absolute_deviations=deltas)
    b0=read(Path(run)/'B/B0_reproduction.json') if (Path(run)/'B/B0_reproduction.json').exists() else {}
    linked=dict(status='不可判定',reason='B0 unavailable or failed')
    b_labels=np.full(len(p['raw_id']),-1,np.int8);b_residual=np.full((len(p['raw_id']),2),np.nan);b_center=b_labels.copy()
    if b0.get('status')=='pass':
        receipt=committed(run,'B',record)
        if receipt:
            b=arrays_of(run,'B',receipt)
            try:
                idx=positions(b['raw_id'],p['raw_id'])
                coord0=data['coarse_coord_0'].cpu().numpy();coord1=data['coarse_coord_1'].cpu().numpy()
                delta=max(max_abs(coord0,b['coarse0'][idx]),max_abs(coord1,b['coarse1'][idx]))
                equal=len(p['raw_id'])==len(b['raw_id']) and np.array_equal(np.sort(p['raw_id']),np.sort(b['raw_id'])) and delta<=1e-4
                if equal:
                    b_labels=b['labels'][idx];b_center=b['center_labels'][idx];b_residual=b['corner_residual'][idx]
                    linked=dict(status='pass',pair_id=record['pair_id'],direction=record.get('direction','0to1'),parent_IDs_equal=True,maximum_coarse_coordinate_deviation=delta)
                else:linked=dict(status='不可判定',reason='B/C coarse parent ID or coordinate differs',max_coordinate_deviation=delta)
            except (GateFailure,KeyError) as exc:linked=dict(status='不可判定',reason=repr(exc))
    signals=dict(fine_confidence=fine,neg_first_step=-norms[0],neg_last_step=-norms[-1],neg_path_length=-norms.sum(axis=0),
        mconf=p['mconf'][pos],head=p['head'][pos])
    for name,score in signals.items():finite(score,name)
    arrays=dict(**p,labels=np.asarray(labels['labels'],np.int8),child_parent_id=cp,child_valid=valid,child_parent_position=pos,
        child_low=p['low'][pos].astype(bool),child_labels=np.asarray(labels['labels'],np.int8)[pos],
        source_input=current['final_source_input'],target_input=final,intermediate_input=initial,gt_target_input=truth,
        trajectory_coords=coords,pixel_updates=px,EPE=errors,native_final_EPE=error,child_lr=p['lr'][pos],
        B_labels=b_labels,B_center_labels=b_center,B_corner_residual=b_residual,
        **{'signal_'+k:v for k,v in signals.items()})
    result=dict(**base(record),C0=dict(reconstruction=dict(status='pass',max_absolute_deviation_px=difference,tolerance_px=1e-2),
        phase2_child_reproduction=compare,B_parent_association=linked),actual=dict(T=T,refine_lookup_radius=radius,fine_scale=fine_scale,
        scale1_present='scale1' in data,unit_formula='float32(all_offset_1[t-1]) * REFINE_LOOKUP_RADIUS * fine_scale * optional scale1[b_idx_it]',
        mathematical_step_zero='intermediate_coord_1; no update at t=0'),resources=resource_sample(),children=n,valid_children=int(valid.sum()))
    del data,labels,sim,conf,offset;gc.collect()
    return arrays,result,ref
def optional_two_step(runtime,stats,fit,run,records):
    import torch
    results=[]
    for r in sorted(records,key=lambda r:r['pair_id'])[:3]:
        receipt=committed(run,'C',r);a=arrays_of(run,'C',receipt);original=int(runtime.model.refine_iters)
        try:
            runtime.model.refine_iters=2
            data,_=runtime.forward(r,full=True)
            current=dict(child_parent_id=data['fine_parent_id'].cpu().numpy(),final_source_input=data['fine_coord_0'].cpu().numpy())
            stored=dict(child_parent_id=a['child_parent_id'],final_source_input=a['source_input'])
            align,reason=child_alignment(current,stored)
            difference=max_abs(data['fine_coord_1'].cpu().numpy(),a['trajectory_coords'][2,align]) if align is not None else None
            results.append(dict(**base(r),status='pass' if difference is not None and difference<=1e-2 else 'failed',
                alignment=reason,max_absolute_deviation_px=difference,tolerance_px=1e-2))
            del data
        finally:runtime.model.refine_iters=original;check_runtime(runtime)
    save(Path(run)/'C/optional_two_step_equivalence.json',dict(status='pass' if all(r['status']=='pass' for r in results) else 'failed',
        pairs=results,new_forward_calls=len(results),scope='Only two-step reconstruction equivalence; no early-stop runtime/memory saving claim'))
