"""Full native forward evaluation; all predicted points are retained before GT checks."""
from .common import *
from .native import inputs,supervision,predict,trajectory
from reliability_v1.comparison import final_source_errors,spearman
import torch
def parent_medians(n,m,e,valid,fine):
    counts=np.bincount(m,minlength=n);vcounts=np.bincount(m[valid],minlength=n)
    pe=np.full(n,np.nan);pf=np.full(n,np.nan);fine_choice=np.full(n,-1,np.int64)
    ev=e[valid]
    for idx,first,num in zip(*np.unique(m[valid],return_index=True,return_counts=True)):pe[idx]=np.median(ev[first:first+num])
    for idx,first,num in zip(*np.unique(m,return_index=True,return_counts=True)):
        pf[idx]=np.median(fine[first:first+num]);fine_choice[idx]=first+int(np.argmax(fine[first:first+num]))
    return counts,vcounts,pe,pf,fine_choice
def actual_one_step(rt,d):
    """Execute the native recurrence once on the same full-forward fine inputs."""
    copy=dict(d);count=[0];old=rt.model.refine_iters
    def called(*args):count[0]+=1
    hook=rt.model.recurrent_refine_unit.register_forward_hook(called)
    try:
        rt.model.refine_iters=1
        with torch.no_grad(),torch.cuda.amp.autocast(enabled=rt.config['amp']):
            rt.model._refinement(copy)
    finally:
        rt.model.refine_iters=old;hook.remove()
    if count[0]!=1:raise AssertionError('Native one-step did not execute exactly once')
    if not torch.equal(copy['fine_coord_0'],d['fine_coord_0']):raise AssertionError('Native one-step source identity mismatch')
    offset_delta=float((copy['all_offset_1'][0]-d['all_offset_1'][0]).abs().max().item()) if d['all_offset_1'].numel() else 0.
    if offset_delta>1e-5:raise AssertionError('Native one-step first recurrence differs from full trajectory')
    return copy['fine_coord_1'].float().cpu().numpy(),dict(RRU_calls=count[0],first_offset_max_deviation=offset_delta)

def evaluate_pair(rt,head,r):
    d=rt.load(r);rt.forward(d);route=head.route;s=supervision(d,route)
    features=inputs(rt,d,route);head.eval();pred=predict(head,features);del features
    target=s['initial']+pred['mu'];errors=np.linalg.norm(target.astype(float)-s['gt'],axis=1);valid=s['valid']
    raw=d['coarse_match_id'].cpu().numpy();m=d['m_idx_it'].cpu().numpy();cp=d['fine_parent_id'].cpu().numpy()
    parentlow=low_grids(d,0,rt.model.texture_threshold)['low'][d['i_idx_c'].cpu().numpy()]
    low=parentlow if route=='A' else parentlow[m]
    native_e,native_valid,native_gt=final_source_errors(d);native_valid=native_valid.astype(bool)
    sim=d['sim_matrix_f'].float();confidence=sim.softmax(1)*sim.softmax(2)
    fine=confidence[d['m_idx_it'],d['i_idx_it'],d['j_idx_it']].cpu().numpy()
    conf=d['coarse_confidence'].float().cpu().numpy();counts,vc,pe,pf,choice=parent_medians(len(raw),m,native_e,native_valid,fine)
    traj=trajectory(d);traj_errors=np.linalg.norm(traj.astype(float)-native_gt['points1_gt_input'][None],axis=2)
    if len(m)>1 and np.any(np.diff(m)<0):raise AssertionError('Native child parent blocks are not contiguous')
    if route=='B' and (not np.array_equal(valid,native_valid) or not np.array_equal(s['source'],d['fine_coord_0'].cpu().numpy())):raise AssertionError('B common child/source/GT mismatch')
    one_target,one_proof=actual_one_step(rt,d)
    one_error=np.linalg.norm(one_target.astype(float)-native_gt['points1_gt_input'],axis=1)
    one_proof['float32_reconstruction_max_deviation']=float(np.max(np.abs(one_target-traj[1]),initial=0))
    scoreconf=conf if route=='A' else conf[m];scorefine=pf if route=='A' else fine
    epi=epipolar(d,s['source'],target);finite(epi,'predicted epipolar distances');finite(errors[valid],'new valid EPE')
    a=dict(raw_id=raw,cp=s['parent'],ordinal=s['ids'],source=s['source'],initial=s['initial'],target=target,gt=s['gt'],valid=valid,inside=s['inside'],offset=s['offset'],
        epe=errors,low=low,parent_low=parentlow,mconf=scoreconf,fine=scorefine,S_m=scoreconf.astype(float)*scorefine,
        epistemic=pred['epistemic'],aleatoric=pred['aleatoric'],mu=pred['mu'],epipolar=epi,gt_reasons=s['reasons'],
        native_cp=cp,native_source=d['fine_coord_0'].cpu().numpy(),native_target=d['fine_coord_1'].cpu().numpy(),
        native_epe=native_e,native_valid=native_valid,native_fine=fine,native_parent_pos=m,native_parent_epe=pe,native_parent_fine=pf,native_fine_choice=choice,native_child_counts=counts,native_valid_child_counts=vc,
        trajectory=traj,trajectory_epe=traj_errors,native_one_target=one_target,native_one_epe=one_error,scale0=np.asarray(d['_gt']['plan0']['scale_input_to_orig_xy']),scale1=np.asarray(d['_gt']['plan1']['scale_input_to_orig_xy']),
        hw0=np.asarray(d['_gt']['plan0']['resized_hw']),hw1=np.asarray(d['_gt']['plan1']['resized_hw']),K0=d['_gt']['K0'],K1=d['_gt']['K1'],T=d['_gt']['T_0to1'])
    metrics={}
    for region,mask in [('all',np.ones(len(valid),bool)),('low_texture',low),('inside',s['inside']),('outside',valid&~s['inside'])]:
        vm=valid&mask
        methods={route:dict(**error_metrics(errors[vm]),Spearman=spearman(-pred['epistemic'][vm],errors[vm])),
            'zero':error_metrics(np.linalg.norm(s['offset'][vm],axis=1))}
        if route=='B':
            for t in [0,1,4]:methods[f't{t}']=error_metrics((native_e if t==4 else one_error if t==1 else traj_errors[t])[vm])
        else:
            common=vm&np.isfinite(pe);methods['native_parent_median']=error_metrics(pe[common]);methods['A_common_parent']=error_metrics(errors[common])
        metrics[region]=dict(**base(r),valid=int(vm.sum()),total=int(mask.sum()),methods=methods)
    result=dict(**base(r),route=route,native_one_step=one_proof,metrics=metrics,predicted=len(valid),valid=int(valid.sum()),inside=int(s['inside'].sum()),low_outputs=int(low.sum()),resources=resource_sample())
    del d,s,pred,sim,confidence;gc.collect();return result,a
def validate(run,route,epoch,rt,head):
    stage=f'{route}/validation_{epoch:02d}';rows=[];start=time.monotonic()
    for index,r in enumerate(records()):
        row=get_unit(run,stage,r)
        if row is None:
            result,a=evaluate_pair(rt,head,r);commit(run,stage,r,result,a);row=result;del a
        rows.append(row)
        beat(run,route,phase='validation',epoch=epoch+1,validation_pairs=index+1,total=750,resources=resource_sample())
    metrics=['mean_EPE','median_EPE','lt1','lt2','lt3','Spearman'];summaries={}
    for region in ['all','low_texture','inside','outside']:
        rs=[r['metrics'][region] for r in rows];methods=list(rs[0]['methods'])
        summaries[region]=scene_summary(rs,metrics,methods)
    result=dict(status='committed',epoch=epoch+1,pairs=len(rows),seconds=time.monotonic()-start,head_sha=state_hash(head),summaries=summaries,valid=sum(r['valid'] for r in rows),predicted=sum(r['predicted'] for r in rows),frozen=rt.check_frozen())
    save(Path(run)/stage/'summary.json',result);return result
