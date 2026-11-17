"""Independent E0: exact native trajectory, center labels and truncated feature contracts."""
from .common import *
from .native import prefix,inputs,supervision,trajectory
from .heads import numerical_checks
from .train_core import selftest
import torch
def differences(x,y):
    if x.shape!=y.shape:return None
    return float(np.max(np.abs(np.asarray(x,dtype=float)-np.asarray(y,dtype=float)),initial=0))
def run(run):
    run=Path(run);out=run/'E0';out.mkdir(exist_ok=True);identity_gate();save(out/'numerical_checks.json',numerical_checks())
    rt=NativeRuntime(out/'runtime');rows=[];failed={'A':[],'B':[]};counts=dict(children=0,valid=0,parents=0)
    assert not rt.model.texture_budget_enabled and rt.model.reliability_head is None
    try:
        for index,r in enumerate(records('epe')):
            row=get_unit(run,'E0',r)
            if row is None:
                beat(run,'E0',phase='native trajectory and prefix equivalence',pair_index=index,pair_id=r['pair_id'])
                base_input=rt.load(r);d=dict(base_input);rt.forward(d)
                c=historical('C',r);b=historical('B',r);a=supervision(d,'A');s=supervision(d,'B');traj=trajectory(d);raw=d['coarse_match_id'].cpu().numpy()
                low=low_grids(d,0,rt.model.texture_threshold)['low'][d['i_idx_c'].cpu().numpy()]
                childlow=low[d['m_idx_it'].cpu().numpy()];checks={}
                checks['B_identity']=all([np.array_equal(raw,c['raw_id']),np.array_equal(s['parent'],c['child_parent_id']),np.array_equal(s['source'],c['source_input']),np.array_equal(s['valid'],c['child_valid'])])
                checks['B_t0_coordinates_max_abs']=differences(traj[0],c['trajectory_coords'][0])
                checks['B_trajectory_max_abs']=differences(traj,c['trajectory_coords'])
                idx=positions(b['raw_id'],raw)
                aref=b['center_gt'][idx]-(b['coarse1'][idx]+4)
                valid_a=a['valid'];checks['A_center_offset_max_abs']=differences(a['offset'][valid_a],aref[valid_a])
                checks['A_visible_mask_equal']=bool(np.array_equal(valid_a,b['center_valid'][idx]))
                label=np.full(len(raw),-1,np.int8);label[valid_a]=((a['offset'][valid_a]+4>=0)&(a['offset'][valid_a]+4<8)).all(1).astype(np.int8)
                checks['A_center_labels_equal']=bool(np.array_equal(label,b['center_labels'][idx]))
                for route in ['A','B']:
                    fullinputs=inputs(rt,d,route)
                    truncated=dict(base_input);prefix(rt,truncated,route);prefixinputs=inputs(rt,truncated,route)
                    ds=[float((x-y).abs().max()) if x.numel() and x.shape==y.shape else 0 if x.shape==y.shape else float('inf') for x,y in zip(fullinputs,prefixinputs)]
                    checks[route+'_input_max_abs']=max(ds,default=0)
                    keys=['coarse_coord_0','coarse_coord_1','coarse_match_id']+(['intermediate_coord_0','intermediate_coord_1','fine_parent_id'] if route=='B' else [])
                    checks[route+'_prefix_coordinates_equal']=all(torch.equal(d[k],truncated[k]) for k in keys)
                    del fullinputs,prefixinputs,truncated
                errors=np.linalg.norm(traj.astype(float)-s['gt'][None],axis=2);ref_errors=c['EPE'];valid=s['valid']
                metrics={f't{t}':error_metrics(errors[t,valid]) for t in [0,1,4]}
                reference={f't{t}':error_metrics(ref_errors[t,c['child_valid'].astype(bool)]) for t in [0,1,4]}
                coverage={}
                for route,su,mask in [('A',a,low),('B',s,childlow)]:
                    coverage[route]={reg:dict(total=int(m.sum()),valid=int((su['valid']&m).sum()),inside=int((su['inside']&m).sum()),outside=int((su['valid']&~su['inside']&m).sum()),outside_fraction=ratio(int((su['valid']&~su['inside']&m).sum()),int((su['valid']&m).sum()))) for reg,m in [('all',np.ones(len(mask),bool)),('low_texture',mask)]}
                arrays=dict(raw_id=raw,cp=s['parent'],source=s['source'],initial=s['initial'],gt=s['gt'],valid=s['valid'],low=childlow,trajectory=traj,EPE=errors,A_source=a['source'],A_initial=a['initial'],A_gt=a['gt'],A_valid=a['valid'],A_inside=a['inside'],A_offset=a['offset'],A_low=low,B_inside=s['inside'],
                    native_displacement=traj[4]-traj[0],center_labels=label,gt_reasons=s['reasons'],A_gt_reasons=a['reasons'])
                row=dict(**base(r),checks=checks,coverage=coverage,metrics=metrics,reference=reference,counts=dict(children=len(s['ids']),valid=int(valid.sum()),parents=len(raw)),resources=resource_sample(),
                    A_reference='Phase3 B.center_gt - (coarse1+4); corner_residual is NOT assumed to use center projection',
                    B_cumulative_displacement=dict(axes=distributions(np.abs(traj[4]-traj[0]).ravel()),Linf=distributions(np.max(np.abs(traj[4]-traj[0]),axis=1)),outside_B_single_box=int((np.max(np.abs(traj[4]-traj[0]),axis=1)>3).sum())))
                commit(run,'E0',r,row,arrays);del d,base_input,a,s,arrays;gc.collect()
            rows.append(row)
            for k in counts:counts[k]+=row['counts'][k]
            ch=row['checks']
            if not ch['B_identity'] or ch['B_t0_coordinates_max_abs'] is None or ch['B_t0_coordinates_max_abs']>1e-5 or ch['B_trajectory_max_abs']>1e-5 or ch['B_input_max_abs']>1e-5 or not ch['B_prefix_coordinates_equal']:failed['B'].append(r['pair_id'])
            if ch['A_center_offset_max_abs'] is None or ch['A_center_offset_max_abs']>1e-3 or not ch['A_visible_mask_equal'] or not ch['A_center_labels_equal'] or ch['A_input_max_abs']>1e-5 or not ch['A_prefix_coordinates_equal']:failed['A'].append(r['pair_id'])
            beat(run,'E0',phase='committed native reproduction',committed=index+1,total=100)
        summaryrows=[dict(**base(r),methods={**{k:{'lt1':v['lt1']} for k,v in r['metrics'].items()},**{'reference_'+k:{'lt1':v['lt1']} for k,v in r['reference'].items()}}) for r in rows]
        summaries=scene_summary(summaryrows,['lt1'],['t0','t1','t4','reference_t0','reference_t1','reference_t4'])
        deviations={t:abs(summaries['scene_equal'][t]['lt1']-summaries['scene_equal']['reference_'+t]['lt1']) for t in ['t0','t1','t4']}
        if counts['children']!=484473 or counts['valid']!=361254 or max(deviations.values())>1e-6:failed['B'].append('aggregate reproduction')
        native_result=dict(status='pass' if not any(failed.values()) else 'partial',failed=failed,counts=counts,step_lt1=summaries,aggregate_max_deviations=deviations,
            prefix_max_abs={route:max(r['checks'][route+'_input_max_abs'] for r in rows) for route in ['A','B']},frozen=rt.check_frozen(),
            fine_channels=48,fine_scale=1,coarse_scale=8,bilinear_contract='native grid_sample: offsets -2.5..2.5 fine cells, normalization by [W,H]*2-1, align_corners=False, zero padding')
        save(out/'native_reproduction.json',native_result);save(out/'supervision_coverage.json',[dict(**base(r),coverage=r['coverage'],displacement=r['B_cumulative_displacement']) for r in rows])
        results={}
        for route in ['A','B']:
            if failed[route]:results[route]=dict(status='failed',reason='E0 coordinate/prefix reproduction',failures=failed[route]);continue
            prior=out/('selftest_'+route)/'summary.json'
            results[route]=read(prior) if prior.exists() else selftest(run,route,rt)
            rt.check_frozen()
        final=dict(status='pass' if all(v['status']=='pass' for v in results.values()) else 'partial',native=native_result,routes=results,selftest_formal_isolation='Separate official worker reinitializes all state, seed2026 cursor0',frozen=rt.check_frozen(),time_beijing=now())
        save(out/'summary.json',final);return final
    finally:
        save(out/'frozen_check.json',dict(time_beijing=now(),unchanged=rt.check_frozen()));del rt;gc.collect();torch.cuda.empty_cache()
