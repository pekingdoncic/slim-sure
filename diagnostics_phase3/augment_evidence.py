"""Complete derived cross-references and static fine-window audit without new inference."""
from common import *
import ast
def main(run):
    run=Path(run);A=run/'A';B=run/'B';C=run/'C'
    r4=read(B/'R4_label_final_EPE_comparison.json')
    if r4['status']=='completed':
        save(A/'R4_label_final_EPE_comparison.json',r4)
        comp=read(A/'A2_same_parent_comparison.json');comp['all/R4_center_label_joint_final_valid']=r4['center_target_scores']
        save(A/'A2_same_parent_comparison.json',comp);metric_csv(A/'A2_same_parent_comparison.csv',comp)
        a=read(A/'summary.json');a['R4']='completed';a['R4_path']=str(A/'R4_label_final_EPE_comparison.json');save(A/'summary.json',a)
    lowrows=[]
    for r in records():
        arr=arrays_of(run,'B',committed(run,'B',r));low=arr['low'].astype(bool);y=arr['labels'];c=arr['center_labels'];both=low&(y>=0)&(c>=0)
        lowrows.append(dict(**base(r),total_low_parents=int(low.sum()),both_valid=int(both.sum()),corner_only=int((low&(y>=0)&(c<0)).sum()),center_only=int((low&(y<0)&(c>=0)).sum()),both_invalid=int((low&(y<0)&(c<0)).sum()),
            corner_valid=int((low&(y>=0)).sum()),corner_positive=int((low&(y==1)).sum()),center_valid=int((low&(c>=0)).sum()),center_positive=int((low&(c==1)).sum()),
            agree=int((both&(y==c)).sum()),matrix=[[int((both&(y==i)&(c==j)).sum()) for j in [0,1]] for i in [0,1]]))
    def grouped(rows):
        keys=[k for k in lowrows[0] if k not in ['pair_id','scene','direction','matrix']]
        d={k:sum(r[k] for r in rows) for k in keys};d['matrix']=np.sum([r['matrix'] for r in rows],axis=0).tolist()
        d.update(agreement=d['agree']/d['both_valid'] if d['both_valid'] else None,corner_positive_fraction=d['corner_positive']/d['corner_valid'] if d['corner_valid'] else None,center_positive_fraction=d['center_positive']/d['center_valid'] if d['center_valid'] else None)
        return d
    per_scene={s:grouped([r for r in lowrows if r['scene']==s]) for s in sorted({r['scene'] for r in lowrows})}
    save(B/'low_texture_label_validity.json',dict(pooled=grouped(lowrows),per_scene=per_scene,per_pair=lowrows,
        scene_equal_agreement=float(np.mean([r['agreement'] for r in per_scene.values() if r['agreement'] is not None]))))
    source=ROOT/'src/slim.py';tree=ast.parse(source.read_text());functions={}
    for node in ast.walk(tree):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in ['_fine_correlation','_get_fine_coord_test','_extract_feat_window','_extract_feat_window_bilinear','_refinement']:
            functions[node.name]=dict(path=str(source),start_line=node.lineno,end_line=node.end_lineno,source_sha256=sha(source))
    actual=read(C/'C0_acceptance.json');first=records()[0];meta=result_of(run,'B',committed(run,'B',first));stride=meta['actual']['stride1'];fine=actual['actual_fine_scale']
    audit=dict(status='completed',source_functions=functions,observed=dict(coarse_stride_xy=stride,fine_scale=fine,refine_iters=actual['actual_T'],lookup_radius=actual['actual_lookup_radius']),
        correlation_window=dict(feature_index_start='round(coarse_coord / fine_scale)',feature_offsets='0..7 in each axis at measured stride8 / fine_scale1',reported_input_pixel_centers='coarse_coord + (0.5+i, 0.5+j), i,j=0..7',
            step_pixels=1,center_extent_relative_to_coarse=[[.5,7.5],[.5,7.5]],cell_footprint_relative_to_coarse='[0,8) x [0,8)',window_center_relative_to_coarse=[4.,4.]),
        boundary=dict(correlation='_extract_feat_window uses direct integer indexing, no clamp or own padding; windows tile the padded R832 feature map. Coarse valid masks constrain selected anchors, not every fine candidate. Partial boundary cells may contain padded image features.',
            refinement='_extract_feat_window_bilinear uses a centered 6x6 grid, offsets -2.5..2.5 fine units, grid_sample align_corners=False and padding_mode=zeros; final GT visibility is checked separately.'),
        anchor_interpretation=dict(corner='coarse_coord is used as the fine-cell top-left anchor. The current label tests GT(corner)-target_corner within inclusive [-4,4] per axis, which is not the positive [0,8) fine-cell footprint.',
            center='The one-way center control projects source_corner+(4,4) and tests target cell [0,8). Its cell geometry is compatible with the centered source/positive-extent target cell convention; it is still not per-child GT and not native spvs_coarse supervision.',
            causal_limit='This establishes coordinate conventions, not a proof that the current label implementation is incorrect or that retraining improves pose.'))
    save(B/'fine_window_geometry_audit.json',audit)
    lines=['# Fine-window geometry audit','',json.dumps(audit,ensure_ascii=False,indent=2)]
    text(B/'fine_window_geometry_audit.md','\n'.join(lines)+'\n')
    # Exact denominators and explicit masks prevent confusion with the helper's generic outside-mask field.
    notes=dict(bin_endpoints='Displacement [0,.5],(.5,1],(1,2],(2,4],>4. Scale <.8,[.8,.9),[.9,1.1),[1.1,1.25),[1.25,1.5],>1.5; invalid projected differences are unknown.',
        classification_counts='valid is the analyzed parent count. Generic row total is the source parent-array size; generic ignore means outside that analyzed mask, including subset exclusions, not only invalid GT. True GT validity categories are in label_agreement / low_texture_label_validity and A1 tables.',
        quality_parent_signal='Median child signal over exactly the valid children used for median parent EPE.',
        robust_curves='Primary columns aggregate per-pair statistics through equal scenes; pooled medians/precision supplied separately.',
        pose_grid='32 px primary occupancy uses the valid resized R832 input area (excluding padding); original-image 32 px occupancy also supplied. Cells intersecting the valid rectangle count in the denominator.',
        random_parent_control='Stable pair-specific seed from SHA256({seed:2026,pair_id,purpose}); one permutation reused as prefixes for all budgets.',
        child_reordering='Parent ID plus source coordinates; ambiguous identities are not matched by target error.',
        A0_execution='All committed cached payloads were checksummed and A0 metrics accumulated in the same per-pair CPU pass as provisional A1-A3 rows; A1-A3 aggregate reports were published only after A0 passed.',
        geometry_seed_adapter='Persistent CPU process schedules one seed at a time using unchanged canonical estimate_pose/pose_error; reuses geometry_worker protocol validation, array identity and commit checksum conventions.')
    save(run/'protocol_assumptions.json',notes)
    return dict(status='completed',R4_backfilled=True,fine_window_audit=str(B/'fine_window_geometry_audit.json'))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();print(json.dumps(main(Path(a.run))))
