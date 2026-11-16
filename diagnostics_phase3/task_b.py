"""B: forward-only 750-pair corner versus one-way center-anchor diagnostic."""
from common import *
import gc
from reliability_data import project_visible,label_parents,_numpy
from reliability_v1.runtime import resource_sample
def label_alternative(data):
    p0=_numpy(data['coarse_coord_0']);p1=_numpy(data['coarse_coord_1'])
    hi,wi=data['image0'].shape[-2:];hc,wc=data['hw0_c'];stride0=np.array([wi/wc,hi/hc],float)
    hi,wi=data['image1'].shape[-2:];hc,wc=data['hw1_c'];stride1=np.array([wi/wc,hi/hc],float)
    center=p0+stride0/2;gt=project_visible(center,data);residual=gt['points1_gt_input']-p1
    rh,rw=data['_gt']['plan1']['resized_hw']
    selected=np.isfinite(p1).all(axis=1)&(p1[:,0]>=0)&(p1[:,1]>=0)&(p1[:,0]<rw)&(p1[:,1]<rh)
    valid=gt['valid_gt']&selected;target=((residual>=0)&(residual<stride1)).all(axis=1)
    labels=np.full(len(p0),-1,np.int8);labels[valid]=target[valid]
    corner=project_visible(p0,data)
    e=gt['points1_gt_input']-corner['points1_gt_input']-stride1/2
    offsets=[(4,0),(-4,0),(0,4),(0,-4)]
    samples=[project_visible(center+np.asarray(x),data) for x in offsets]
    jacvalid=np.logical_and.reduce([x['valid_gt'] for x in samples])&valid
    jx=(samples[0]['points1_gt_input']-samples[1]['points1_gt_input'])/8
    jy=(samples[2]['points1_gt_input']-samples[3]['points1_gt_input'])/8
    det=jx[:,0]*jy[:,1]-jx[:,1]*jy[:,0]
    scale=np.full(len(p0),np.nan);scale[jacvalid]=np.sqrt(np.abs(det[jacvalid]))
    finite(residual[valid],'center valid residual');finite(e[valid&corner['valid_gt']],'anchor displacement');finite(scale[jacvalid],'Jacobian scale')
    return dict(center_labels=labels,center_valid=valid,center_gt=gt['points1_gt_input'],anchor_displacement=e,local_scale=scale,local_scale_valid=jacvalid,
        stride0=stride0,stride1=stride1,coarse0=p0,coarse1=p1)
def confusion(y0,y1,mask):
    return [[int((mask&(y0==i)&(y1==j)).sum()) for j in [0,1]] for i in [0,1]]
def bins(values,common,y0,y1,edges,labels,unknown=False):
    ids=np.searchsorted(edges,values,side='right' if unknown else 'left')
    if unknown:ids=np.where(values==edges[-1],len(edges)-1,ids)
    out=[];known=common&np.isfinite(values)
    for i,label in enumerate(labels):
        m=known&(ids==i);n=int(m.sum());ag=int((m&(y0==y1)).sum())
        out.append(dict(bin=label,count=n,agree=ag,flip=n-ag,share=n/int(common.sum()) if common.any() else None,agreement=ag/n if n else None))
    if unknown:
        m=common&~np.isfinite(values);n=int(m.sum());ag=int((m&(y0==y1)).sum())
        out.append(dict(bin='unknown',count=n,agree=ag,flip=n-ag,share=n/int(common.sum()) if common.any() else None,agreement=ag/n if n else None))
    return out
def aggregate_counts(items,keys):
    return {k:sum(x[k] for x in items) for k in keys}
def ba_row(record,predictions,truth):
    out={}
    for name,p in predictions.items():
        c=np.array([[int(((truth==i)&(p==j)).sum()) for j in [0,1]] for i in [0,1]])
        tn,fp,fn,tp=c.ravel();recall=tp/(tp+fn) if tp+fn else None;specificity=tn/(tn+fp) if tn+fp else None
        out[name]=dict(matrix_true_rows_pred_columns=c.tolist(),precision=tp/(tp+fp) if tp+fp else None,recall=recall,
            BA=(recall+specificity)/2 if recall is not None and specificity is not None else None)
    return dict(**base(record),valid=len(truth),methods=out)
def main(run):
    import subprocess,torch
    run=Path(run);out=run/'B';recs=records();checks=[];allrows={};pairs=[];maxerr={m:0. for m in METHODS};epe_pair_ids={x['pair_id'] for x in records('epe')}
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
    if gpu and gpu!='No running processes found':raise RuntimeError('GPU is already used: '+gpu)
    runtime=None;start=time.monotonic()
    try:
        runtime,stats,fit=start_runtime(run,'B')
        for index,r in enumerate(recs):
            receipt=committed(run,'B',r)
            if receipt is None:
                data,corner=runtime.forward(r,full=False);p=forward_scores(runtime,data,stats,fit)
                # Explicitly reuse the current label function; no native-label substitution.
                y=np.asarray(corner['labels'],np.int8);pred,ref=cache(r,'validation');raw=p['raw_id']
                check=dict(**base(r),layer1_parent_ids=False,layer2_labels=False,layer2_scores=False,max_abs={})
                if len(raw)!=len(pred['raw_id']) or not np.array_equal(np.sort(raw),np.sort(pred['raw_id'])):
                    save(out/'B0_reproduction.json',dict(status='failed',failed_layer=1,pair=check));raise GateFailure('B0 layer 1 parent IDs mismatch')
                check['layer1_parent_ids']=True;align=positions(raw,pred['raw_id'])
                check['layer2_labels']=bool(np.array_equal(y[align],pred['labels']))
                check['low_texture_mask_equal']=bool(np.array_equal(p['low'][align],pred['low']))
                for m in METHODS:check['max_abs'][m]=float(np.max(np.abs(p[m][align].astype(float)-pred[m].astype(float)),initial=0))
                check['layer2_scores']=all(check['max_abs'][m]<=1e-4 for m in ['head','mconf'])
                if not check['layer2_labels'] or not check['layer2_scores']:
                    save(out/'B0_reproduction.json',dict(status='failed',failed_layer=2,pair=check));raise GateFailure('B0 layer 2 labels/scores mismatch')
                alternative=label_alternative(data);yc=alternative['center_labels'];common=(y>=0)&(yc>=0)
                finite(alternative['anchor_displacement'][common],'joint valid displacement')
                counts=dict(total=len(raw),both_valid=int(common.sum()),corner_only=int(((y>=0)&(yc<0)).sum()),center_only=int(((y<0)&(yc>=0)).sum()),both_invalid=int(((y<0)&(yc<0)).sum()),
                    corner_valid=int((y>=0).sum()),corner_positive=int((y==1).sum()),center_valid=int((yc>=0).sum()),center_positive=int((yc==1).sum()),
                    agree=int((common&(y==yc)).sum()),matrix=confusion(y,yc,common))
                cls={};ebins={};localbins={}
                for region,mask in [('all',np.ones(len(y),bool)),('low_texture',p['low'].astype(bool))]:
                    cls[region+'/corner_own_valid']=classify(r,y,{m:p[m] for m in METHODS},mask&(y>=0))
                    cls[region+'/center_own_valid']=classify(r,yc,{m:p[m] for m in METHODS},mask&(yc>=0))
                    cls[region+'/corner_common_valid']=classify(r,y,{m:p[m] for m in METHODS},mask&common)
                    cls[region+'/center_common_valid']=classify(r,yc,{m:p[m] for m in METHODS},mask&common)
                    e=alternative['anchor_displacement']
                    ebins[region+'/linf']=bins(np.max(np.abs(e),axis=1),common&mask,y,yc,[.5,1,2,4],['[0,0.5]','(0.5,1]','(1,2]','(2,4]','>4'])
                    ebins[region+'/l2']=bins(np.linalg.norm(e,axis=1),common&mask,y,yc,[.5,1,2,4],['[0,0.5]','(0.5,1]','(1,2]','(2,4]','>4'])
                    localbins[region]=bins(alternative['local_scale'],common&mask,y,yc,[.8,.9,1.1,1.25,1.5],['<0.8','[0.8,0.9)','[0.9,1.1)','[1.1,1.25)','[1.25,1.5]','>1.5'],True)
                arrays=dict(**p,labels=y,corner_residual=np.asarray(corner['residual_input']),corner_gt=np.asarray(corner['points1_gt_input']),**alternative,
                    resized_hw0=np.asarray(data['_gt']['plan0']['resized_hw']),resized_hw1=np.asarray(data['_gt']['plan1']['resized_hw']),
                    input_to_orig0=np.asarray(data['_gt']['plan0']['scale_input_to_orig_xy']),input_to_orig1=np.asarray(data['_gt']['plan1']['scale_input_to_orig_xy']))
                result=dict(**base(r),reproduction=check,counts=counts,classification=cls,displacement_bins=ebins,local_scale_bins=localbins,
                    resources=resource_sample(),actual=dict(fine_scale_not_produced_by_coarse_only=True,stride0=alternative['stride0'].tolist(),stride1=alternative['stride1'].tolist()))
                receipt=commit(run,'B',r,result,arrays,ref)
                del data,corner,p,arrays,alternative;gc.collect()
            one=result_of(run,'B',receipt);pairs.append(one);checks.append(one['reproduction'])
            for m in METHODS:maxerr[m]=max(maxerr[m],one['reproduction']['max_abs'][m])
            for name,row in one['classification'].items():allrows.setdefault(name,[]).append(row)
            heartbeat(run,'B',phase='coarse forward / B0 layers 1-2',committed_pairs=index+1,total_pairs=750,max_absolute_score_deviation=maxerr,elapsed_seconds=time.monotonic()-start)
        frozen_check=check_runtime(runtime);save(out/'frozen_check.json',frozen_check)
    finally:
        if runtime is not None:
            try:save(out/'frozen_check.json',check_runtime(runtime))
            finally:del runtime;gc.collect();torch.cuda.empty_cache()
    summ={name:summary(rs,['score_AUROC','AP']) for name,rs in allrows.items()}
    baseline=summ['all/corner_own_valid'];frozen=read(SOURCE/'formal_train'/('validation_%02d'%read(SOURCE/'formal_train/status.json')['state']['best_epoch'])/'summary.json')
    defined={m:750-baseline['undefined_pair_counts'][m]['score_AUROC'] for m in METHODS}
    deviations={m:abs(baseline['scene_equal'][m]['score_AUROC']-frozen['ranking']['scene_equal'][m]['score_AUROC']) for m in METHODS}
    passed=all(defined[m]==747 and deviations[m]<=1e-6 for m in ['head','mconf'])
    b0=dict(status='pass' if passed else 'failed',failed_layer=None if passed else 3,pairs=750,layer1='pass',layer2='pass',layer3='pass' if passed else 'failed',
        max_absolute_score_deviation=maxerr,defined_pairs=defined,aggregate_AUROC_absolute_deviation=deviations,
        label_mismatches=0,parent_ID_mismatches=0,low_texture_masks_equal=all(x['low_texture_mask_equal'] for x in checks))
    save(out/'B0_reproduction.json',b0)
    if not passed:raise GateFailure('B0 layer 3 failed; B/R4/cross-stage dependence blocked')
    save(out/'label_score_comparison.json',summ);save(out/'label_score_rows.json',allrows);metric_csv(out/'label_score_comparison.csv',summ)
    keys=['total','both_valid','corner_only','center_only','both_invalid','corner_valid','corner_positive','center_valid','center_positive','agree']
    pooled=aggregate_counts([x['counts'] for x in pairs],keys);pooled['matrix']=np.sum([x['counts']['matrix'] for x in pairs],axis=0).tolist()
    pooled['agreement']=pooled['agree']/pooled['both_valid'];pooled['corner_positive_fraction']=pooled['corner_positive']/pooled['corner_valid'];pooled['center_positive_fraction']=pooled['center_positive']/pooled['center_valid']
    scenes={}
    for scene in sorted({r['scene'] for r in pairs}):
        a=[r['counts'] for r in pairs if r['scene']==scene];d=aggregate_counts(a,keys);d['matrix']=np.sum([x['matrix'] for x in a],axis=0).tolist()
        d['agreement']=d['agree']/d['both_valid'] if d['both_valid'] else None;scenes[scene]=d
    agreement=dict(pooled=pooled,per_scene=scenes,scene_equal_agreement=float(np.mean([x['agreement'] for x in scenes.values() if x['agreement'] is not None])))
    save(out/'label_agreement.json',agreement);csvout(out/'label_agreement_per_scene.csv',[dict(scene=s,**v) for s,v in scenes.items()])
    def combine_bins(field):
        outputs={}
        for key in pairs[0][field]:
            rows=[]
            for i in range(len(pairs[0][field][key])):
                samples=[x[field][key][i] for x in pairs];n=sum(x['count'] for x in samples);ag=sum(x['agree'] for x in samples)
                denominator=sum(sum(z['count'] for z in x[field][key]) for x in pairs)
                per_scene={}
                for scene in scenes:
                    a=[x[field][key][i] for x in pairs if x['scene']==scene];sn=sum(v['count'] for v in a);sa=sum(v['agree'] for v in a)
                    per_scene[scene]=dict(count=sn,agree=sa,agreement=sa/sn if sn else None)
                rows.append(dict(bin=samples[0]['bin'],count=n,agree=ag,flip=n-ag,share=n/denominator if denominator else None,agreement=ag/n if n else None,
                    per_scene=per_scene,scene_equal_agreement=float(np.mean(v)) if (v:=[a['agreement'] for a in per_scene.values() if a['agreement'] is not None]) else None))
            outputs[key]=rows
        return outputs
    dbins=combine_bins('displacement_bins');sbins=combine_bins('local_scale_bins')
    save(out/'anchor_displacement_bins.json',dbins);save(out/'local_scale_bins.json',sbins)
    # At most the fixed 750-pair joint-valid float64 2-vectors; no image/features retained.
    es=[];lows=[];export=[]
    for r in recs:
        receipt=committed(run,'B',r);a=arrays_of(run,'B',receipt);mask=(a['labels']>=0)&(a['center_labels']>=0)
        es.append(a['anchor_displacement'][mask]);lows.append(a['anchor_displacement'][mask&a['low'].astype(bool)])
        if r['pair_id'] in epe_pair_ids:
            export.append(dict(**base(r),payload=str(out/'pairs'/receipt['files']['arrays']['name']),sha256=receipt['files']['arrays']['sha256'],
                fields=['raw_id','labels','center_labels','coarse0','coarse1','corner_residual'],coordinate_contract='R832 input pixels; current corner convention; stride recorded'))
    e=np.concatenate(es);lowe=np.concatenate(lows)
    save(out/'anchor_displacement_distribution.json',dict(all=dict(x=distributions(e[:,0]),y=distributions(e[:,1])),low_texture=dict(x=distributions(lowe[:,0]),y=distributions(lowe[:,1]))))
    save(out/'coarse_100_index.json',dict(status='committed',records=export,pairs=len(export),key=['pair_id','direction','raw_id']))
    a0=read(run/'A/A0_reproduction.json') if (run/'A/A0_reproduction.json').exists() else {}
    r4rows=[];r4class=[]
    if a0.get('status')=='pass':
        for r in records('epe'):
            b=arrays_of(run,'B',committed(run,'B',r));a=arrays_of(run,'A',committed(run,'A',r));align=positions(b['raw_id'],a['raw_id'])
            if not np.array_equal(b['labels'][align],a['labels']):raise GateFailure('R4/B-A corner label association mismatch')
            yc=b['center_labels'][align];mask=a['valid_final_parent']&(a['labels']>=0)&(yc>=0);truth=(a['parent_epe'][mask]<2).astype(np.int8)
            r4rows.append(ba_row(r,dict(corner=a['labels'][mask],center=yc[mask]),truth))
            r4class.append(classify(r,yc,{m:a[m] for m in METHODS},mask))
        comparison=scene_bootstrap(r4rows,'BA','center','corner',2000,2026)
        per_scene={s:float(np.mean(v)) if (v:=[r['methods']['center']['BA']-r['methods']['corner']['BA'] for r in r4rows if r['scene']==s and all(r['methods'][m]['BA'] is not None for m in ['corner','center'])]) else None for s in scenes}
        pooled_matrices={m:np.sum([r['methods'][m]['matrix_true_rows_pred_columns'] for r in r4rows],axis=0).tolist() for m in ['corner','center']}
        r4=dict(status='completed',pairs=100,parents=sum(r['valid'] for r in r4rows),rows=r4rows,paired_delta_BA=comparison,per_scene_delta_BA=per_scene,pooled_confusion=pooled_matrices,center_target_scores=summary(r4class,['score_AUROC','AP']))
        # Pooled precision/recall/BA are derived from pooled confusion, separately from pair/scene means.
        r4['pooled_metrics']={}
        for m,matrix in pooled_matrices.items():
            tn,fp,fn,tp=np.asarray(matrix).ravel()
            recall=tp/(tp+fn) if tp+fn else None;spec=tn/(tn+fp) if tn+fp else None
            r4['pooled_metrics'][m]=dict(precision=tp/(tp+fp) if tp+fp else None,recall=recall,BA=(recall+spec)/2 if recall is not None and spec is not None else None)
        save(out/'R4_label_final_EPE_comparison.json',r4);csvout(out/'R4_per_pair.csv',r4rows)
    else:
        r4={'status':'不可判定','reason':'A0 missing or failed'};save(out/'R4_label_final_EPE_comparison.json',r4)
    triggered=pooled['agreement']<.95 or any(x['share'] is not None and x['share']>=.05 and x['agreement']<.9 for x in dbins['all/linf'])
    ba_ci=r4.get('paired_delta_BA',{}).get('ci95')
    optional=dict(status='跳过',reason='Optional native-supervision three-factor decomposition not executed; one-way center-anchor labels are not claimed to be native SLiM supervision.')
    save(out/'optional_native_supervision.json',optional)
    result=dict(status='completed',B0=b0,G1=dict(status='触发' if triggered else '未触发',agreement=pooled['agreement'],sensitive_bins=[x for x in dbins['all/linf'] if x['share'] is not None and x['share']>=.05 and x['agreement']<.9],
        label_training_control_supported=bool(triggered and ba_ci and ba_ci[0]>0),R4_delta_BA_ci95=ba_ci),pairs=750,optimizer_updates=0,checkpoint_writes=0,frozen_check=frozen_check,R4_status=r4['status'])
    save(out/'summary.json',result);return result
