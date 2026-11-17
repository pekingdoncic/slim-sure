"""CPU reporting helpers; audit committed journals, checkpoints, and fixed support."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from .common import *
from collections import Counter
def f(v,digits=6):
    if v is None:return '无定义'
    if isinstance(v,(float,np.floating)):return f'{v:.{digits}f}'
    return str(v)
def table(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(f(x).replace('|','/').replace('\n',' ') for x in row)+'|' for row in rows])
def ci(x):
    if not x or x.get('mean_difference') is None:return '无定义'
    if x.get('ci95') is None:return f(x['mean_difference'])+' [CI undefined]'
    return f(x['mean_difference'])+' ['+', '.join(f(v) for v in x['ci95'])+']'
def audit_training(run):
    import torch
    from .train_core import head_identity
    run=Path(run);result={};curve=[];validation_rows=[];checkpoint_rows=[]
    plans=[read(SOURCE/'plans'/f'epoch_{epoch:02d}.json')['visits'] for epoch in range(2)]
    for route in ['A','B']:
        out=run/route;p=out/'training_summary.json'
        if not p.exists():
            result[route]=dict(status='not_completed');continue
        training=read(p);expected=0;updates=0;seen=set();changed=set();skips=Counter();epochs=[];curve_bucket=[]
        maxima=Counter();minima={};sum_samples=0;sum_outside=0;sum_valid=0;sum_predicted=0;floor_hits=np.zeros(3)
        for block in sorted((out/'blocks').glob('epoch_*_pos_*.json')):
            for row in read(block):
                ep,pos=row['epoch'],row['position']
                if ep*13800+pos!=expected or row['pair_id']!=plans[ep][pos]['pair_id']:raise AssertionError('Committed journal cursor/identity mismatch')
                expected+=1;seen.add(row['pair_id']);updates+=int(row['updated'])
                if row['updated']:
                    changed.add(row['pair_id']);sum_samples+=row['samples'];curve_bucket.append(row);floor_hits+=np.asarray(row['floor_rates'])*row['samples']*2
                    if not all(np.isfinite(row[k]) for k in ['loss','nll','regularizer','grad_norm']):raise FloatingPointError('Nonfinite journal')
                else:skips[row['reason']]+=1
                sum_outside+=row['outside'];sum_valid+=row['valid_GT'];sum_predicted+=row['predicted']
                for k,v in row['resources'].items():
                    if isinstance(v,(float,int)) and k not in ['monotonic','time']:maxima[k]=max(maxima.get(k,v),v);minima[k]=min(minima.get(k,v),v)
                if expected%100==0 and curve_bucket:
                    curve.append(dict(route=route,visits=expected,epoch=ep+1,updates=updates,**{k:float(np.mean([x[k] for x in curve_bucket])) for k in ['loss','nll','regularizer','grad_norm']},
                        mean_floor_rates=np.mean([x['floor_rates'] for x in curve_bucket],axis=0).tolist()))
                    curve_bucket=[]
        if expected!=27600 or updates!=training['updates'] or dict(skips)!=training['skip']:raise AssertionError('Training summary/journal mismatch')
        ident=head_identity(run,route);objects={}
        for name in ['last','epoch_00','epoch_01']:
            pointer=out/(name+'.json');meta=read(pointer);payload=pointer.parent/meta['payload']
            if sha(payload)!=meta['sha256'] or meta['identity_sha']!=digest(ident):raise IdentityFailure('Checkpoint checksum/identity mismatch')
            obj=torch.load(payload,map_location='cpu',weights_only=False)
            if obj['identity']!=ident or not all(k in obj for k in ['head','optimizer','rng','position','extra']):raise IdentityFailure('Checkpoint embedded payload mismatch')
            if 'scaler' not in obj['extra'] or not obj['rng']:raise AssertionError('Missing scaler/RNG')
            if any(not torch.isfinite(v).all() for v in obj['head'].values()):raise FloatingPointError('Nonfinite checkpoint head')
            pos=obj['position'];steps=[int(v['step']) for v in obj['optimizer']['state'].values()]
            if not steps or min(steps)!=pos['updates'] or max(steps)!=pos['updates']:raise AssertionError('Optimizer step counters do not match committed updates')
            for state in obj['optimizer']['state'].values():
                for k,v in state.items():
                    if torch.is_tensor(v) and not torch.isfinite(v).all():raise FloatingPointError('Nonfinite optimizer state')
            item=dict(route=route,name=name,pointer=str(pointer),payload=str(payload),size=payload.stat().st_size,sha256=meta['sha256'],identity_sha=meta['identity_sha'],position=pos,
                optimizer_min_step=min(steps),optimizer_max_step=max(steps),RNG_fields=list(obj['rng']),scaler=obj['extra']['scaler'])
            checkpoint_rows.append(item);objects[name]=obj
        if objects['last']['position']['epoch']!=2 or objects['last']['position']['next_pos']!=0 or objects['epoch_01']['position']['completed_epoch']!=2:raise AssertionError('Final epoch cursor invalid')
        if any(not torch.equal(v,objects['epoch_01']['head'][k]) for k,v in objects['last']['head'].items()):raise AssertionError('Last and fixed epoch2 weights differ')
        del objects
        for ep in range(2):
            stage=f'{route}/validation_{ep:02d}';sm=read(run/stage/'summary.json')
            if sm['pairs']!=750 or not sm['frozen']:raise AssertionError('Incomplete validation/frozen check')
            for r in records():
                x=receipt(run,stage,r)
                if x is None:raise AssertionError('Missing committed validation pair')
            for region,s in sm['summaries'].items():
                for m,vals in s['scene_equal'].items():
                    validation_rows.append(dict(route=route,epoch=ep+1,region=region,method=m,**vals))
            epochs.append(dict(epoch=ep+1,planned=13800,visits=training['epoch_visits'][ep],updates=training['epoch_updates'][ep],
                skip=training['epoch_visits'][ep]-training['epoch_updates'][ep],training_seconds=training['epoch_training_seconds'][ep],validation_seconds=sm['seconds'],
                validation_pairs=750,validation_predicted=sm['predicted'],validation_valid_GT=sm['valid'],validation_head_sha=sm['head_sha']))
        physical=0
        with (out/'physical_visits.jsonl').open() as fp:
            for line in fp:
                if line.strip():physical+=1
        planned=set(v['pair_id'] for plan in plans for v in plan)
        result[route]=dict(status='completed',summary=training,epochs=epochs,committed_visits=expected,committed_updates=updates,physical_log_rows=physical,
            skip_reasons=dict(skips),planned_unique_pairs=len(planned),visited_unique_pairs=len(seen),update_unique_pairs=len(changed),unique_coverage=ratio(len(seen),len(planned)),
            supervised_samples=sum_samples,floor_hit_fractions=dict(zip(['nu','alpha_minus_one','beta'],(floor_hits/(sum_samples*2)).tolist() if sum_samples else [None]*3)),visited_valid_GT=sum_valid,visited_outside=sum_outside,visited_predictions=sum_predicted,resource_maxima=dict(maxima),resource_minima=minima,
            first50=read(out/'first50_eta.json'),formal_initialization=read(out/'formal_initialization.json'),final_frozen_check=read(out/'frozen_exit.json'))
        beat(run,'report',phase='CPU training/checkpoint audit',route=route,visits=expected)
    out=Path(read(run/'context.json')['results'])
    save(out/'training.json',result);save(out/'checkpoints.json',checkpoint_rows);csvout(out/'training_curve.csv',curve);csvout(out/'validation_by_epoch.csv',validation_rows)
    csvout(out/'checkpoints.csv',[{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()} for row in checkpoint_rows])
    return result,curve,checkpoint_rows
def coverage_summary(run):
    run=Path(run);out=Path(read(run/'context.json')['results']);rows=[];pooled={};displacements=[]
    for item in read(run/'E0/supervision_coverage.json'):
        for route,regions in item['coverage'].items():
            for region,z in regions.items():
                x=dict(**base(item),route=route,region=region,**z,GT_rate=ratio(z['valid'],z['total']),inside_fraction=ratio(z['inside'],z['valid']))
                rows.append(x)
                q=pooled.setdefault(route+'/'+region,dict(total=0,valid=0,inside=0,outside=0,pairs=0,scenes=set()))
                for k in ['total','valid','inside','outside']:q[k]+=z[k]
                q['pairs']+=int(z['valid']>0)
                if z['valid']:q['scenes'].add(item['scene'])
        displacements.append(dict(**base(item),**item['displacement']))
    by_scene={}
    for key,v in pooled.items():
        route,region=key.split('/');subset=[x for x in rows if x['route']==route and x['region']==region]
        sr=[]
        for scene in sorted({x['scene'] for x in subset}):
            values=[x for x in subset if x['scene']==scene]
            metrics={k:float(np.mean(valid)) if (valid:=[x[k] for x in values if x[k] is not None]) else None for k in ['GT_rate','inside_fraction','outside_fraction']}
            sr.append(dict(scene=scene,**metrics,**{k:sum(x[k] for x in values) for k in ['total','valid','inside','outside']}))
        by_scene[key]=sr;v['scenes']=len(v['scenes']);v['GT_rate']=ratio(v['valid'],v['total']);v['inside_fraction']=ratio(v['inside'],v['valid']);v['outside_fraction']=ratio(v['outside'],v['valid'])
        v['scene_equal']={k:float(np.mean(a)) if (a:=[x[k] for x in sr if x[k] is not None]) else None for k in ['GT_rate','inside_fraction','outside_fraction']}
    disp=dict(points=sum(x['Linf']['count'] for x in displacements),outside_B_single_box=sum(x['outside_B_single_box'] for x in displacements),
        observed_max_Linf=max(x['Linf']['quantiles']['max'] for x in displacements),definition='Observed native cumulative four-step displacement; B prediction support independently bounded to each axis [-3,3].')
    result=dict(pooled=pooled,per_scene=by_scene,native_displacement=disp)
    save(out/'supervision_coverage.json',result);csvout(out/'supervision_coverage_pairs.csv',rows);return result
def stage_inventory(run):
    run=Path(run);items=[]
    for stage in ['E0','A','B','evaluate','report']:
        p=run/stage
        if not p.exists():continue
        value=dict(stage=stage,supervisor_log=str(p/'supervisor.log'),worker_log=str(p/'worker.log'),heartbeat_path=str(p/'heartbeat.json'))
        for name in ['launch','status','worker_status','heartbeat','supervisor_heartbeat']:
            q=p/(name+'.json')
            if q.exists():
                data=read(q);value[name]={k:v for k,v in data.items() if k!='result'}
        value['failure_evidence']=[dict(path=str(x),sha256=sha(x),reason=read(x).get('reason')) for x in p.glob('failure_*.json')]
        items.append(value)
    return items
def figures(out,training,curve,metrics,efficiency):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(10,3.7))
    for route in training:
        rs=[x for x in curve if x['route']==route]
        if rs:
            ax[0].plot([x['visits'] for x in rs],[x['loss'] for x in rs],label=route)
            ax[1].plot([x['visits'] for x in rs],[x['grad_norm'] for x in rs],label=route)
    ax[0].set(xlabel='Planned visits',ylabel='NIG loss + regularizer');ax[1].set(xlabel='Planned visits',ylabel='Gradient norm')
    for x in ax:x.legend();x.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'training.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(10,3.7))
    if 'B/all' in metrics['quality']:
        d=metrics['quality']['B/all']['scene_equal'];names=['t0','t1','t4','B']
        ax[0].bar(names,[d[n]['lt1'] for n in names]);ax[0].set(ylabel='Scene-equal EPE <1px',ylim=(0,1))
    d=efficiency.get('summary',{}).get('scene_equal',{})
    if d:ax[1].bar(list(d),[1000*v['seconds'] for v in d.values()]);ax[1].set(ylabel='Fine-stage milliseconds')
    fig.tight_layout();fig.savefig(out/'coordinate_and_efficiency.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(10,3.7))
    for x,region in zip(ax,['all','low_texture']):
        d=metrics['within_window'].get(region+'/common',{}).get('scene_equal',{})
        if d:x.bar(list(d),[v['precision_lt2'] or 0 for v in d.values()])
        x.set(title=region,ylabel='Common-valid window EPE <2px',ylim=(0,1))
    fig.tight_layout();fig.savefig(out/'within_window.png',dpi=150);plt.close(fig)
def artifact_manifest(run,out):
    run=Path(run);out=Path(out);items=[]
    excluded={'artifact_manifest.json','artifact_manifest.csv'}
    for root,kind in [(out,'results'),(ROOT/'evidential_fine','implementation')]:
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.name not in excluded and '__pycache__' not in p.parts:
                items.append(dict(kind=kind,path=str(p),size=p.stat().st_size,sha256=sha(p)))
    # All bulky detail artifacts are linked by absolute path, never copied to results.
    for p in sorted(run.rglob('*')):
        if p.is_file() and (p.suffix in ['.npz','.pt'] or p.name.endswith('.result.json') or 'blocks' in p.parts or p.name=='physical_visits.jsonl' or p.stat().st_size>10*1024**2):
            items.append(dict(kind='run_detail',path=str(p),size=p.stat().st_size,sha256=sha(p)))
    save(out/'artifact_manifest.json',dict(time_beijing=now(),scope='Immutable result/code and committed detail artifacts; live supervisor files excluded. Final delivery receipt separately verifies stopped process state.',artifacts=items))
    csvout(out/'artifact_manifest.csv',items)
    total=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    if total>50*1024**2:raise RuntimeError('Results directory exceeds50MiB')
    if any(p.stat().st_size>10*1024**2 for p in out.rglob('*') if p.is_file()):raise RuntimeError('Detail file>10MiB must remain in run')
    return dict(result_bytes=total,artifacts=len(items))
