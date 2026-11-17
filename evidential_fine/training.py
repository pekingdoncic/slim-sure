"""Two fixed scene-balanced epochs, head-only updates and exact checkpoint cursor."""
from .common import *
from .train_core import create,head_identity,save_state,restore,update
from .native import prefix,inputs,supervision,selected_points
from .eval_pair import validate
from reliability_data import ManifestDataset
from reliability_v1.scene_protocol import skip_guard
import torch
class SafePause(RuntimeError):pass
def reconcile_blocks(out,state):
    """Only checkpoint-covered contiguous visits are committed; preserve orphan files."""
    out=Path(out);limit=int(state['visits']);cursor=0;updates=0;orphan=[];kept=0
    for path in sorted((out/'blocks').glob('epoch_*_pos_*.json')):
        rows=read(path)
        if not rows:raise AssertionError('Empty training block')
        positions=[int(row['epoch'])*13800+int(row['position']) for row in rows]
        if positions!=list(range(positions[0],positions[-1]+1)):raise AssertionError('Noncontiguous block positions')
        if positions[0]>=limit:
            orphan.append(path);continue
        if positions[-1]>=limit:raise AssertionError('Checkpoint splits a supposedly atomic block')
        if positions[0]!=cursor:raise AssertionError('Committed block has gap or duplicate')
        cursor=positions[-1]+1;updates+=sum(bool(row['updated']) for row in rows);kept+=1
    if cursor!=limit or updates!=int(state['updates']):raise AssertionError('Checkpoint and committed training journal disagree')
    moved=[]
    if orphan:
        dest=out/'recovery'/('uncommitted_'+str(time.time_ns()));dest.mkdir(parents=True)
        for path in orphan:
            target=dest/path.name
            if not target.resolve().is_relative_to(out.resolve()):raise AssertionError('Orphan destination escaped route directory')
            moved.append(dict(original=str(path),preserved=str(target),sha256=sha(path)));path.rename(target)
    record=dict(time_beijing=now(),committed_visits=limit,committed_updates=updates,committed_blocks=kept,uncommitted_preserved=moved)
    save(out/'recovery'/('reconcile_'+str(time.time_ns())+'.json'),record)
    return record

def train(run,route,resume=False):
    run=Path(run);out=run/route;out.mkdir(exist_ok=True);identity_gate()
    if read(run/'E0/summary.json')['routes'][route]['status']!='pass':raise RuntimeError('Route E0 did not pass')
    rt=NativeRuntime(out/'runtime');head,opt,scaler=create(route,2026);ident=head_identity(run,route);dataset=ManifestDataset(SOURCE/'train_manifest.jsonl')
    state=dict(epoch=0,next_pos=0,phase='train',updates=0,visits=0,guard={},route=route,epoch_updates=[0,0],epoch_visits=[0,0],epoch_seconds=[0.,0.])
    pointer=out/'last.json';initial=state_hash(head);started=time.monotonic();chunk=[];recent=[]
    if resume:
        if not pointer.exists():raise RuntimeError('No valid checkpoint for resume')
        state=restore(pointer,head,opt,scaler,ident)
        reconcile_blocks(out,state)
        save(out/('resume_'+str(time.time_ns())+'.json'),dict(state=state,last_pointer_sha=sha(pointer),time_beijing=now()))
    else:
        if pointer.exists():raise RuntimeError('Existing progress; resume required')
        assert state['next_pos']==0 and state['updates']==0
        save(out/'formal_initialization.json',dict(seed=2026,cursor=0,updates=0,head_sha=initial,selftest_head_sha=read(run/'E0'/('selftest_'+route)/'summary.json')['final_head_sha'],
            optimizer_state_empty=len(opt.state)==0,scaler_state=scaler.state_dict(),parameter_count=sum(p.numel() for p in head.parameters()),identity=ident,time_beijing=now()))
        save_state(pointer,head,opt,scaler,state,ident)
    def commit_chunk():
        if chunk:
            name=f"epoch_{chunk[0]['epoch']:02d}_pos_{chunk[0]['position']:06d}_{chunk[-1]['position']:06d}.json"
            path=out/'blocks'/name
            if path.exists() and read(path)!=chunk:raise AssertionError('Training block identity collision')
            if not path.exists():save(path,chunk)
            extra=dict(block=str(path),block_sha=sha(path),committed_visits=state['visits'])
        else:extra={}
        rt.check_frozen();save_state(pointer,head,opt,scaler,state,ident,extra);chunk.clear()
        save(out/'progress.json',dict(state=state,last_checkpoint=read(pointer),time_beijing=now(),resources=resource_sample(),elapsed_this_process=time.monotonic()-started))
    try:
        for epoch in range(state['epoch'],2):
            planpath=SOURCE/'plans'/f'epoch_{epoch:02d}.json';plan=read(planpath)
            expected=read(run/'preregistered_protocol.json')['plans'][f'epoch_{epoch:02d}']
            if sha(planpath)!=expected or plan['content_sha256']!=digest({k:v for k,v in plan.items() if k!='content_sha256'}):raise IdentityFailure('Frozen plan changed')
            if len(plan['visits'])!=13800:raise IdentityFailure('Planned epoch visits changed')
            if state['phase']=='train':
                for pos in range(state['next_pos'],len(plan['visits'])):
                    v=plan['visits'][pos];r=dataset[v['manifest_position']]
                    if r['pair_id']!=v['pair_id']:raise AssertionError('Sampling restore cursor mismatch')
                    if (out/'pause_requested').exists():commit_chunk();raise SafePause((out/'pause_requested').read_text())
                    beat(run,route,phase='train',epoch=epoch+1,next_position=pos,pair_id=r['pair_id'],updates=state['updates'])
                    start=time.perf_counter();d=rt.load(r);prefix(rt,d,route);s=supervision(d,route);ix=selected_points(s,route,epoch,r['pair_id'])
                    if len(ix):
                        features=inputs(rt,d,route,ix);target=torch.tensor(s['offset'][ix],dtype=torch.float32,device='cuda')
                        result=update(head,opt,scaler,features,target,rt);result.update(updated=True,reason=None,selection_sha=digest(dict(parent=s['parent'][ix].tolist(),ordinal=s['ids'][ix].tolist())))
                        del features,target
                    else:result=dict(updated=False,reason='no_supervision',samples=0)
                    result.update(valid_GT=int(s['valid'].sum()),inside=int(s['inside'].sum()),outside=int((s['valid']&~s['inside']).sum()),predicted=len(s['ids']))
                    del d,s;elapsed=time.perf_counter()-start
                    state['guard']=skip_guard(state['guard'],result['reason']);state['updates']+=int(result['updated']);state['visits']+=1;state['next_pos']=pos+1
                    state['epoch_updates'][epoch]+=int(result['updated']);state['epoch_visits'][epoch]+=1;state['epoch_seconds'][epoch]+=elapsed
                    row=dict(epoch=epoch,position=pos,pair_id=r['pair_id'],scene=r['scene'],seconds=elapsed,**result,resources=resource_sample())
                    append_json(out/'physical_visits.jsonl',row);chunk.append(row);recent.append(elapsed);recent=recent[-100:]
                    if state['visits']==50 or state['visits']%1000==0:
                        mean=float(np.mean(recent));remaining=27600-state['visits']
                        other=27600 if route=='A' else 0
                        prior=read(run/'A/first50_eta.json')['seconds_per_visit'] if route=='B' and (run/'A/first50_eta.json').exists() else mean
                        validation_extra=3000*max(.3,mean) if route=='A' else 1500*max(.3,mean)
                        rest=remaining*mean+other*mean+validation_extra+1800
                        eta=dict(time_beijing=now(),visits=state['visits'],updates=state['updates'],skips=state['guard'].get('skip_reasons',{}),
                            seconds_per_visit=mean,pairs_per_second=1/mean,remaining_this_route_visits=remaining,remaining_other_route_visits=other,
                            training_remaining_seconds=remaining*mean+other*mean,validation_allowance_seconds=validation_extra,E0_final_geometry_report_allowance_seconds=1800,total_remaining_seconds=rest,
                            estimate_basis='actual complete visit load+prefix+GT+head backward; other route uses this proxy until own first50; validation conservative max(0.3,visit cost); final evaluation/geometry/report allowance30min',
                            exceeds_16_hours=rest>16*3600,resources=resource_sample())
                        save(out/('first50_eta.json' if state['visits']==50 else 'eta.json'),eta)
                        if rest>16*3600 and not (run/'degradation.json').exists():
                            save(run/'degradation.json',dict(reason='measured total >16h',time_beijing=now(),skipped=['Phase5 pose','A efficiency timing'],T_optional_items_already_executed=True,main_training_unchanged=True,estimate=eta))
                        print(json.dumps(dict(event='ETA',**eta)),flush=True)
                    if len(chunk)>=50 or state['guard']['pause']:commit_chunk();gc.collect()
                    if state['guard']['pause']:raise SafePause('Frozen first100 skip guard >20; no resampling')
                state['phase']='validation';commit_chunk()
            if state['phase']=='validation':
                summary=validate(run,route,epoch,rt,head)
                save_state(out/f'epoch_{epoch:02d}.json',head,opt,scaler,dict(state,completed_epoch=epoch+1),ident,dict(validation_sha=sha(out/f'validation_{epoch:02d}/summary.json')))
                state.update(epoch=epoch+1,next_pos=0,phase='train');commit_chunk()
        final=dict(status='committed',epochs=2,visits=state['visits'],updates=state['updates'],skip=state['guard'].get('skip_reasons',{}),
            epoch_updates=state['epoch_updates'],epoch_visits=state['epoch_visits'],epoch_training_seconds=state['epoch_seconds'],
            head_sha=state_hash(head),last=read(pointer),final_epoch_pointer=str(out/'epoch_01.json'),frozen=rt.check_frozen(),seconds_this_process=time.monotonic()-started,time_beijing=now())
        assert state['visits']==27600 and state['epoch_visits']==[13800,13800]
        save(out/'training_summary.json',final);return final
    except BaseException as e:
        save(out/('training_failure_'+str(time.time_ns())+'.json'),dict(reason=repr(e),last_valid_pointer=str(pointer),state_in_memory=state,
            last_valid_pointer_sha=sha(pointer) if pointer.exists() else None,uncommitted_rows=len(chunk),
            resume_command=f'/home/bupt/miniconda3/envs/slim/bin/python -B evidential_fine/run_phase5.py --run {run} --stage {route} --background --resume',time_beijing=now()))
        raise
    finally:
        dataset.close();save(out/'frozen_exit.json',dict(unchanged=rt.check_frozen(),time_beijing=now()))
        del rt,head,opt,scaler;gc.collect();torch.cuda.empty_cache()
