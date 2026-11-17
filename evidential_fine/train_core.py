"""Only new-head parameters receive gradients; atomic state includes cursor and RNG."""
from .common import *
from .heads import EvidentialHead,nig_loss
from .native import prefix,inputs,supervision,selected_points,predict
import torch
from reliability_v1.runtime import save_checkpoint,load_checkpoint,rng_state,set_rng
def create(route,seed=2026):
    seed_all(seed);head=EvidentialHead(route).cuda().float()
    opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001,betas=(.9,.999),eps=1e-8)
    scaler=torch.cuda.amp.GradScaler(enabled=False)
    assert {id(p) for p in head.parameters()}=={id(p) for g in opt.param_groups for p in g['params']}
    return head,opt,scaler
def head_identity(run,route):
    return dict(frozen_identity=EXPECTED,route=route,protocol_sha=sha(Path(run)/'preregistered_protocol.json'),
        implementation={n:sha(ROOT/'evidential_fine'/n) for n in ['heads.py','native.py','train_core.py']})
def save_state(path,head,opt,scaler,state,ident,extra=None):
    return save_checkpoint(path,head,opt,state,ident,dict(scaler=scaler.state_dict(),**(extra or {})))
def restore(path,head,opt,scaler,ident):
    obj=load_checkpoint(path,head,opt,ident,restore_rng=True);scaler.load_state_dict(obj['extra']['scaler']);return obj['position']
def update(head,opt,scaler,features,target,rt=None):
    head.train();opt.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=False):
        out=head(features);loss,parts=nig_loss(out,target,head.R)
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite NIG loss')
    for k,v in out.items():
        if not torch.isfinite(v).all():raise FloatingPointError('Nonfinite new-head output '+k)
    scaler.scale(loss).backward();scaler.unscale_(opt)
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in head.parameters()):raise FloatingPointError('New-head gradient absent/nonfinite')
    if rt is not None and any(p.grad is not None or p.requires_grad for p in rt.model.parameters()):raise AssertionError('Original received gradient')
    grad_norm=float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad.float()) for p in head.parameters()])))
    scaler.step(opt);scaler.update()
    if any(not torch.isfinite(p).all() for p in head.parameters()):raise FloatingPointError('Nonfinite new parameter')
    row=dict(loss=float(loss.detach()),nll=float(parts['nll'].detach()),regularizer=float(parts['regularizer'].detach()),grad_norm=grad_norm,
        floor_rates=out['floor_rates'].detach().cpu().numpy().tolist(),samples=len(target),outputs_finite=True)
    opt.zero_grad(set_to_none=True);return row
def fixed_training_records():
    from reliability_data import ManifestDataset
    plan=read(SOURCE/'plans/epoch_00.json');data=ManifestDataset(SOURCE/'train_manifest.jsonl');rows=[];seen=set()
    try:
        for v in plan['visits']:
            if v['pair_id'] in seen:continue
            r=data[v['manifest_position']];assert r['pair_id']==v['pair_id'];rows.append(r);seen.add(v['pair_id'])
            if len(rows)==8:break
    finally:data.close()
    return rows
def selftest(run,route,rt):
    run=Path(run);out=run/'E0'/('selftest_'+route);out.mkdir(parents=True,exist_ok=True);recs=fixed_training_records();sets=[];selection=[]
    for r in recs:
        beat(run,'E0',phase='selftest input freeze',route=route,pair_id=r['pair_id'])
        d=rt.load(r);prefix(rt,d,route);s=supervision(d,route);ix=selected_points(s,route,0,r['pair_id'])
        features=tuple(x.detach().cpu() for x in inputs(rt,d,route,ix));target=torch.tensor(s['offset'][ix],dtype=torch.float32)
        value=dict(**base(r),valid_GT=int(s['valid'].sum()),inside=int(s['inside'].sum()),sampled=len(ix),parent_ids=s['parent'][ix].tolist(),child_ordinals=s['ids'][ix].tolist(),
            zero_EPE_mean=float(np.linalg.norm(s['offset'][ix],axis=1).mean()) if len(ix) else None)
        npzout(out/(pairname(r['pair_id'])+'.npz'),dict(offset=s['offset'][ix],parent=s['parent'][ix],ordinal=s['ids'][ix],**{f'feature{k}':v.numpy() for k,v in enumerate(features)}))
        selection.append(value);sets.append((r,features,target));del d,s;gc.collect()
    freeze=dict(status='frozen_before_updates',records=selection,pairs=8,sampling='same native scene_protocol parent rule; stable child analogue',criteria=dict(updates=200,last20_loss_lt_first20=True,final_fixed_mean_EPE_lt_zero=True,finite=True))
    save(out/'fixed_point_set.json',freeze)
    if not any(len(t) for _,f,t in sets):return dict(status='failed',reason='No fixed supervised points')
    head,opt,scaler=create(route,2027);initial=state_hash(head);ident=head_identity(run,route);rows=[];cursor=0;updates=0;last_features=None;last_target=None
    try:
        while updates<200:
            r,features,target=sets[cursor%8];cursor+=1
            if not len(target):continue
            features=tuple(x.cuda() for x in features);target=target.cuda()
            if updates==199:
                rt.check_frozen();save_state(out/'resume_199.json',head,opt,scaler,dict(updates=199,cursor=cursor-1),ident)
                last_features=features;last_target=target
            result=update(head,opt,scaler,features,target,rt);updates+=1;rows.append(dict(step=updates,pair_id=r['pair_id'],**result,resources=resource_sample()))
            append_json(out/'updates.jsonl',rows[-1]);beat(run,'E0',phase='selftest',route=route,updates=updates)
        continuous={k:v.detach().clone() for k,v in head.state_dict().items()};lastloss=rows[-1]['loss']
        head2,opt2,scaler2=create(route,2027);position=restore(out/'resume_199.json',head2,opt2,scaler2,ident)
        assert position['updates']==199 and position['cursor']==cursor-1
        rerun=update(head2,opt2,scaler2,last_features,last_target,rt)
        delta=max(float((continuous[k]-head2.state_dict()[k]).abs().max()) for k in continuous)
        resume=dict(status='pass' if delta<=1e-5 and abs(lastloss-rerun['loss'])<=1e-5 else 'failed',parameter_max_abs=delta,loss_abs=abs(lastloss-rerun['loss']),rng_and_cursor_restored=True,scaler_restored=True)
        del head2,opt2,scaler2,last_features,last_target
        head.eval();perpair=[];err=[];zero=[]
        for r,f,y in sets:
            if not len(y):continue
            pred=predict(head,tuple(x.cuda() for x in f));e=np.linalg.norm(pred['mu']-y.numpy(),axis=1);z=np.linalg.norm(y.numpy(),axis=1)
            perpair.append(dict(**base(r),count=len(e),mean_EPE=float(e.mean()),zero_mean_EPE=float(z.mean()),difference=float(e.mean()-z.mean())));err.extend(e.tolist());zero.extend(z.tolist())
        first=float(np.mean([r['loss'] for r in rows[:20]]));last=float(np.mean([r['loss'] for r in rows[-20:]]));changed=state_hash(head)!=initial
        result=dict(status='pass' if last<first and np.mean(err)<np.mean(zero) and changed and resume['status']=='pass' else 'failed',updates=200,first20_loss=first,last20_loss=last,loss_difference=last-first,
            final_fixed_mean_EPE=float(np.mean(err)),zero_mean_EPE=float(np.mean(zero)),EPE_difference=float(np.mean(err)-np.mean(zero)),parameters_changed=changed,resume=resume,
            per_pair=perpair,original_parameters_BN_unchanged=rt.check_frozen(),gradient_owner='new head only',optimizer_set='exactly new head',initial_head_sha=initial,final_head_sha=state_hash(head))
        save_state(out/'selftest_200.json',head,opt,scaler,dict(updates=200,cursor=cursor),ident)
        save(out/'summary.json',result);csvout(out/'learning_curve.csv',[{k:v for k,v in r.items() if k!='resources'} for r in rows]);return result
    finally:
        del head,opt,scaler,sets;gc.collect();torch.cuda.empty_cache()
