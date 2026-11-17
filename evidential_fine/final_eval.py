"""Fixed epoch2 weights, full-forward100 evaluation, then real replacement timing."""
from .common import *
from .train_core import create,head_identity
from .eval_pair import evaluate_pair
from reliability_v1.runtime import load_checkpoint
import torch
def execute(run):
    run=Path(run);identity_gate();out=run/'evaluate';out.mkdir(exist_ok=True);rt=NativeRuntime(out/'runtime');heads={};summaries={}
    try:
        for route in ['A','B']:
            p=run/route/'training_summary.json'
            if not p.exists() or read(p)['status']!='committed':
                summaries[route]=dict(status='not_executed',reason='No complete epoch2 checkpoint');continue
            head,opt,scaler=create(route,2026);del opt,scaler
            pointer=run/route/'epoch_01.json';obj=load_checkpoint(pointer,head,None,head_identity(run,route),restore_rng=False)
            if obj['position'].get('completed_epoch')!=2:raise AssertionError('Final evaluation must use epoch2')
            head.eval();heads[route]=head;rows=[]
            for i,r in enumerate(records('epe')):
                row=get_unit(run,'final_'+route,r)
                if row is None:
                    result,a=evaluate_pair(rt,head,r);commit(run,'final_'+route,r,result,a);row=result;del a
                rows.append(row);beat(run,'evaluate',phase='full evaluation',route=route,pairs=i+1,total=100,resources=resource_sample())
            summaries[route]=dict(status='committed',pairs=100,head_sha=state_hash(head),checkpoint_pointer=str(pointer),checkpoint_sha=sha(pointer),frozen=rt.check_frozen())
            save(run/('final_'+route)/'summary.json',summaries[route])
        from .efficiency import measure
        efficiency=measure(run,rt,heads) if heads else dict(status='not_executed')
        result=dict(status='completed' if heads else 'partial',routes=summaries,efficiency=efficiency,time_beijing=now());save(out/'summary.json',result);return result
    finally:
        save(out/'frozen_check.json',dict(unchanged=rt.check_frozen(),time_beijing=now()));del rt,heads;gc.collect();torch.cuda.empty_cache()
