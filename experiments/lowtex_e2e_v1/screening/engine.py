import time
import torch
from .data import gpu

def update(iterator,n,bs,model,cfg,objective,policy,step,check_grad=False):
    policy.apply(step)
    opt=policy.optimizer
    opt.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    durations=dict(load_s=0.,forward_s=0.,backward_s=0.)
    samples=[]
    for _ in range(0,n,bs):
        t=time.perf_counter()
        batch=gpu(next(iterator))
        torch.cuda.synchronize()
        durations["load_s"]+=time.perf_counter()-t
        t=time.perf_counter()
        loss,records=model.micro_loss(batch,cfg,objective)
        if not torch.isfinite(loss):raise FloatingPointError(records)
        torch.cuda.synchronize();durations["forward_s"]+=time.perf_counter()-t
        t=time.perf_counter()
        (loss/n).backward()
        torch.cuda.synchronize();durations["backward_s"]+=time.perf_counter()-t
        samples.extend(records)
    if len(samples)!=n:raise ValueError("effective batch or tail mismatch")
    expected=[]
    if check_grad:
        targets=[("backbone",model.feature_backbone)] if policy.head is None or step>=policy.warmup else []
        if policy.head is not None:targets.append(("head",policy.head))
        for name,mod in targets:
            par=next((p for p in mod.parameters() if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0),None)
            if par is None:raise RuntimeError(f"no gradient reaching {name}")
            expected.append((name,par,par.detach().clone(),float(par.grad.norm())))
        if step<policy.warmup and any(p.grad is not None for p in model.feature_backbone.parameters()):raise ValueError("frozen backbone acquired gradient")
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    opt.step();policy.scheduler.step();torch.cuda.synchronize()
    gradients={}
    for name,par,before,g in expected:
        delta=float((before-par.detach()).abs().max())
        if delta==0:raise RuntimeError(f"{name} parameter did not update")
        gradients[name]=[g,delta]
    return dict(step=step+1,seconds=time.perf_counter()-start,grad_norm=float(norm),gradient_checks=gradients,
                allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved(),
                min_free_now=torch.cuda.mem_get_info()[0],samples=samples,**durations)
