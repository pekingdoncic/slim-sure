import json
import torch
from .data import N,OUT,Rows,gpu,save
from .engine import update
from .model import components

def main():
    torch.set_num_threads(2)
    model,cfg,loss,pol=components("exp0",False,2215669)
    model.train()
    data=Rows(0,2)
    pair=gpu([data[0],data[1]])
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
        joined=model.compute_features(*[torch.cat([d["image"+str(v)] for _,d in pair]) for v in (0,1)])
        errors=[]
        for i,(_,d) in enumerate(pair):
            separate=model.compute_features(d["image0"],d["image1"])
            for v in (0,1):
                for level,x in enumerate(separate[v]):
                    y=joined[v][level][i:i+1]
                    errors.append(float((y.float()-x.float()).norm()/x.float().norm().clamp_min(1e-7)))
    def probe(size):
        model.zero_grad(set_to_none=True);torch.manual_seed(66);torch.cuda.manual_seed_all(66)
        losses=[];counts=[]
        for i in range(0,2,size):
            pairs=gpu([data[j] for j in range(i,i+size)])
            val,stats=model.micro_loss(pairs,cfg,loss)
            (val/2).backward()
            losses.extend(s["loss"] for s in stats)
            counts.extend([(s["parents_after"],s["children_after"],s["valid_gt"]) for s in stats])
        return losses,counts,float(model.coarse_temperature.grad)
    a=probe(1);b=probe(2)
    loss_deltas=[abs(x-y)/max(abs(x),1e-8) for x,y in zip(a[0],b[0])]
    result=dict(relative_pyramid_errors=errors,loss_relative_differences=loss_deltas,single=a,batched=b)
    result["pass"]=max(errors)<.05 and max(loss_deltas)<.05 and a[1]==b[1] and all(torch.isfinite(torch.tensor([a[2],b[2]])))
    tail=Rows(N-1,N)
    final=update(iter([[tail[0]]]),1,2,model,cfg,loss,pol,0,check_grad=True)
    assert len(final["samples"])==1 and final["samples"][0]["identity"][0]==N-1
    result["last_partial_batch"]=dict(real_pairs=1,position=N-1,gradient_checks=final["gradient_checks"])
    save(OUT/"batch_equivalence.json",result)
    print(json.dumps(result))
    if not result["pass"]:raise RuntimeError("B2 pair composition or numerical agreement failed")
if __name__=="__main__":main()
