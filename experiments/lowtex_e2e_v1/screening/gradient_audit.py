"""Compare retained smoke weights to their initialization, and isolate the coarse objective on a real pair."""
import argparse
import torch
from configs.cfg_outdoor_train import get_cfg_defaults
from .data import EXP,ROOT,Rows,gpu,read,save
from .model import BalancedLoss,CropHead,ScreenModel,components

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--run",required=True);args=parser.parse_args()
    d=(EXP/args.run).resolve()
    if EXP not in d.parents:raise ValueError(d)
    identity=read(d/"identity.json");variant=identity["exp"]
    if (d/"actual_gradients.json").exists():return
    saved=torch.load(d/"checkpoint_last.pt",map_location="cpu")
    base=torch.load(ROOT/"ckpt/megadepth_19epochs.ckpt",map_location="cpu")["state_dict"]
    cfg=get_cfg_defaults();initial=ScreenModel(cfg,False);initial.load_state_dict(base)
    torch.manual_seed(6166)
    if variant in ("crop","single","joint"):initial.crop=CropHead(variant!="single")
    elif variant=="rru_matched":
        for m in initial.recurrent_refine_unit.modules():
            if hasattr(m,"reset_parameters"):m.reset_parameters()
    changes={}
    for k,p in initial.named_parameters():
        changes[k]=float((p.detach()-saved["model"][k]).abs().max())
    def maximum(prefix):return max((v for k,v in changes.items() if k.startswith(prefix)),default=0.)
    result={"step":saved["step"],"coarse_temperature_delta":changes["coarse_temperature"],"backbone_delta":maximum("feature_backbone."),
            "head_delta":maximum("crop.") if initial.crop is not None else maximum("recurrent_refine_unit.")}
    assert result["backbone_delta"]>0 and result["head_delta"]>0
    if variant in ("single","crop","joint"):
        result["middle_projection_delta"]=maximum("crop.middle.")
        assert (result["middle_projection_delta"]>0)==(variant!="single")
        assert maximum("crop.high.")>0 and maximum("crop.regress.")>0
    if variant in ("exp1a","joint"):
        model,cfg,_,pol=components(variant,bool(identity["config"]["cp"]),2215669)
        model.load_state_dict(saved["model"]);pol.apply(pol.warmup);model.train()
        class CoarseOnlyProbe(BalancedLoss):
            def compute_coarse_loss(self,data):
                self.coarse=super().compute_coarse_loss(data)
                return self.coarse
        criterion=CoarseOnlyProbe(cfg.LOSS)
        sample=Rows(0,1)[0]
        _,stats=model.micro_loss(gpu([sample]),cfg,criterion)
        param=next(model.feature_backbone.parameters())
        g,t=torch.autograd.grad(criterion.coarse,(param,model.coarse_temperature))
        result.update(isolated_coarse_backbone_grad=float(g.float().norm()),isolated_coarse_temperature_grad=float(t.abs()),
                      geometry=stats[0]["balanced"],sample=sample[0])
        assert torch.isfinite(g).all() and torch.isfinite(t) and g.norm()>0 and t.abs()>0
    save(d/"actual_gradients.json",result);print(result)
if __name__=="__main__":main()
