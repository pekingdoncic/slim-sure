"""Compose native feature extraction, norm matching, fine correlation and RRU with optional CCS."""
from contextlib import contextmanager
from time import perf_counter
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from configs.cfg_outdoor_test import get_cfg_defaults
from src.slim import SLiM
from src.loss import SLiM_Loss
from src.utils.supervision import spvs_intermediate, compute_supervision_fine
from .ccfi import CCFI
from .block_router import BlockRouter
from .hypothesis_graph import HypothesisGraph
from .sparse_correlation import edge_logits
from .sparse_dual_softmax import dual_logp, select_mnn
from ..data.supervision_adapter import coarse_truth
from ..losses.routing_loss import routing_loss
from ..losses.sparse_matching_loss import native_focal, sparse_focal
from ..losses.hard_negative_ranking import ranking


@contextmanager
def timed(record, key):
    if record is None:
        yield
        return
    torch.cuda.synchronize()
    start = perf_counter()
    yield
    torch.cuda.synchronize()
    record[key] = (perf_counter()-start)*1000


@contextmanager
def fine_rng(seed, device):
    with torch.random.fork_rng(devices=[device]):
        torch.cuda.manual_seed(seed)
        yield


class CCSMatcher(nn.Module):
    def __init__(self,cfg,weights):
        super().__init__()
        self.cfg = cfg
        self.native_cfg = get_cfg_defaults()
        self.core = SLiM(self.native_cfg.MODEL)
        state = torch.load(weights,map_location="cpu",weights_only=False)
        self.core.load_state_dict(dict(state["state_dict"]),strict=True)
        self.core.sparse_direct = cfg["native_skip_unused_matrix"]
        self.loss_fn = SLiM_Loss(self.native_cfg.LOSS)
        c = self.core.d_coarse
        self.ccfi = CCFI(c,cfg) if cfg["ccfi"] else None
        self.router = BlockRouter(cfg) if cfg["bbr"] else None
        self.graph = HypothesisGraph(c,cfg) if cfg["lhg"] else None
        self.checkpointing = "off"

    def train(self,mode=True):
        super().train(mode)
        if mode and self.cfg["freeze_backbone_bn_stats"]:
            for m in self.core.feature_backbone.modules():
                if isinstance(m,nn.modules.batchnorm._BatchNorm):
                    m.eval()
        return self

    def features(self,d,times=None):
        core = self.core
        def extract(a,b):
            xa,xb = core.feature_backbone(a,b)
            return xa[core.coarse_scale_idx],xb[core.coarse_scale_idx],xa[core.fine_scale_idx],xb[core.fine_scale_idx]
        def call(fn,*args,full=False):
            active = self.training and self.checkpointing in (("full",) if full else ("full","selective"))
            return checkpoint(fn,*args,use_reentrant=False) if active else fn(*args)
        with timed(times,"Backbone"):
            a,b,fa,fb = call(extract,d["image0"],d["image1"])
        with timed(times,"Fine Upsample"):
            fa,fb = call(core.fine_upsample,fa,fb,full=True)
        context = ()
        if self.ccfi is not None:
            with timed(times,"CCFI"):
                a,b,*context = call(self.ccfi,a,b,d["mask0"],d["mask1"],full=True)
        return (a,b,fa,fb,*context)

    def sparse(self,d,features,gt,prob,rng,times=None):
        a,b,_,_,s,m0,m1 = features
        feat = [x[0].flatten(1).T.contiguous() for x in (a,b)]
        masks = [d["mask0"][0],d["mask1"][0]]
        if self.cfg["norm_filter"]:
            for k in range(2):
                norm = feat[k].float().norm(dim=-1).view_as(masks[k]).masked_fill(~masks[k],0)
                masks[k] = masks[k] & (norm>norm.max()*.65)
        with timed(times,"BBR"):
            keys,truth,stats,pred = self.router(s[0],(m0[0],m1[0]),masks,gt,prob,self.training,rng)
        a,b = feat
        with timed(times,"Sparse Correlation"):
            base = edge_logits(a,b,keys,self.core.coarse_temperature,self.cfg["correlation_chunk"])
        scores = base
        raw = table = None
        if self.graph is not None:
            with timed(times,"LHG"):
                scores,raw,table,added = self.graph(a,b,keys,base,s[0],masks[0].shape,gt,prob,rng)
            stats["injected_hypotheses"] = added
        with timed(times,"Sparse Dual-Softmax"):
            logp = dual_logp(scores,keys,len(a),len(b))
        d.update(keys=keys,base_scores=base,updated_scores=scores,logp=logp,raw_top=raw,
                 top=table,route_pred=pred,route_stats=stats,route_masks=masks)
        if gt is None:
            return
        coarse,npos = sparse_focal(logp,keys,gt,len(b),self.loss_fn.coarse_gamma)
        router = routing_loss(s[0],m0[0],m1[0],truth)
        terms = {"coarse":coarse,"route":router}
        stats["coarse_positives"] = npos
        if table is not None:
            terms["rank"],stats["rank_negatives"] = ranking(scores,keys,table,gt,masks[0].shape,self.cfg)
            if len(keys):
                from .block_router import locate
                ok,at = locate(keys,gt["i"]*len(b)+gt["j"])
                stats["topm_predicted_hits"] = int((ok & (raw[gt["i"]]==at[:,None]).any(1)).sum())
        return terms

    def prime(self,d,features):
        a,b,fa,fb = features[:4]
        d.update(batch_size=1, hw0_i=d["image0"].shape[-2:],hw1_i=d["image1"].shape[-2:],
                 feat0_c=a,feat1_c=b,feat0_f=fa,feat1_f=fb,hw0_c=a.shape[-2:],hw1_c=b.shape[-2:],
                 hw0_f=fa.shape[-2:],hw1_f=fb.shape[-2:],coarse_scale=8,fine_scale=1,refine_iters=self.core.refine_iters)

    def forward(self,d,diagnostics=False,times=None):
        if self.training:
            raise RuntimeError("inference requires eval(); training uses training_loss")
        if d["image0"].shape[0]!=1:
            raise ValueError("evaluation protocol uses one pair")
        self.core.capture_reliability = diagnostics
        if self.ccfi is None:
            self.core(d,training=False)
            if times is not None:
                for k,v in (("Backbone","feat_extract_time"),("Fine Upsample","upsample_time"),
                            ("Native Coarse","coarse_match_time"),("Fine Matching","fine_match_time"),("RRU","refine_time")):
                    times[k] = 1000*d[v]
            return d
        features = self.features(d,times)
        self.prime(d,features)
        if self.router is None:
            with timed(times,"Native Coarse"):
                self.core._coarse_match_norm_clue(d,*features[:2],d["mask0"].flatten(1),d["mask1"].flatten(1))
        else:
            self.sparse(d,features,None,0.,None,times)
            n = d["mask0"].numel()
            sel = select_mnn(d["logp"],d["keys"],n,n,self.core.coarse_match_thres,self.core.max_coarse_matches)
            k = d["keys"][sel].long()
            d.update(spv_b_ids=torch.zeros_like(k),spv_i_ids=k//n,spv_j_ids=k%n)
            self.core._get_coarse_coord_train(d)
        with timed(times,"Fine Matching"):
            self.core._fine_correlation(d)
            self.core._get_fine_coord_test(d)
        with timed(times,"RRU"):
            self.core._refinement(d)
        return d

    def pair_loss(self,d,features,injection,position):
        self.prime(d,features)
        with torch.autocast(device_type=features[0].device.type,enabled=False):
            gt = coarse_truth(d,self.cfg)
        device = features[0].device
        rng = torch.Generator(device=device).manual_seed(self.cfg["seed_fine"]+position)
        coarse_rng = torch.Generator(device=device).manual_seed(47591+position)
        route_rng = torch.Generator(device=device).manual_seed(92831+position)
        if self.router is not None:
            terms = self.sparse(d,features,gt,injection,route_rng)
            stats = d["route_stats"]
        else:
            a,b = [f[0].flatten(1).T for f in features[:2]]
            terms = dict(coarse=native_focal(a,b,d["mask0"].flatten(),d["mask1"].flatten(),gt["native_i"],
                       gt["native_j"],self.core.coarse_temperature,self.loss_fn,self.cfg["dense_loss_chunk"],coarse_rng))
            stats = {"coarse_positives":int(len(gt["native_i"])*self.loss_fn.coarse_percent)}
        i,j = gt["native_i"],gt["native_j"]
        cap = self.cfg["max_fine_train_matches"]
        idx = torch.randperm(len(i),generator=rng,device=device)[:cap].sort().values
        d.update(spv_b_ids=torch.zeros_like(i[idx]),spv_i_ids=i[idx],spv_j_ids=j[idx],dataset_name=["MegaDepth"])
        stats.update(fine_parents=len(idx),valid_geometry=len(gt["i"]),fine_positives=0,refine_positives=0)
        if len(idx):
            with torch.autocast(device_type=device.type,enabled=False):
                spvs_intermediate(d,self.native_cfg)
            if len(d["spv_b_ids_it"]):
                self.core._get_coarse_coord_train(d)
                self.core._fine_correlation(d)
                prev = self.core.max_fine_matches
                try:
                    self.core.max_fine_matches = min(cap,prev)
                    with fine_rng(self.cfg["seed_fine"]+position,device):
                        self.core._get_fine_coord_train(d)
                    if self.checkpointing=="full":
                        # Keep checkpoint recomputation independent of the restored inference cap.
                        refine_inputs={k:d[k] for k in ("b_idx_it","scale0","scale1","fine_scale","intermediate_coord_0","intermediate_coord_1")}
                        def refine(fa,fb):
                            prior = self.core.max_fine_matches
                            try:
                                self.core.max_fine_matches = min(cap,prior)
                                local = dict(refine_inputs,feat0_f=fa,feat1_f=fb)
                                self.core._refinement(local)
                                return local["fine_coord_0"],local["fine_coord_1"],local["all_offset_1"]
                            finally:
                                self.core.max_fine_matches = prior
                        d["fine_coord_0"],d["fine_coord_1"],d["all_offset_1"] = checkpoint(
                            refine,d["feat0_f"],d["feat1_f"],use_reentrant=False)
                    else:
                        self.core._refinement(d)
                finally:
                    self.core.max_fine_matches = prev
                with torch.autocast(device_type=device.type,enabled=False):
                    compute_supervision_fine(d,self.native_cfg)
                    f = d["sim_matrix_f"].float()
                    d["conf_matrix_f"] = f.softmax(1)*f.softmax(2)
                    d["all_offset_1"] = d["all_offset_1"].float()
                    terms["fine"] = self.loss_fn.compute_fine_loss(d)
                    rf = self.loss_fn.compute_refine_loss(d)
                    if rf is not None:
                        terms["refine"] = rf
                stats.update(fine_positives=int(d["conf_matrix_f_gt"].sum()),refine_positives=int(d["correct_mask"].sum()))
        weights = dict(coarse=self.loss_fn.coarse_weight,fine=self.loss_fn.fine_weight,
                       refine=self.loss_fn.refine_weight,route=self.cfg["route_loss_weight"],rank=self.cfg["rank_loss_weight"])
        total = sum(v*weights[k] for k,v in terms.items())
        stats.update({k+"_loss":float(v.detach()) for k,v in terms.items()})
        return total,stats

    def training_loss(self,d,injection):
        features = self.features(d)
        losses,stats = [],[]
        for b,pos in enumerate(d["position"].tolist()):
            one = {k:v[b:b+1] for k,v in d.items() if isinstance(v,torch.Tensor)}
            value,record = self.pair_loss(one,tuple(x[b:b+1] for x in features),injection,pos)
            losses.append(value)
            stats.append(record)
        return torch.stack(losses).mean(),stats
