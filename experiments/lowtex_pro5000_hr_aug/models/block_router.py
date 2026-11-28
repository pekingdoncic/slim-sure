import torch


def block_ids(ids, width, ratio=4):
    return (ids//width//ratio)*(width//ratio)+(ids%width//ratio)


def locate(keys, queries):
    """Membership in strictly increasing, packed edge keys, including the empty case."""
    if not keys.numel():
        return torch.zeros_like(queries,dtype=torch.bool),torch.zeros_like(queries,dtype=torch.long)
    at = torch.searchsorted(keys,queries.to(keys.dtype)).clamp_max(len(keys)-1)
    return keys[at]==queries,at


def halo_membership(blocks,i,j,hw,halo):
    h,w=hw
    by,bx=j//w,j%w
    src=block_ids(i,w)
    covered=torch.zeros_like(i,dtype=torch.bool)
    for dy in (-halo,0,halo):
        for dx in (-halo,0,halo):
            y,x=(by+dy).clamp(0,h-1),(bx+dx).clamp(0,w-1)
            covered |= blocks[src,(y//4)*(w//4)+x//4]
    return covered


def gt_blocks(gt,w,nblocks):
    i,j = gt["i"],gt["j"]
    reverse_j = torch.where(gt["reverse"])[0]
    forward = block_ids(i,w)*nblocks+block_ids(j,w)
    reverse = block_ids(gt["back"][reverse_j],w)*nblocks+block_ids(reverse_j,w)
    return torch.cat((forward,reverse)).unique(sorted=True)


class BlockRouter:
    def __init__(self,cfg):
        self.cfg = cfg
        if cfg["block_stride"] != 32 or cfg["coarse_stride"] != 8:
            raise ValueError("native coarse/block strides are 8/32")

    @torch.no_grad()
    def __call__(self, scores, masks, coarse_masks, gt=None, injection=0., training=False, generator=None):
        if not training and injection:
            raise ValueError("GT injection during evaluation")
        a,b = masks
        m0,m1 = coarse_masks
        h,w = m0.shape
        nb = len(b)
        allowed = a[:,None]&b[None,:]
        logits = scores.detach().masked_fill(~allowed,-torch.inf)
        k = self.cfg["router_topk"]
        predicted = torch.zeros_like(allowed)
        predicted.scatter_(1,logits.topk(min(k,nb),dim=1).indices,True)
        if self.cfg["bidirectional_routing"]:
            predicted.scatter_(0,logits.topk(min(k,len(a)),dim=0).indices,True)
        predicted &= allowed
        chosen = predicted.clone()
        truth = gt_blocks(gt,w,nb) if gt is not None else torch.empty(0,dtype=torch.long,device=a.device)
        missing = truth[~predicted.flatten()[truth]]
        if training and injection and missing.numel():
            add = missing[torch.rand(len(missing),device=a.device,generator=generator)<injection]
            chosen.view(-1)[add] = True
        else:
            add = missing[:0]
        halo = int(self.cfg["halo_radius_coarse_grid"])
        if not 0<=halo<=4:
            raise ValueError("halo must be bounded by one block")
        def offsets(radius):
            t = torch.arange(-radius,4+radius,device=a.device)
            y,x = torch.meshgrid(t,t,indexing="ij")
            return y.flatten(),x.flatten()
        def expand(block,delta,mask):
            y = (block[:,None]//(w//4))*4+delta[0]
            x = (block[:,None]%(w//4))*4+delta[1]
            index = (y*w+x).clamp(0,h*w-1)
            valid = (y>=0)&(x>=0)&(y<h)&(x<w)&mask.flatten()[index]
            return index,valid
        edges = []
        pre = 0
        src_delta,dst_delta = offsets(0),offsets(halo)
        for pairs in chosen.nonzero().split(self.cfg["route_block_chunk"]):
            src,src_ok = expand(pairs[:,0],src_delta,m0)
            dst,dst_ok = expand(pairs[:,1],dst_delta,m1)
            valid = src_ok[:,:,None]&dst_ok[:,None,:]
            keys = (src[:,:,None]*(h*w)+dst[:,None,:])[valid]
            pre += len(keys)
            edges.append(keys.to(torch.int32))
        keys = torch.cat(edges).unique(sorted=True) if edges else torch.empty(0,dtype=torch.int32,device=a.device)
        total = len(keys)
        budget = self.cfg["edge_budget_per_pair"]
        if total>budget:
            if self.cfg["edge_budget_policy"] != "truncate":
                raise RuntimeError(f"BBR edges {total} exceed explicit budget {budget} (raw {pre})")
            # Round-robin over source rows; deterministic target-ID order within each row.
            row = keys//(h*w)
            counts = torch.bincount(row.long(),minlength=h*w)
            local = torch.arange(total,device=a.device)-(counts.cumsum(0)-counts)[row.long()]
            keys = keys[local.argsort(stable=True)[:budget]].sort().values
        stats = dict(raw_edges=pre,deduplicated_edges=total,edges=len(keys),truncated=total-len(keys),
                     gt_blocks=len(truth),injected_blocks=len(add),missing_blocks=len(missing))
        if gt is not None:
            i,j = gt["i"],gt["j"]
            src_block = block_ids(i,w)
            cover = halo_membership(predicted,i,j,(h,w),halo)
            supplied = halo_membership(chosen,i,j,(h,w),halo)
            valid_end = m0.flatten()[i]&m1.flatten()[j]
            retained,_ = locate(keys,i*(h*w)+j)
            route = predicted[src_block,block_ids(j,w)]
            stats.update(route_hits=int(route.sum()),predicted_halo_hits=int((cover&valid_end).sum()),
                         route_den=len(i),retained_gt=int(retained.sum()),
                         truncated_gt=int((supplied&valid_end&~retained).sum()))
        return keys,truth,stats,predicted
