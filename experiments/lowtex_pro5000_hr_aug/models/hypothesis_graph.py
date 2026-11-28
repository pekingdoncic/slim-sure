import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .block_router import block_ids, locate


def row_z(x,row,n):
    count = torch.bincount(row,minlength=n).clamp_min(1)
    mean = x.new_zeros(n).scatter_add_(0,row,x)/count
    centered = x-mean[row]
    var = x.new_zeros(n).scatter_add_(0,row,centered.square())/count
    return centered/var[row].clamp_min(1e-8).sqrt()


@torch.no_grad()
def top_table(scores,row,n,m):
    """Bounded per-row selection; ties use sorted raw target ID."""
    table = torch.full((n,m),-1,dtype=torch.long,device=scores.device)
    work = scores.detach().float().clone()
    ids = torch.arange(len(work),device=work.device)
    for slot in range(m):
        peak = work.new_full((n,),-torch.inf).scatter_reduce_(0,row,work,reduce="amax")
        winners = torch.full_like(peak,len(work),dtype=torch.long)
        winners.scatter_reduce_(0,row,torch.where(work==peak[row],ids,len(work)),reduce="amin")
        live = peak.isfinite()
        table[live,slot] = winners[live]
        work[winners[live]] = -torch.inf
    return table


class HypothesisGraph(nn.Module):
    def __init__(self, channels, cfg):
        super().__init__()
        self.cfg = cfg
        dim = cfg["hypothesis_feature_dim"]
        self.src,self.dst = nn.Linear(channels,dim),nn.Linear(channels,dim)
        self.embed = nn.Sequential(nn.Linear(2*dim+6,dim),nn.LayerNorm(dim),nn.GELU())
        self.layers = nn.ModuleList(nn.Sequential(nn.Linear(2*dim,dim),nn.LayerNorm(dim),nn.GELU())
                                    for _ in range(cfg["graph_layers"]))
        self.out = nn.Linear(dim,1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, a,b,keys,scores,blocks,hw,gt=None,injection=0.,generator=None):
        if injection and not self.training:
            raise ValueError("evaluation cannot add GT hypotheses")
        h,w = hw
        n = h*w
        row,col = (keys//n).long(),(keys%n).long()
        srcblock,dstblock = block_ids(row,w),block_ids(col,w)
        with torch.no_grad():
            assoc = blocks.detach()[srcblock,dstblock]
            fused = row_z(scores.detach(),row,n)+self.cfg["block_selection_weight"]*row_z(assoc,row,n)
            m = self.cfg["hypothesis_topm"]
            table = top_table(fused,row,n,m)
            raw = table.clone()
            added = 0
            if gt is not None and injection>0:
                i,j = gt["i"],gt["j"]
                ok,at = locate(keys,i*n+j)
                need = ok & ~(table[i]==at[:,None]).any(1)
                need &= torch.rand(len(i),device=i.device,generator=generator)<injection
                table[i[need],-1] = at[need]
                added = int(need.sum())
        if not len(keys):
            return scores,raw,table,added
        live = table>=0
        ids = table[live]
        i,j = row[ids],col[ids]
        lookup = torch.full_like(table,-1)
        lookup[live] = torch.arange(len(ids),device=i.device)
        def xy(x):
            return torch.stack((x%w,x//w),1).float()
        # Margin/entropy of the local predicted distribution, before scheduled GT replacement.
        selected = fused[raw.clamp_min(0)].masked_fill(raw<0,-torch.inf)
        delta = selected[:,0]-selected[:,min(1,m-1)]
        margin = torch.where(delta.isfinite(),delta,0)
        probs = selected.masked_fill(~(raw>=0).any(1)[:,None],0).softmax(1)
        entropy = -(probs*probs.clamp_min(1e-30).log()).sum(1)
        info = torch.cat((scores[ids,None],blocks[srcblock[ids],dstblock[ids],None],
                          (xy(j)-xy(i))/max(h,w),margin[i,None],entropy[i,None]),1)
        hidden = self.embed(torch.cat((self.src(a)[i],self.dst(b)[j],info),1))
        offsets = [(0,-1),(0,1),(-1,0),(1,0),(-1,-1),(-1,1),(1,-1),(1,1)]
        degree = self.cfg["source_neighbor_degree"]
        if not 1<=degree<=len(offsets) or self.cfg["graph_max_degree"]<1:
            raise ValueError("invalid graph neighborhood")
        # Slots interleave neighbors so a small degree cap still covers several directions.
        slots = [(dy,dx,k) for k in range(m) for dy,dx in offsets[:degree]][:self.cfg["graph_max_degree"]]
        dy,dx,rank = torch.tensor(slots,device=i.device).T
        def message(hidden, start_ids, layer):
            src,tgt = i[start_ids],j[start_ids]
            y,x = src[:,None]//w+dy,src[:,None]%w+dx
            inside = (y>=0)&(x>=0)&(y<h)&(x<w)
            neighbors = lookup[(y*w+x).clamp(0,n-1),rank]
            valid = inside&(neighbors>=0)
            ix = neighbors.clamp_min(0)
            # Relative target proximity weights local evidence; messages contain learned node features.
            dst = j[ix]
            dist = (dst%w-tgt[:,None]%w).square()+(dst//w-tgt[:,None]//w).square()
            weight = valid/(1+dist.float())
            pooled = (hidden[ix]*weight[:,:,None]).sum(1)/weight.sum(1,keepdim=True).clamp_min(1e-8)
            return hidden[start_ids]+layer(torch.cat((hidden[start_ids],pooled),1))
        nodes = torch.arange(len(ids),device=i.device)
        for layer in self.layers:
            hidden = torch.cat([checkpoint(message,hidden,chunk,layer,use_reentrant=False)
                                if self.training else message(hidden,chunk,layer)
                                for chunk in nodes.split(self.cfg["graph_chunk_size"])])
        residual = self.out(hidden).squeeze(1).float()
        return scores.index_add(0,ids,residual),raw,table,added
