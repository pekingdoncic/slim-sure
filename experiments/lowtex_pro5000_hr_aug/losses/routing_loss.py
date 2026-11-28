def routing_loss(logits, valid0, valid1, blocks):
    """Symmetric conditional cross-entropy, uniform across visible positive blocks per query."""
    import torch
    if not len(blocks):
        return logits.sum()*0
    n = logits.shape[1]
    i,j = blocks//n,blocks%n
    def one(x,r,c,mask):
        unique,idx,counts = r.unique(return_inverse=True,return_counts=True)
        rows = x[unique].float().masked_fill(~mask[None],-torch.inf)
        per_edge = rows.logsumexp(1)[idx]-rows[idx,c]
        return (per_edge/counts[idx]).sum()/len(unique)
    return .5*(one(logits,i,j,valid1)+one(logits.T,j,i,valid0))
