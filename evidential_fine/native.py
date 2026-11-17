"""Allowed process-local wrappers; frozen SLiM source is never edited."""
from .common import *
import torch
from torch.nn import functional as F
from reliability_data import project_visible
class EndNativePrefix(Exception):pass
def prefix(rt,d,route):
    method='_fine_correlation' if route=='A' else '_refinement';old=getattr(rt.model,method);counts={'stop_calls':0}
    def stop(data):
        counts['stop_calls']+=1
        raise EndNativePrefix()
    setattr(rt.model,method,stop)
    try:
        with torch.no_grad(),torch.cuda.amp.autocast(enabled=rt.config['amp']):
            try:rt.model(d)
            except EndNativePrefix:pass
        if counts['stop_calls']!=1:raise RuntimeError('Native prefix stop not reached once')
        if route=='B':d['fine_parent_id']=d['coarse_match_id'][d['m_idx_it']]
        if 'fine_coord_1' in d:raise RuntimeError('Prefix unexpectedly executed refinement')
    finally:setattr(rt.model,method,old)
    return d
def inputs(rt,d,route,ix=None):
    """The same extraction function is used after full or truncated forwards."""
    if route=='A':
        n=len(d['coarse_match_id']);idx=torch.arange(n,device='cuda') if ix is None else torch.as_tensor(ix,device='cuda')
        b=d['b_idx_c'][idx];patches=[]
        if float(d['coarse_scale'])!=8 or float(d['fine_scale'])!=1:raise RuntimeError('A actual native stride/window differs')
        for side in [0,1]:
            scale=d['fine_scale']*d['scale'+str(side)][b] if 'scale'+str(side) in d else d['fine_scale']
            w=rt.model._extract_feat_window(d['feat'+str(side)+'_f'],b,d['coarse_coord_'+str(side)][idx]/scale,8).float()
            p=F.avg_pool2d(w,4).permute(0,2,3,1).reshape(len(idx),-1);patches.append(p)
        return (d['coarse_descriptor0'][idx].float(),d['coarse_descriptor1'][idx].float(),*patches)
    n=len(d['fine_parent_id']);idx=torch.arange(n,device='cuda') if ix is None else torch.as_tensor(ix,device='cuda')
    b=d['b_idx_it'][idx];windows=[]
    for side in [0,1]:
        scale=d['fine_scale']*d['scale'+str(side)][b] if 'scale'+str(side) in d else d['fine_scale']
        with torch.no_grad(),torch.cuda.amp.autocast(enabled=rt.config['amp']):
            w=rt.model._extract_feat_window_bilinear(d['feat'+str(side)+'_f'],b,d['intermediate_coord_'+str(side)][idx]/scale,6)
        windows.append(w.float())
    return (torch.cat(windows,dim=1),)
def supervision(d,route):
    if route=='A':
        source=d['coarse_coord_0'].float().cpu().numpy()+4
        target=d['coarse_coord_1'].float().cpu().numpy()+4
        ids=d['coarse_match_id'].cpu().numpy();parent=ids.copy()
    else:
        source=d['intermediate_coord_0'].float().cpu().numpy()
        target=d['intermediate_coord_1'].float().cpu().numpy()
        parent=d['fine_parent_id'].cpu().numpy();ids=np.arange(len(parent),dtype=np.int64)
    gt=project_visible(source,d);truth=gt['points1_gt_input'];offset=truth-target;valid=gt['valid_gt'].astype(bool)
    R=8 if route=='A' else 3;inside=valid&(np.max(np.abs(offset),axis=1)<R)
    finite(offset[valid],'supervision offset')
    return dict(source=source,initial=target,gt=truth,offset=offset,valid=valid,inside=inside,reasons=gt['invalid_reason'],ids=ids,parent=parent,R=R)
def selected_points(s,route,epoch,pair_id,seed=2026):
    from reliability_v1.scene_protocol import parent_indices,derived_seed,uniform_indices
    if route=='A':return parent_indices(s['inside'],s['ids'],seed,epoch,pair_id,cap=1024)
    ix=np.flatnonzero(s['inside']);ix=ix[np.lexsort((s['ids'][ix],s['parent'][ix]))]
    if len(ix)>2048:ix=ix[uniform_indices(len(ix),2048,derived_seed(seed,'fine_children',epoch,pair_id))]
    return ix
def predict(head,features,chunk=4096):
    # Chunking changes only head batch size, never selects or drops outputs.
    n=len(features[0]);results=[]
    with torch.no_grad():
        for start in range(0,n,chunk):
            with torch.cuda.amp.autocast(enabled=False):out=head(tuple(x[start:start+chunk] for x in features))
            results.append({k:v.detach().cpu().numpy() for k,v in out.items() if k!='floor_rates'})
    if not results:return dict(mu=np.empty((0,2)),nu=np.empty((0,2)),alpha=np.empty((0,2)),beta=np.empty((0,2)),aleatoric=np.empty(0),epistemic=np.empty(0))
    value={k:np.concatenate([r[k] for r in results]) for k in results[0]}
    for k,v in value.items():finite(v,k)
    return value
def trajectory(d):
    radius=3;scale=d['fine_scale']
    if 'scale1' in d:scale=scale*d['scale1'][d['b_idx_it']].float()
    px=(d['all_offset_1'].float()*radius*scale).cpu().numpy()
    initial=d['intermediate_coord_1'].float().cpu().numpy()
    return np.concatenate([initial[None],initial[None]+np.cumsum(px,axis=0,dtype=np.float32)],axis=0)
