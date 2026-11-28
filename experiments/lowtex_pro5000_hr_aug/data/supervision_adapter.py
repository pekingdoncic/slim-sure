"""Per-pair grid supervision. No coarse pair matrix is created."""
import torch
from src.utils.geometry import warp_kpts


def centers(hw, stride, device):
    ids = torch.arange(hw[0]*hw[1],device=device)
    return torch.stack((ids%hw[1],ids//hw[1]),-1).float()*stride + stride/2


def pixel_depth(depth, xy):
    h,w = depth.shape[-2:]
    ij = xy.floor().long()
    inside = xy.isfinite().all(-1) & (ij[...,0]>=0) & (ij[...,0]<w) & (ij[...,1]>=0) & (ij[...,1]<h)
    return depth[0,ij[...,1].clamp(0,h-1),ij[...,0].clamp(0,w-1)],inside


@torch.no_grad()
def project(d, xy, reverse=False, rel=.05, absolute=.01):
    a,b = ("1","0") if reverse else ("0","1")
    z,valid = pixel_depth(d["depth"+a],xy)
    valid &= z.isfinite() & (z>0)
    xyz = torch.linalg.solve(d["K"+a][0],torch.cat((xy,torch.ones_like(xy[:,:1])),1).T).T*z[:,None]
    t = d["T_"+a+"to"+b][0]
    xyz = xyz@t[:3,:3].T + t[:3,3]
    uvz = xyz@d["K"+b][0].T
    uv = uvz[:,:2]/uvz[:,2:3].clamp_min(1e-10)
    actual,inside = pixel_depth(d["depth"+b],uv)
    valid &= inside & (xyz[:,2]>0) & (actual>0) & actual.isfinite()
    valid &= (actual-xyz[:,2]).abs() <= absolute + actual*rel
    return uv,valid


@torch.no_grad()
def coarse_truth(d, cfg):
    hw = d["mask0"].shape[-2:]
    device = d["image0"].device
    xy = centers(hw,8,device)
    masks = [d[f"mask{i}"].flatten() for i in (0,1)]
    maps,visible,native = [],[],[]
    for i in (0,1):
        j=1-i
        uv,ok = project(d,xy,bool(i),cfg["depth_relative_tolerance"],cfg["depth_absolute_tolerance"])
        def cells(points):
            q = (points/8).floor().long()
            inside = (q[:,0]>=0)&(q[:,0]<hw[1])&(q[:,1]>=0)&(q[:,1]<hw[0])
            idx = (q[:,0]+q[:,1]*hw[1]).clamp(0,len(xy)-1)
            return idx,inside & masks[i] & masks[j][idx]
        idx,legal = cells(uv)
        maps.append(idx)
        visible.append(ok & legal)
        # Native train labels deliberately ignore depth consistency, following spvs_coarse.
        _,p = warp_kpts((xy*masks[i][:,None])[None],d[f"depth{i}"],d[f"depth{j}"],
                        d[f"T_{i}to{j}"],d[f"K{i}"],d[f"K{j}"])
        q = (p[0]/8-.5).round().long()
        inside = (q[:,0]>=0)&(q[:,0]<hw[1])&(q[:,1]>=0)&(q[:,1]<hw[0])
        native.append(torch.where(inside,q[:,0]+q[:,1]*hw[1],0))
    ids = torch.arange(len(xy),device=device)
    ni = ids[(native[1][native[0]]==ids) & (ids!=0)]
    i = ids[visible[0]]
    j = maps[0][i]
    return dict(i=i,j=j,reverse=visible[1],back=maps[1],native_i=ni,native_j=native[0][ni],
                target=maps[0],valid=visible[0])
