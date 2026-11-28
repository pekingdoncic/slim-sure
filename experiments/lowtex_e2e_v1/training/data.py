from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from src.datasets.utils.dataset import read_megadepth_gray,read_megadepth_depth


@lru_cache(maxsize=64)
def cameras(index, name):
    with np.load(index/(name+".npz"),allow_pickle=True) as a:
        return {k:a[k] for k in ("image_paths","depth_paths","intrinsics","poses")}


def load_pair(base, name, frames, side, device, key):
    meta=cameras(base/"index/scene_info_0.1_0.7_no_sfm",name)
    d={"dataset_name":["MegaDepth"],"pair_names":[key]}
    gt={}
    wh=[]
    for number,i in enumerate(frames):
        s=str(number)
        rgb=base/"train"/meta["image_paths"][i]
        depth=read_megadepth_depth(base/"train"/meta["depth_paths"][i])
        image,mask,scale=read_megadepth_gray(str(rgb),side,1,True,None)
        rh,rw=int(mask[:,0].sum()),int(mask[0,:].sum())
        height,width=round(rh*scale[1].item()), round(rw*scale[0].item())
        wh.append([width,height])
        d["image"+s]=image.unsqueeze(0).to(device)
        d["scale"+s]=scale.unsqueeze(0).to(device)
        d["valid_mask"+s]=mask[None].to(device)
        d["mask"+s]=d["valid_mask"+s][:,::8,::8]
        d["depth"+s]=depth[None].to(device)
        k=np.array(meta["intrinsics"][i],dtype=np.float64)
        d["K"+s]=torch.from_numpy(k).float()[None].to(device)
        gt["depth"+s]=depth.numpy()
        gt["K"+s]=k
        gt["plan"+s]=dict(orig_hw=[height,width],resized_hw=[rh,rw],scale_input_to_orig_xy=scale.tolist())
    t=meta["poses"][frames[1]]@np.linalg.inv(meta["poses"][frames[0]])
    gt["T_0to1"]=np.asarray(t,dtype=np.float64)
    d["T_0to1"]=torch.as_tensor(t,dtype=torch.float32,device=device)[None]
    d["T_1to0"]=d["T_0to1"].inverse()
    d["_gt"]=gt
    return d,wh
