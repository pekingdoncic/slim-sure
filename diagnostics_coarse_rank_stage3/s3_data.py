"""Exact R832 preprocessing, original-pixel K, W2C poses, depth in meters."""
from s3_common import *
import cv2,h5py
from reliability_data import plan_geometry,verify_frozen_file
def load_pair(r,device='cuda',include_gt=True):
 import torch
 d=dict(pair_id=r['pair_id'],scene_id=r['scene'],direction='0to1',model_coordinate_space='input');g={}
 for side in [0,1]:
  ik='image'+str(side);dk='depth'+str(side);verify_frozen_file(r[ik],r['file_sha256'][ik])
  raw=cv2.imread(r[ik],cv2.IMREAD_GRAYSCALE)
  if raw is None:raise RuntimeError('Approved image unreadable '+r[ik])
  plan=plan_geometry(raw.shape,832);rh,rw=plan['resized_hw'];im=cv2.resize(raw,(rw,rh),interpolation=cv2.INTER_LINEAR)
  canvas=np.zeros((832,832),np.float32);canvas[:rh,:rw]=im.astype(np.float32)/255.;mask=np.zeros((832,832),bool);mask[:rh,:rw]=True
  d[ik]=torch.from_numpy(canvas)[None,None].to(device);d['valid_mask'+str(side)]=torch.from_numpy(mask)[None].to(device);d['mask'+str(side)]=torch.from_numpy(mask[::8,::8].copy())[None].to(device)
  d['input_to_orig_scale'+str(side)]=torch.tensor(plan['scale_input_to_orig_xy'],dtype=torch.float32,device=device)[None];d['preprocessing_plan'+str(side)]=plan
  if include_gt:
   verify_frozen_file(r[dk],r['file_sha256'][dk])
   if r['dataset']=='scannet1500':
    dep=cv2.imread(r[dk],cv2.IMREAD_UNCHANGED)
    if dep is None:raise RuntimeError('Approved depth unreadable '+r[dk])
    dep=dep.astype(np.float64)/1000.
   else:
    with h5py.File(r[dk],'r') as f:dep=f['depth'][()]
   g[dk]=dep;g['plan'+str(side)]=plan;g['K'+str(side)]=np.asarray(r['K'+str(side)],np.float64);g['pose'+str(side)]=np.asarray(r['pose'+str(side)],np.float64)
 if include_gt:
  g.update(T_0to1=g['pose1']@np.linalg.inv(g['pose0']),depth_absolute_tolerance=.01,depth_relative_tolerance=.05);d['_gt']=g
  for k in ['K0','K1','T_0to1']:d[k]=torch.as_tensor(g[k],dtype=torch.float32,device=device)[None]
 return d
