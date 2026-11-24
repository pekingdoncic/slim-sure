"""Preflight equivalence and domain-coordinate checks; no official scores used to tune."""
from s3_common import *
from s3_data import load_pair
import torch
def main():
 from reliability_data import load_pair as native_load,project_visible
 from s3_forward import reconstruct
 # Exact loader equivalence against frozen MegaDepth path, three original diagnostic pairs.
 rows=[]
 for r in prior.recs()[:3]:
  rr=dict(r,dataset='megadepth1500');a=native_load(r,device='cpu');b=load_pair(rr,device='cpu')
  for k in ['image0','image1','valid_mask0','valid_mask1','mask0','mask1','K0','K1','T_0to1']:assert torch.equal(a[k],b[k]),k
  for k in ['depth0','depth1','K0','K1','pose0','pose1','T_0to1']:assert np.array_equal(a['_gt'][k],b['_gt'][k]),k
  rows.append(dict(pair_id=r['pair_id'],loader_exact=True))
 # Indoor scaling is fixed by stored 640x480 K and original image dimensions.
 r=records('scannet1500')[0];d=load_pair(r,device='cpu');g=d['_gt']
 for side in [0,1]:
  assert g['depth'+str(side)].shape==(480,640);assert np.isfinite(g['K'+str(side)]).all()
  assert np.allclose(g['pose'+str(side)]@np.linalg.inv(g['pose'+str(side)]),np.eye(4),atol=1e-12)
 # Native-feature reconstruction must reproduce actual historical full P/logit hashes.
 rt=NativeRuntime(RUN/'validation/runtime');install_empty_guard(rt.model)
 for r in prior.recs()[:3]:
  d=rt.load(r);rt.forward(d);torch.cuda.synchronize();c,p=reconstruct(d,rt.model);ref=prior.get('A1',r,True)
  assert str(c['P_sha'])==str(ref['capture_P_sha']) and str(c['sim_sha'])==str(ref['capture_sim_sha'])
  assert np.array_equal(d['fine_coord_0'].cpu().numpy(),ref['source']) and np.array_equal(d['fine_coord_1'].cpu().numpy(),ref['target'])
  assert np.array_equal(p['B'],ref['plan_B']) and np.array_equal(p['C'],ref['plan_C']);del d,ref,c,p;gc.collect()
 rt.check_frozen()
 save(RUN/'validation/status.json',dict(status='pass',time=now(),diagnostic_pairs=3,loader_checks=rows,score_reconstruction_exact=True,selection_exact=True,scannet_depth='uint16 millimeters /1000',scannet_K='diag(raw_width/640,raw_height/480,1) @ native_K',scannet_pose='inverse(raw_C2W), then W2C1@inverse(W2C0)',official_forward_units=0))
 print('Validation pass')
if __name__=='__main__':main()
