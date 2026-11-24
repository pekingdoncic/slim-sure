"""Opt-in boundary guard for native zero-size fine windows; frozen source unchanged."""
from common import *
import types
def install_empty_guard(model):
 fine=model._fine_correlation;refine=model._refinement
 def fine_guard(self,data):
  if data['b_idx_c'].numel()==0:
   width=int(data['coarse_scale']/data['fine_scale'])
   data['sim_matrix_f']=data['feat0_f'].new_empty((0,width*width,width*width))
   data['_empty_fine_guard_used']=True
   return
  return fine(data=data)
 def refine_guard(self,data,print_mem=False):
  if data['b_idx_it'].numel()==0:
   data.update(fine_coord_0=data['intermediate_coord_0'].clone(),fine_coord_1=data['intermediate_coord_1'].clone(),
    all_offset_1=data['intermediate_coord_1'].new_empty((self.refine_iters,0,2)),num_matches=0)
   data['_empty_refine_guard_used']=True
   return
  return refine(data=data,print_mem=print_mem)
 model._fine_correlation=types.MethodType(fine_guard,model);model._refinement=types.MethodType(refine_guard,model)
 return dict(guard_sha=sha(__file__),scope='Only zero-parent fine correlation and zero-child refinement; all nonempty calls delegate unchanged native methods')
