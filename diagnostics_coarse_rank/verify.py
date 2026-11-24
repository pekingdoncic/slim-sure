"""Independent edge-contract checks; CPU only except optional real empty forward."""
from common import *
from policies import select,allocate,xy,tests
from capture import ranks_at,boundary_counts
def main():
 import torch
 rng=np.random.RandomState(2026);checks=[]
 for n0,n1 in [(1,1),(3,7),(7,3),(0,0),(11,13)]:
  if not n0:continue
  mat=rng.randint(0,5,(n0,n1)).astype(np.float32);a=torch.tensor(mat);raw0=torch.arange(n0)*3+2;raw1=torch.arange(n1)*7+1
  rr,cc=np.indices(mat.shape);r=torch.tensor(rr.ravel());c=torch.tensor(cc.ravel())
  x,y=ranks_at(a,r,c,raw0,raw1,block=2)
  for k,(i,j) in enumerate(zip(r.numpy(),c.numpy())):
   row=np.lexsort((raw1.numpy(),-mat[i]));col=np.lexsort((raw0.numpy(),-mat[:,j]))
   assert x[k]==np.flatnonzero(row==j)[0]+1 and y[k]==np.flatnonzero(col==i)[0]+1
  checks.append('full_matrix_rank_ties_'+str((n0,n1)))
 for M in range(41):
  for it in range(20):
   ni=rng.randint(0,30,M);li=np.arange(30);p=rng.rand(30,5);p[rng.rand(30)<.2]=0
   q,r,rows=allocate(ni,li,np.tile(np.arange(5),(30,1)),p,.5)
   nonempty=(p>0).any(1)
   possible=[k for k in range(q+1) if (nonempty&~np.isin(li,ni[:M-k])).sum()>=k]
   assert r==max(possible) and len(rows)==r
 checks.append('allocator_exhaustive_820_random_cases')
 # Every real pair: q0 exact A, disable neighborhood exact B, no positive padding selected.
 for rec in recs():
  a=get('A1',rec,True);c={k[8:]:v for k,v in a.items() if k.startswith('capture_')}
  p=select(c,neighbor=False);z=select(c,fraction=0)
  native=np.stack([c['native_i'],c['native_j']],1)
  assert np.array_equal(z['B0'],native) and np.array_equal(z['B'],native) and np.array_equal(z['C'],native)
  assert np.array_equal(p['C'],p['B']) and np.array_equal(p['B'],a['plan_B'])
  assert np.all(p['targets'][np.arange(len(p['sources'])),p['bidx']]>=0)
 checks+=['100real_pairs_zero_quota_exact','100real_pairs_C_disabled_exact_B','100real_pairs_positive_finite_pool_only']
 x=dict(status='pass',time=now(),checks=checks,selector_tests=tests(),code=code_identity())
 save(OUT/'contract_checks.json',x);print(json.dumps(x))
if __name__=='__main__':main()
