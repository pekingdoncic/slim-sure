"""Official descriptive replication; authorized fixed manifests only."""
import os,sys,json,time,hashlib,math,gc,subprocess,traceback,datetime,shutil
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1');sys.dont_write_bytecode=True
ROOT=Path('/home/bupt/projects/SLiM-master');OLD=ROOT/'runs/coarse_rank_20261123T224428';CODE=ROOT/'diagnostics_coarse_rank_stage3'
sys.path.insert(0,str(ROOT/'diagnostics_coarse_rank'));sys.path.insert(0,str(ROOT))
import common as prior
from common import NativeRuntime,low_grids,sha,save,read,now,npzout,npzload,array_sha,ratio,resource_sample,state_hash
from policies import select
from capture import ranks_at,matrix_hash,install
from empty_guard import install_empty_guard
import numpy as np
RUN=OLD/'stage3_official_20261124';OUT=ROOT/'experiment_results/coarse_rank_20261123T224428/stage3_official_20261124'
APPROVED={'megadepth1500':'21f702c9f6dcdf97c76322ba72f60a5deee090f3d8bbc5bef10fc6a71e693a21','scannet1500':'248c9a744b3efba107ff85fbb6063e50f6ab9f20d82b37592a9a6a5959c6f8c4'}
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def records(ds):
 assert ds in APPROVED
 return [json.loads(x) for x in (RUN/(ds+'_records.jsonl')).open()]
def gate(check_new=True):
 prior.gate();ctx=read(RUN/'context.json')
 assert sha(RUN/'protocol.json')==ctx['protocol_sha'] and sha(RUN/'geometry_protocol.json')==ctx['geometry_sha']
 for p,h in ctx['upstream_hashes'].items():assert sha(p)==h,p
 for ds,h in APPROVED.items():assert sha(OLD/'inventory'/(ds+'.jsonl'))==h
 for ds in APPROVED:assert sha(RUN/(ds+'_records.jsonl'))==ctx['record_hashes'][ds]
 if check_new:
  code=read(RUN/'code_identity.json')
  for name,h in code['execution_files'].items():assert sha(CODE/name)==h,('execution code changed',name)
 return ctx
def identity(ds):return digest(dict(context=sha(RUN/'context.json'),code=read(RUN/'code_identity.json')['execution_files'],dataset=ds))
def paths(ds,m,r):
 p=RUN/ds/m/'pairs'/r['pair_id']
 return p.with_suffix('.npz'),p.with_suffix('.json'),p.with_suffix('.receipt.json')
def valid(ds,m,r,verify=True):
 a,j,p=paths(ds,m,r)
 if not p.exists():return False
 x=read(p);assert x['identity']==identity(ds) and x['pair_id']==r['pair_id']
 if verify:assert sha(a)==x['arrays_sha'] and sha(j)==x['result_sha']
 return True
def get(ds,m,r,arrays=False):
 assert valid(ds,m,r)
 return npzload(paths(ds,m,r)[0]) if arrays else read(paths(ds,m,r)[1])
def commit(ds,m,r,a,result):
 assert not valid(ds,m,r)
 ap,jp,rp=paths(ds,m,r);ap.parent.mkdir(parents=True,exist_ok=True);npzout(ap,a);save(jp,result)
 save(rp,dict(status='committed',identity=identity(ds),pair_id=r['pair_id'],scene=r['scene'],time=now(),arrays_sha=sha(ap),result_sha=sha(jp)))
def beat(stage,**kw):save(RUN/'heartbeats'/(stage+'.json'),dict(time=now(),unix=time.time(),pid=os.getpid(),stage=stage,**kw))
def monitor(phase):
 from reliability_v1.environment_monitor import EnvironmentMonitor
 candidates=[RUN/'initial_environment.json',*list((RUN/'environment_probe').glob('*/status.json'))]
 boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();items=[(p,read(p)) for p in candidates if p.exists()]
 items=[(p,x) for p,x in items if x.get('status')=='pass' and x.get('boot_id')==boot];p,x=max(items,key=lambda t:t[0].stat().st_mtime_ns)
 g=EnvironmentMonitor.__new__(EnvironmentMonitor);g.run=RUN;g.out=RUN/'environment_probe'/phase;g.out.mkdir(parents=True,exist_ok=True);g.minimum_host_bytes=2*2**30
 g.boot=boot;g.cursor=x['journal_cursor'];g.started=time.monotonic();g.checks=0;g.last_sample=None;g.blocked_reason=None;g.status=None
 save(g.out/'seed.json',dict(source=str(p),source_sha=sha(p),cursor=g.cursor));g.check(False);return g
