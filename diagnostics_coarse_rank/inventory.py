"""Read-only local inventory; new derived SQLite and manifests only."""
import os,sys,json,hashlib,time,sqlite3,gc
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from PIL import Image
import h5py
ROOT=Path('/home/bupt/projects/SLiM-master');sys.path.insert(0,str(ROOT))
from reliability_v1.runtime import sha,atomic_json,now
RUN=ROOT/'runs/coarse_rank_20261123T224428';OUT=ROOT/'experiment_results'/RUN.name
def save(p,x):atomic_json(p,x)
def norm(s):return str(s).replace('Undistorted_SfM/','phoenix/S6/zl548/MegaDepth_v1/').replace('/images/','/dense0/imgs/').replace('.JPG','.jpg').replace('.png','.jpg')
def imageid(path):
 s=str(path)
 if 'MegaDepth_v1/' in s:return 'md/'+s.split('MegaDepth_v1/')[-1]
 if '/scannet/test/' in s:return 'sn/'+s.split('/scannet/test/')[-1]
 if s.startswith('scene'):return 'sn/'+s
 return s
def sceneof(s):
 s=imageid(s)
 return s.split('/')[1].split('_')[0] if s.startswith('sn/') else s.split('/')[1]
def validmeta(K,T):
 return np.shape(K)==(3,3) and np.shape(T)==(4,4) and np.isfinite(K).all() and np.isfinite(T).all() and abs(np.linalg.det(np.asarray(K,float)))>1e-12 and abs(np.linalg.det(np.asarray(T,float)))>1e-12
def main():
 out=RUN/'inventory';out.mkdir(exist_ok=True)
 if (out/'route_frozen.json').exists():print('Already frozen');return
 db=sqlite3.connect(out/'inventory.sqlite')
 db.executescript('CREATE TABLE IF NOT EXISTS frames(dataset TEXT, key TEXT, scene TEXT, path TEXT, depth TEXT, record TEXT, readable INTEGER,reason TEXT, PRIMARY KEY(dataset,key));CREATE TABLE IF NOT EXISTS pairs(dataset TEXT,key TEXT,scene TEXT,a TEXT,b TEXT,record TEXT,readable INTEGER,PRIMARY KEY(dataset,key));')
 db.execute('pragma cache_size=-32768')
 source=sqlite3.connect((ROOT/'runs/phase1_20261115T004137/data_audit.sqlite').as_uri()+'?mode=ro&immutable=1',uri=True)
 phases={}
 for split in ['train','val']:
  scenes={x[0] for x in source.execute('select distinct scene from pairs where split=?',(split,))}
  images={imageid(x[0]) for x in source.execute('select image from frames') if sceneof(x[0]) in scenes}
  phases['phase1_2_'+split]={'scenes':scenes,'images':images,'pairs':set(),'pair_overlap_basis':'scene-disjoint implies no path-identity pair overlap'}
 for label,path in [('phaseT_and_phase3_4',ROOT/'runs/phase1_20261115T004137/fixed_final_epe.json'),('phase5_validation',ROOT/'runs/phase1_20261115T004137/fixed_validation.json')]:
  x=json.loads(path.read_text()); recs=x['records']; imgs=set(); ps=set();sc=set()
  for r in recs:
   a,b=map(imageid,[r['image0'],r['image1']]);imgs.update([a,b]);ps.add(tuple(sorted([a,b])));sc.add(r['scene'])
  phases[label]={'scenes':sc,'images':imgs,'pairs':ps,'evidence':str(path),'sha256':sha(path)}
 phases['phase5_training']=phases['phase1_2_train']
 prior_evidence=[]
 prior=Path('/home/bupt/projects/hybrid_match/outputs')
 used={'scenes':set(),'images':set(),'pairs':set()}
 # Stage0 complete benchmarks cover all later subsets of these official indices.
 for d in sorted((prior/'stage0').iterdir()):
  p=d/'pair_metadata.json'
  if not p.exists():continue
  x=json.loads(p.read_text())
  for r in x.values():
   a,b=map(imageid,r['image_ids']);used['images'].update([a,b]);used['pairs'].add(tuple(sorted([a,b])));used['scenes'].add(sceneof(a))
  prior_evidence.append(dict(path=str(p),sha256=sha(p),pairs=len(x)))
 for sub in ['m2b1_cpu_20261112T133600/samples','m2b2_pilot_diagnosis_20261112T213600/pair_lists']:
  for p in sorted((prior/sub).glob('*.json')):
   prior_evidence.append(dict(path=str(p),sha256=sha(p),scope='prior selection/pilot subsets of stage0 official benchmarks; schema retained for review'))
 phases['prior_hybrid_benchmark_selection']=used
 inputs=[];counts={};scene_rows=[];frame_memo={}
 def frame(dataset,sc,path,depth,K,T):
  key=imageid(path)
  old=db.execute('select readable from frames where dataset=? and key=?',(dataset,key)).fetchone()
  if old:return key,bool(old[0])
  reasons=[];rec={'key':key,'scene':sc,'image':str(path),'depth':str(depth),'K':np.asarray(K).tolist(),'pose':np.asarray(T).tolist()}
  if not validmeta(K,T):reasons.append('invalid_K_or_pose')
  try:
   with Image.open(path) as im:
    rec['image_hw']=[im.height,im.width];im.verify()
   st=path.stat();rec['image_stat']=[st.st_size,st.st_mtime_ns]
  except Exception as e:reasons.append('image:'+type(e).__name__)
  try:
   if depth.suffix=='.h5':
    with h5py.File(depth,'r') as f:
     a=f['depth'];rec['depth_shape']=list(a.shape);sample=np.asarray(a[0:1,0:1]);rec['depth_header_read']=True
   else:
    with Image.open(depth) as im:rec['depth_shape']=[im.height,im.width];im.verify()
   st=depth.stat();rec['depth_stat']=[st.st_size,st.st_mtime_ns]
  except Exception as e:reasons.append('depth:'+type(e).__name__)
  db.execute('insert into frames values(?,?,?,?,?,?,?,?)',(dataset,key,sc,str(path),str(depth),json.dumps(rec,separators=(',',':')),not reasons,';'.join(reasons)))
  return key,not reasons
 def pair(dataset,sc,a,b,record,ok):
  key=hashlib.sha256(('\0'.join(sorted([a,b]))).encode()).hexdigest()
  record.update(pair_id=key,scene=sc,frame0=a,frame1=b,direction='0to1')
  db.execute('insert or ignore into pairs values(?,?,?,?,?,?,?)',(dataset,key,sc,a,b,json.dumps(record,separators=(',',':')),int(ok)))
 for dataset,listing,directory,base in [
   ('megadepth1500',Path('/home/bupt/projects/SURE-main/assets/megadepth_test_1500_scene_info/megadepth_test_1500.txt'),Path('/home/bupt/projects/SURE-main/assets/megadepth_test_1500_scene_info'),ROOT/'data/megadepth/test'),
   ('megadepth_other_holdout',ROOT/'data/megadepth/index/trainvaltest_list/test_list.txt',ROOT/'data/megadepth/index/scene_info_0.1_0.7',ROOT/'data/megadepth/train')]:
  count=Counter();inputs.append(dict(dataset=dataset,path=str(listing),sha256=sha(listing)))
  for stem in listing.read_text().split():
   p=directory/(stem+'.npz');inputs.append(dict(dataset=dataset,path=str(p),sha256=sha(p),bytes=p.stat().st_size))
   # All these local serialized index files <=22MB. Only one index is alive.
   if p.stat().st_size>128*2**20:raise RuntimeError('Index exceeds preread memory envelope')
   z=np.load(p,allow_pickle=True); obj=z if isinstance(z,dict) else {k:z[k] for k in z.files}
   if not isinstance(z,dict):z.close()
   sc=stem.split('_')[0];local={}
   for ordinal,info in enumerate(obj['pair_infos']):
    ij,overlap,*rest=info; i,j=map(int,ij);count['raw']+=1
    if not np.isfinite(overlap) or overlap<=0:count['nonpositive_overlap']+=1;continue
    fs=[]
    for k in [i,j]:
     if k not in local:
      im=base/norm(obj['image_paths'][k]);dep=base/str(obj['depth_paths'][k])
      local[k]=frame(dataset,sc,im,dep,np.asarray(obj['intrinsics'][k]).reshape(3,3),obj['poses'][k])
     fs.append(local[k])
    pair(dataset,sc,fs[0][0],fs[1][0],{'index':str(p),'index_sha':inputs[-1]['sha256'],'ordinal':ordinal,'frame_indices':[i,j],'overlap':float(overlap)},fs[0][1] and fs[1][1])
    if ordinal%4096==0:db.commit()
   db.commit();del obj,z;gc.collect()
   print(json.dumps(dict(dataset=dataset,index=stem,raw=count['raw'],time=now())),flush=True)
  counts[dataset]=dict(count)
 dataset='scannet1500'; idir=Path('/home/bupt/projects/SURE-main/assets/scannet_test_1500');cnt=Counter()
 for name in ['test.npz','intrinsics.npz','scannet_test.txt']:inputs.append(dict(dataset=dataset,path=str(idir/name),sha256=sha(idir/name)))
 z=np.load(idir/'test.npz');names=z['name'];z.close()
 zz=np.load(idir/'intrinsics.npz');Ks={k:zz[k] for k in zz.files};zz.close()
 for ordinal,(s,scan,i,j) in enumerate(names):
  seq='scene%04d_%02d'%(s,scan);sc='scene%04d'%s;base=ROOT/'data/scannet/test'/seq;fs=[]
  for k in [i,j]:
   try:T=np.loadtxt(base/'pose'/f'{k}.txt')
   except Exception:T=np.full((4,4),np.nan)
   K=np.asarray(Ks.get(seq,np.full((3,3),np.nan))).reshape(3,3)
   fs.append(frame(dataset,sc,base/'color'/f'{k}.jpg',base/'depth'/f'{k}.png',K,T))
  pair(dataset,sc,fs[0][0],fs[1][0],{'index':str(idir/'test.npz'),'ordinal':ordinal,'sequence':seq,'pose_convention':'camera-to-world files; invert before relative 0to1','K_frame_hw':[480,640]},fs[0][1] and fs[1][1]);cnt['raw']+=1
 db.commit();counts[dataset]=dict(cnt)
 candidates=[]
 for dataset,count in counts.items():
  scenes=[x[0] for x in db.execute('select distinct scene from pairs where dataset=? order by scene',(dataset,))]
  imgs={x[0] for x in db.execute('select key from frames where dataset=?',(dataset,))}
  overlap={};not_new=set()
  for label,old in phases.items():
   common=sorted(set(scenes)&old['scenes']);not_new.update(common)
   # Candidate streamed, only old small pair sets are materialized.
   n=sum(tuple(sorted([a,b])) in old['pairs'] for a,b in db.execute('select a,b from pairs where dataset=?',(dataset,))) if old['pairs'] else (0 if not common else None)
   overlap[label]=dict(scenes=common,scene_count=len(common),image_count=len(imgs&old['images']),unordered_pair_count=n)
  pp=out/(dataset+'.jsonl'); fp=out/(dataset+'_frames.jsonl')
  for path,table in [(pp,'pairs'),(fp,'frames')]:
   with path.open('w') as f:
    for (record,) in db.execute(f'select record from {table} where dataset=? order by key',(dataset,)):f.write(record+'\n')
    f.flush();os.fsync(f.fileno())
  per=[]
  for sc in scenes:
   n,readable=db.execute('select count(*),sum(readable) from pairs where dataset=? and scene=?',(dataset,sc)).fetchone()
   per.append(dict(scene=sc,pairs=n,readable_pairs=readable,independent_of_known_used=sc not in not_new))
  n,ok=db.execute('select count(*),sum(readable) from pairs where dataset=?',(dataset,)).fetchone()
  reason=dict(db.execute('select reason,count(*) from frames where dataset=? and readable=0 group by reason',(dataset,)))
  candidate=dict(dataset=dataset,raw_pairs=count['raw'],deduplicated_pairs=n,readable_pairs=ok,unique_images=len(imgs),scenes=scenes,per_scene=per,manifest=str(pp),sha256=sha(pp),frame_manifest=str(fp),frame_sha256=sha(fp),overlap=overlap,metadata_missing=reason,known_independent_scenes=[r['scene'] for r in per if r['independent_of_known_used'] and r['readable_pairs']])
  if dataset=='scannet1500':
   seqs=sorted({r.split('/')[1] for r in imgs});candidate.update(sequence_ids=seqs,sequence_count=len(seqs),physical_scene_merge_basis='Index name first integer (sceneXXXX); second integer is scan suffix. This is benchmark scene-id grouping, no independent building/site metadata available.',domain='indoor cross-domain, never pooled with outdoors')
  candidates.append(candidate)
 res=dict(time=now(),status='frozen_before_first_model_output',inputs=inputs,candidates=candidates,prior_evidence=prior_evidence,unverified=['No external provenance ledger beyond local frozen records; unknown off-machine parameter selection cannot be excluded.','Cross-scene content duplicates not established by full image-content hash; image overlap uses normalized dataset/path identity.','Readability checks image decoder structure and depth dataset header/sample, not full GT pixel validity; eventual evaluable scene counts await approved inference.'],independent_model_runs=0,route='outdoor_descriptive_only: official 2 scenes have prior use; other preexisting holdout <=8 scenes; indoor separate descriptive cross-domain with prior selection exposure',source_resolution='SURE configured data root is absent; resolved actual existing SLiM data roots explicitly',scope='Index-positive-overlap rows preserved and reverse duplicates removed; no result-based selection')
 save(out/'route_frozen.json',res);save(OUT/'independent_inventory.json',res)
 db.close();print(json.dumps({k:res[k] for k in ['status','route','independent_model_runs']}),flush=True)
if __name__=='__main__':main()
