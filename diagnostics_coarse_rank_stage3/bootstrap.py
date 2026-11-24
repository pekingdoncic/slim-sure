"""Preserve approval and freeze derived actual file identities before official forward."""
from s3_common import *
from reliability_v1.environment_monitor import FAULT,NEW_WARNING,system_sample
from reliability_v1.evaluation import canonical_protocol
def main():
 RUN.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True);(RUN/'heartbeats').mkdir(exist_ok=True)
 if (RUN/'context.json').exists():print('Existing context; no replacement');return
 prior.gate()
 up={str(ROOT/'diagnostics_coarse_rank'/name):h for name,h in read(OLD/'experiment_code_identity.json')['files'].items()}
 for p,h in up.items():assert sha(p)==h
 inv=read(OLD/'inventory/route_frozen.json');chs={c['dataset']:c for c in inv['candidates']}
 for ds,h in APPROVED.items():
  c=chs[ds];assert sha(c['manifest'])==h and sha(c['frame_manifest'])==c['frame_sha256']
  up[c['manifest']]=h;up[c['frame_manifest']]=c['frame_sha256']
 for x in inv['inputs']:
  if x['dataset'] in APPROVED:assert sha(x['path'])==x['sha256'];up[x['path']]=x['sha256']
 # Same boot continuity, streamed since previous reviewed cursor. No arbitrary record truncation.
 old=read(OLD/'handoff.json')['final_environment'];boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();assert boot==old['boot_id'],'Boot changed: requires full new review'
 cmd=['journalctl','-k','-b','--no-pager','-o','json','--after-cursor='+old['journal_cursor']]
 p=subprocess.Popen(cmd,stdout=subprocess.PIPE,text=True);faults=[];warnings=[];cursor=old['journal_cursor'];count=0
 with (RUN/'kernel_since_previous_handoff.jsonl').open('w') as f:
  for line in p.stdout:
   x=json.loads(line);count+=1;assert x['_BOOT_ID']==boot.replace('-','');cursor=x['__CURSOR'];f.write(line)
   if FAULT.search(x['MESSAGE']):faults.append(x)
   if NEW_WARNING.search(x['MESSAGE']):warnings.append(x)
 assert p.wait()==0 and not faults and not warnings,(faults,warnings)
 env=dict(status='pass',time=now(),boot_id=boot,journal_cursor=cursor,records=count,resources=system_sample(),linux_free_bytes=shutil.disk_usage(RUN).free,scope='New kernel records since prior reviewed same-boot handoff; historical warnings retained in parent run')
 assert env['resources']['host_c_free_bytes']>14*2**30 and env['linux_free_bytes']>14*2**30
 save(RUN/'initial_environment.json',env)
 save(RUN/'approval.json',dict(time_recorded=now(),user_text='\u786e\u8ba4\u4e24\u4efd\u5b98\u65b9\u5b8c\u6574\u6e05\u5355\uff0c\u5f00\u59cb\u9636\u6bb53\u63cf\u8ff0\u6027\u590d\u9a8c',approved=APPROVED,not_approved='megadepth_other_holdout',interpretation='Both official complete unordered-deduplicated manifests, known prior exposure; descriptive separate domains'))
 record_hashes={};audit=[]
 for ds,h in APPROVED.items():
  frames={}
  for line in Path(chs[ds]['frame_manifest']).open():
   fr=json.loads(line)
   for kind in ['image','depth']:
    q=Path(fr[kind]);st=q.stat();assert [st.st_size,st.st_mtime_ns]==fr[kind+'_stat'],q
    fr[kind+'_sha256']=sha(q)
   if ds=='scannet1500':
    q=Path(fr['image']).parent.parent/'pose'/(Path(fr['image']).stem+'.txt');current=np.loadtxt(q)
    assert np.array_equal(current,np.asarray(fr['pose'])),q;fr['pose_path']=str(q);fr['pose_sha256']=sha(q)
   frames[fr['key']]=fr
  output=RUN/(ds+'_records.jsonl');num=0
  with output.open('w') as f:
   for line in Path(chs[ds]['manifest']).open():
    pair=json.loads(line);rec=dict(pair,dataset=ds,approved_manifest_sha=h,file_sha256={})
    for side in [0,1]:
     fr=frames[pair['frame'+str(side)]];K=np.asarray(fr['K'],float);pose=np.asarray(fr['pose'],float)
     if ds=='scannet1500':
      hh,ww=fr['image_hw'];K=np.diag([ww/640.,hh/480.,1.])@K;pose=np.linalg.inv(pose)
     rec.update({f'image{side}':fr['image'],f'depth{side}':fr['depth'],f'K{side}':K.tolist(),f'pose{side}':pose.tolist(),f'orig_hw{side}':fr['image_hw']})
     for k in ['image','depth']:rec['file_sha256'][k+str(side)]=fr[k+'_sha256']
    rec['coordinate_contract']='input pixels; original raw-image K; W2C relative pose1@inv(pose0); ScanNet depth_mm/1000'
    f.write(json.dumps(rec,sort_keys=True,separators=(',',':'))+'\n');num+=1
   f.flush();os.fsync(f.fileno())
  assert num==chs[ds]['deduplicated_pairs'];record_hashes[ds]=sha(output);audit.append(dict(dataset=ds,pairs=num,frames=len(frames),record_sha=record_hashes[ds]))
  print(json.dumps(audit[-1]),flush=True);del frames
 # Probe one actual image/depth was already hashed; independent derived write/fsync/readback.
 probe=RUN/'bounded_probe.tmp';blob=os.urandom(1024*1024)
 with probe.open('wb') as f:f.write(blob);f.flush();os.fsync(f.fileno())
 assert probe.read_bytes()==blob;probe.unlink()
 geom=canonical_protocol();geom['scope']='User-approved official benchmarks, separate descriptive replication with known prior exposure';geom['schema']='coarse_rank_stage3_official_geometry_v1'
 save(RUN/'geometry_protocol.json',geom)
 protocol=dict(approved=APPROVED,methods=['A','B0','B','C_if_gate'],source_protocol_sha=sha(OLD/'protocol.json'),resolution=832,AMP=True,seed=2026,warmup_per_pair=1,timed_repeats=3,primary_radius=4,optional_methods_excluded=True,training=False,geometry=geom,scene_bootstrap=2000,bootstrap_seed=2026,scene_min_for_independent_interpretation=10,prior_exposure=True,storage_forecast_gib=12,reserve_gib=2)
 save(RUN/'protocol.json',protocol)
 save(RUN/'context.json',dict(time=now(),run=str(RUN),results=str(OUT),upstream_hashes=up,approved=APPROVED,record_hashes=record_hashes,protocol_sha=sha(RUN/'protocol.json'),geometry_sha=sha(RUN/'geometry_protocol.json'),inventory_audit=audit,reference_HEAD=prior.gate()['HEAD']))
 save(RUN/'stage_status.json',dict(status='prepared',stage='data_adapter_validation',approved_pairs=2988,official_model_pair_units_completed=0))
 print(json.dumps(dict(status='prepared',run=str(RUN),resources=env)))
if __name__=='__main__':main()
