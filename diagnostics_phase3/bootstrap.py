"""Read-only identity/environment bootstrap; all evidence goes to a fresh Phase 3 run."""
from pathlib import Path
import os,sys,json,datetime,subprocess,shutil,hashlib,time
ROOT=Path('/home/bupt/projects/SLiM-master');OLD=ROOT/'runs/phase1_20261115T004137'
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT))
from reliability_v1.phase2_runtime import make_identity
from reliability_v1.runtime import digest,sha,atomic_json
from reliability_v1.environment_monitor import FAULT,NEW_WARNING,system_sample
EXPECTED='1dc74dff202052885202e401ecf8fe44c40714b5877f0f547320aad86661808a'
cfg=json.loads((ROOT/'configs/reliability_phase2_20261115.json').read_text())
ident=make_identity(cfg,OLD);actual=digest(ident)
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
git=lambda *args:subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
proof=dict(time_beijing=now.isoformat(),branch=git('branch','--show-current'),HEAD=git('rev-parse','HEAD'),
    worktree=git('status','--porcelain=v1','--untracked-files=normal'),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
    identity_sha=actual,expected_identity_sha=EXPECTED,identity_status='pass' if actual==EXPECTED else 'failed',
    linux_filesystem_free_bytes=shutil.disk_usage(ROOT/'runs').free,host_c_free_bytes=shutil.disk_usage('/mnt/c').free,
    minimum_free_bytes=2*2**30)
print(json.dumps(proof,indent=2),flush=True)
assert actual==EXPECTED,'Frozen identity mismatch: globally stopped'
assert min(proof['linux_filesystem_free_bytes'],proof['host_c_free_bytes'])>=2*2**30,'Capacity below 2 GiB'
OUT=ROOT/'runs'/('phase3_diag_'+now.strftime('%Y%m%dT%H%M%S'));OUT.mkdir(exist_ok=False)
proof['output']=str(OUT);atomic_json(OUT/'bootstrap.json',proof)
atomic_json(OUT/'frozen_identity.json',ident)
frozen_files=[OLD/'formal_train/best.json',OLD/'final_epe/summary.json',OLD/'fixed_final_epe.json',OLD/'fixed_validation.json',ROOT/'configs/reliability_phase2_20261115.json']
best=json.loads((OLD/'formal_train/best.json').read_text());assert sha(OLD/'formal_train'/best['payload'])==best['sha256']
records=json.loads((OLD/'fixed_validation.json').read_text())['records']
epes=json.loads((OLD/'fixed_final_epe.json').read_text())['records']
assert len(records)==750 and len(epes)==100 and {r['pair_id'] for r in epes}<={r['pair_id'] for r in records}
assert all(str(r['scene']) not in ('0015','0022') for r in records+epes)
assert len({r['scene'] for r in records})==15
p=subprocess.Popen(['journalctl','-k','-b','--no-pager','-o','json'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
faults=[];warnings=[];count=0;cursor=None;boot=proof['boot_id'].replace('-','')
with (OUT/'kernel_boot_review.jsonl').open('w') as f:
    for line in p.stdout:
        row=json.loads(line);count+=1
        if row.get('_BOOT_ID')!=boot:raise RuntimeError('Kernel boot mismatch')
        cursor=row.get('__CURSOR',cursor);message=row.get('MESSAGE','')
        f.write(line)
        if FAULT.search(message):faults.append(dict(cursor=cursor,message=message))
        if NEW_WARNING.search(message):warnings.append(dict(cursor=cursor,message=message))
p.wait()
assert p.returncode==0,p.stderr.read()
atomic_json(OUT/'initial_environment.json',dict(status='failed' if faults else 'pass',kernel_records=count,cursor=cursor,boot_id=proof['boot_id'],
    current_boot_faults=faults,current_boot_warning_records=warnings,resources=system_sample(),time_beijing=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    kernel_review_scope='Streamed current boot only; bounded per-record memory. Current-boot warnings separately listed; historical failure files remain read-only.'))
assert not faults,'Current-boot storage fault; globally stopped'
# Bounded read and new-directory fsync/readback test.
sample=records[0];file=Path(sample['image0']);h=hashlib.sha256()
with file.open('rb') as f:h.update(f.read(256*1024))
temp=OUT/'bounded_write_probe.tmp';data=os.urandom(1024*1024)
with temp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
assert temp.read_bytes()==data;temp.unlink()
directory=os.open(OUT,os.O_RDONLY|os.O_DIRECTORY);os.fsync(directory);os.close(directory)
atomic_json(OUT/'input_refs.json',dict(source_run=str(OLD),frozen_paths={str(p):sha(p) for p in frozen_files},
    best_pointer=best,fixed_validation_pairs=750,fixed_final_epe_pairs=100,excluded_scenes=['0015','0022'],
    bounded_data_probe=dict(path=str(file),read_limit_bytes=256*1024,sha256=h.hexdigest()),write_probe_bytes=len(data)))
atomic_json(OUT/'stage_status.json',dict(status='prepared',A='pending',B='pending',C='pending',A4='pending',identity_sha=actual,output=str(OUT)))
print('PHASE3_OUTPUT='+str(OUT),flush=True)
