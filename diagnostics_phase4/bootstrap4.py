from common4 import *
import subprocess,shutil
from reliability_v1.phase2_runtime import make_identity
from reliability_v1.environment_monitor import FAULT,NEW_WARNING,system_sample
from runner import guard_for
def main():
    ident=make_identity(read(CONFIG),SOURCE);actual=digest(ident)
    if actual!=EXPECTED:raise GateFailure('GLOBAL IDENTITY MISMATCH: expected '+EXPECTED+', actual '+actual)
    for p in sorted((ROOT/'runs').glob('phase4_diag_*')):
        if (p/'stage_status.json').exists() and read(p/'stage_status.json').get('status')!='completed':
            raise GateFailure('Existing Phase 4 progress; resume explicit run: '+str(p))
    t=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)));rid=t.strftime('%Y%m%dT%H%M%S')
    run=ROOT/'runs'/('phase4_diag_'+rid);res=ROOT/'experiment_results'/('phase4_'+rid)
    linux=__import__('shutil').disk_usage(ROOT).free;host=__import__('shutil').disk_usage('/mnt/c').free
    if min(linux,host)<2*2**30:raise GateFailure('Free capacity below 2 GiB: '+str((linux,host)))
    run.mkdir(exist_ok=False);res.mkdir(parents=True,exist_ok=False)
    context=dict(run_id=rid,run=str(run),results=str(res),phase2=str(SOURCE),phase3=str(P3),identity_sha=actual,time_beijing=now(),
        branch=subprocess.check_output(['git','-C',str(ROOT),'branch','--show-current'],text=True).strip(),
        HEAD=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        worktree=subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain=v1','--untracked-files=normal'],text=True).strip(),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),linux_free_bytes=linux,host_c_free_bytes=host,
        CPU_only=True,new_forward_calls=0,optimizer_updates=0,checkpoint_writes=0)
    save(run/'context.json',context);save(run/'frozen_identity.json',ident)
    save(run/'stage_status.json',dict(status='prepared',run_id=rid,stages={k:'pending' for k in ['P0','P1','P2','P2b','P3','P4','report']}))
    # Streaming current-boot review avoids truncating a long-lived WSL journal.
    p=subprocess.Popen(['journalctl','-k','-b','--no-pager','-o','json'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    faults=[];warnings=[];cursor=None;count=0
    with (run/'kernel_boot_review.jsonl').open('w') as f:
        for line in p.stdout:
            row=json.loads(line);count+=1
            if row.get('_BOOT_ID')!=context['boot_id'].replace('-',''):raise GateFailure('Kernel boot identity mismatch')
            cursor=row['__CURSOR'];msg=row.get('MESSAGE','');f.write(line)
            if FAULT.search(msg):faults.append(dict(cursor=cursor,message=msg))
            if NEW_WARNING.search(msg):warnings.append(dict(cursor=cursor,message=msg))
    p.wait()
    if p.returncode or not cursor:raise GateFailure('Cannot establish current-boot kernel continuity')
    save(run/'initial_environment.json',dict(status='failed' if faults else 'pass',boot_id=context['boot_id'],cursor=cursor,kernel_records=count,new_faults=faults,historical_boot_warnings=warnings,resources=system_sample(),time_beijing=now()))
    if faults:raise GateFailure('Current boot storage fault')
    for folder in [run,res]:
        path=folder/'write_probe.tmp';blob=os.urandom(1024*1024)
        with path.open('xb') as f:f.write(blob);f.flush();os.fsync(f.fileno())
        if path.read_bytes()!=blob:raise GateFailure('Write/readback mismatch')
        path.unlink();fd=os.open(folder,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
    guard=guard_for(run,'bootstrap');guard.check()
    # Phase 3 aggregate roots and code are verified before importing analysis adapters.
    integrity=source_json('final_integrity.json')
    for name,value in integrity['diagnostic_code_sha256'].items():
        if sha(ROOT/'diagnostics_phase3'/name)!=value:raise GateFailure('Phase 3 code changed: '+name)
    refs=source_json('input_refs.json')
    for name,value in refs['frozen_paths'].items():
        if sha(name)!=value:raise GateFailure('Frozen source differs: '+name)
    save(run/'source_roots.json',dict(phase3_hash_receipt=str(P3/'delivery_artifact_sha256.json'),phase3_hash_receipt_sha256=sha(P3/'delivery_artifact_sha256.json'),
        phase3_integrity=integrity,phase3_frozen_references=refs['frozen_paths']))
    artifact(run,'P0_bootstrap',context)
    print(json.dumps(context,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
