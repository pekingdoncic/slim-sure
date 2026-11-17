"""Final CPU acceptance after every owned worker and observer has exited."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path('/home/bupt/projects/SLiM-master');sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from evidential_fine.common import *
from evidential_fine.report_helpers import artifact_manifest,stage_inventory

def owned_processes(run,t_run):
    owned=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():continue
        try:
            cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            state=(p/'stat').read_text().split()[2]
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
        if state=='Z':continue
        if (str(run) in cmd or str(t_run) in cmd) and any(k in cmd for k in [
            'run_phase5.py','pose_units.py','continue_chain.py','resource_observer.py','diagnostics_threshold/run']):
            owned.append(dict(pid=int(p.name),state=state,command=cmd))
    return owned

def verify_manifest(path):
    obj=read(path);items=obj if isinstance(obj,list) else obj['artifacts']
    size=0
    for item in items:
        p=Path(item['path'])
        if not p.is_file() or p.stat().st_size!=item['size'] or sha(p)!=item['sha256']:
            raise IdentityFailure('Artifact size/SHA mismatch: '+str(p))
        size+=item['size']
    return dict(path=str(path),sha256=sha(path),artifacts=len(items),total_hashed_bytes=size)

def execute(run):
    run=Path(run);c=read(run/'context.json');out=Path(c['results']);tr=Path(c['T_run']);to=ROOT/'experiment_results'/tr.name
    start=read(run/'startup.json')
    if read(run/'stage_status.json').get('status')!='completed':raise RuntimeError('Report supervisor has not completed')
    if read(run/'chain/status.json').get('status')!='completed':raise RuntimeError('Serial coordinator has not completed')
    if not (run/'observation/exit.json').exists():raise RuntimeError('Resource observer has not exited')
    live=owned_processes(run,tr)
    if live:raise RuntimeError('Owned processes remain live: '+str(live))
    identity_gate()
    env=dict(os.environ,GIT_OPTIONAL_LOCKS='0')
    def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,env=env,text=True).rstrip('\n')
    head=git('rev-parse','HEAD');branch=git('branch','--show-current')
    if head!=start['HEAD'] or branch!=start['branch']:raise IdentityFailure('Branch/HEAD changed')
    status=git('status','--porcelain','--untracked-files=normal')
    original={line for line in start['worktree'].splitlines() if not line.startswith('?? ')}
    current={line for line in status.splitlines() if not line.startswith('?? ')}
    if current!=original:raise IdentityFailure('Tracked worktree status differs from preserved initial differences')
    allowed=['diagnostics_threshold/','evidential_fine/',str(to.relative_to(ROOT))+'/',str(out.relative_to(ROOT))+'/']
    unexpected=[line for line in status.splitlines() if line.startswith('?? ') and not any(line[3:].startswith(a) for a in allowed)]
    if unexpected:raise IdentityFailure('Unexpected new untracked path: '+str(unexpected))
    stages=stage_inventory(run)
    for row in stages:
        if row['status'].get('status')!='completed' or row['worker_status'].get('status')!='completed':
            raise RuntimeError('Stage not completed: '+row['stage'])
    training=read(out/'training.json')
    for route in ['A','B']:
        x=training[route]
        if x['status']!='completed' or x['committed_visits']!=27600 or len(x['epochs'])!=2:raise RuntimeError('Training incomplete')
        if any(e['validation_pairs']!=750 or e['visits']!=13800 for e in x['epochs']):raise RuntimeError('Epoch incomplete')
        if read(run/('final_'+route)/'summary.json')['pairs']!=100:raise RuntimeError('Final evaluation incomplete')
    if read(run/'efficiency/summary.json')['pairs']!=100:raise RuntimeError('Efficiency incomplete')
    pose=read(out/'pose.json')
    if pose['status']!='completed' or pose['new_units']!=2000 or pose['missing']!=0:raise RuntimeError('Pose coverage incomplete')
    t_verified=verify_manifest(to/'artifact_manifest.json')
    for folder in [to,out]:
        sizes=[p.stat().st_size for p in folder.rglob('*') if p.is_file()]
        if sum(sizes)>50*1024**2 or max(sizes,default=0)>10*1024**2:raise RuntimeError('Result footprint exceeds boundary')
    versions=run/'report_versions';versions.mkdir(exist_ok=True)
    for name in ['Phase5_Report.md','summary.json','stages.json','artifact_manifest.json']:
        p=out/name;old=versions/(name+'.before_final_acceptance')
        if p.exists() and not old.exists():shutil.copyfile(p,old)
    peaks=read(run/'observation/observed_peaks.json')
    checkpoint_size=max(x['size'] for x in read(out/'checkpoints.json'))
    final_time=now();wall=(datetime.datetime.fromisoformat(final_time)-datetime.datetime.fromisoformat(c['created'])).total_seconds()
    final=dict(status='completed',time_beijing=final_time,total_chain_wall_seconds=wall,identity=EXPECTED,HEAD=head,branch=branch,
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),processes_remaining=live,
        observer=read(run/'observation/exit.json'),coordinator=read(run/'chain/status.json'),stages=stages,
        training_visits={k:training[k]['committed_visits'] for k in training},training_updates={k:training[k]['committed_updates'] for k in training},
        final_pairs={k:100 for k in ['A','B']},efficiency_pairs=100,new_geometry_units=pose['new_units'],
        T_manifest_verification=t_verified,worktree_status=status,initial_tracked_status_preserved=True,
        frozen_identity_match=True,resources=peaks,largest_checkpoint_payload_bytes=checkpoint_size,
        disk_peak_scope='30s sampled apparent bytes, not a continuous exact peak; a simultaneous atomic checkpoint copy is at most one payload in addition.',
        decisions_T=read(to/'decisions.json'),decisions_Phase5=read(out/'decisions.json'),
        final_delivery_receipt=str(run/'final_delivery_receipt.json'))
    save(out/'final_acceptance.json',final);save(out/'stages.json',stages)
    d=read(out/'summary.json');d.update(final_verification_time=final_time,total_chain_wall_seconds=wall,all_owned_processes_exited=True,resources=peaks,final_acceptance='final_acceptance.json');save(out/'summary.json',d)
    report=out/'Phase5_Report.md';body=(versions/'Phase5_Report.md.before_final_acceptance').read_text()
    body+='\n\n## Final stopped-state acceptance\n\nVerified at '+final_time+'. Both training routes, final 100-pair evaluation per route, 100-pair efficiency measurement, 2,000 new geometry units, report, coordinator and resource observer have completed. No owned processes remain. Total wall time from both-task startup: '+f'{wall/3600:.3f}'+' hours. See [final acceptance](final_acceptance.json) for the complete state, resource observations, preserved Git differences and decisions. Immutable artifact checksums are in [the manifest](artifact_manifest.json); the manifest checksum itself is in the run-level final delivery receipt.\n'
    text(report,body)
    footprint=artifact_manifest(run,out);p_verified=verify_manifest(out/'artifact_manifest.json')
    result=dict(status='completed',time_beijing=now(),total_chain_wall_seconds=wall,T_report=str(to/'PhaseT_Report.md'),Phase5_report=str(report),
        T_manifest=t_verified,Phase5_manifest=p_verified,Phase5_report_sha256=sha(report),acceptance_sha256=sha(out/'final_acceptance.json'),
        footprint=footprint,all_owned_processes_exited=not owned_processes(run,tr),identity=EXPECTED)
    if not result['all_owned_processes_exited']:raise RuntimeError('Owned process appeared during finalization')
    save(run/'final_delivery_receipt.json',result)
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':execute(sys.argv[1])
