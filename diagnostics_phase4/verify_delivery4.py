"""Final read-only receipt verification and Phase 4 delivery seal."""
from common4 import *
from runner import guard_for
from pose_statistics import verified_unit
from reliability_v1.phase2_runtime import make_identity
from report_phase3 import disk
import argparse,subprocess
def main(run):
    run=Path(run);out=results(run);guard=guard_for(run,'delivery')
    if digest(make_identity(read(CONFIG),SOURCE))!=EXPECTED:raise GateFailure('GLOBAL IDENTITY MISMATCH')
    stage_names=['P0','P1','P2','P2b','P3','P4','report']
    states={s:read(run/s/'status.json') for s in stage_names}
    if any(x['status']!='completed' or x['returncode']!=0 for x in states.values()):raise GateFailure('Stage not actually complete')
    active=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():continue
        try:cmd=(p/'cmdline').read_bytes()
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if (read(run/'context.json')['run_id'].encode() in cmd) and (b'run_phase4.py' in cmd or b'pose_units.py' in cmd):
            active.append(dict(pid=int(p.name),command=cmd.replace(b'\0',b' ').decode(errors='replace')))
    if active:raise GateFailure('A task is still active: '+str(active))
    roots=read(run/'source_roots.json')
    if sha(roots['phase3_hash_receipt'])!=roots['phase3_hash_receipt_sha256']:raise GateFailure('Phase 3 root receipt changed')
    for path,h in roots['phase3_frozen_references'].items():
        if sha(path)!=h:raise GateFailure('Read-only input changed')
    integ=read(out/'final_integrity.json')
    actual={p.name:sha(p) for p in CODE.glob('*.py')}
    if actual!=integ['diagnostic_code_sha256']:raise GateFailure('Phase 4 code changed after report')
    counts={}
    for stage in ['P0','P1','P2','P2b']:
        counts[stage]=0
        for r in records('epe'):
            x=own_receipt(run,stage,r)
            if not x or x['status']!='committed':raise GateFailure('Missing pair commit')
            counts[stage]+=1
        guard.check()
    geometry={}
    for stage,expected in [('P3',5000),('P4',2000)]:
        plan=read(run/stage/'geometry_plan.json');predchecked={};n=fail=0
        for i,key in enumerate(plan['formal_unit_order']):
            row=verified_unit(plan,key)
            if row is None or row['unit']!=key:raise GateFailure('Missing/misaligned geometry: '+key)
            u=plan['units'][key]
            if u['prediction'] not in predchecked:predchecked[u['prediction']]=sha(u['prediction'])
            if predchecked[u['prediction']]!=u['prediction_sha256']:raise GateFailure('Geometry source changed')
            n+=1;fail+=row['solver_status']=='geometry_failure'
            if i%500==0:guard.check()
        started=collections.Counter();commits=collections.Counter()
        for file in (run/stage/'geometry').glob('*_attempts.jsonl'):
            with file.open() as f:
                for line in f:
                    v=json.loads(line)
                    if v['status']=='started':started[v['unit']]+=1
                    elif v['status']=='committed':commits[v['unit']]+=1
        if n!=expected or len(started)!=expected or any(v!=1 for v in started.values()) or any(v!=1 for v in commits.values()):
            raise GateFailure('Unexpected formal call count/repeated unit')
        geometry[stage]=dict(planned=expected,verified_committed=n,actual_solver_starts=sum(started.values()),repeated_units=0,solver_failures=int(fail),missing=0,prediction_files=len(predchecked))
    for stage in stage_names:
        env=read(run/'environment_probe'/stage/'status.json')
        if env['status']!='pass' or env['new_faults']:raise GateFailure('Environment guard failure')
    if not checkpoint_free(run) or any(out.rglob('*.pt')) or any(out.rglob('*.ckpt')) or any(out.rglob('*.log')):raise GateFailure('Unexpected checkpoint/log in result folder')
    for item in read(out/'artifact_manifest.json'):
        for ref in item['files']:
            p=Path(ref['path']) if ref['location']=='runs' else out/ref['path']
            if not p.is_file() or sha(p)!=ref['sha256']:raise GateFailure('Report artifact changed: '+str(p))
    guard.check()
    status=read(run/'stage_status.json');status.update(status='completed',last_verified_beijing=now(),active_diagnostic_writers=[],active_monitoring=False,
        monitoring_end_reason='All authorized phases, report, immutable-source and unit receipts verified; all writers exited',resume_needed=False,
        result_folder=str(out),new_forwards=0,optimizer_updates=0,checkpoint_writes=0)
    save(run/'stage_status.json',status)
    inventory=run/'run_artifact_inventory.jsonl'
    with inventory.open('w') as f:
        for p in sorted(run.rglob('*')):
            if not p.is_file() or p==inventory or 'environment_probe/delivery/' in str(p.relative_to(run)) or p.name.startswith('delivery_'):continue
            f.write(json.dumps(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)),sort_keys=True)+'\n')
        f.flush();os.fsync(f.fileno())
    result=dict(status='completed',time_beijing=now(),run_id=read(run/'context.json')['run_id'],identity_sha=EXPECTED,HEAD=integ['HEAD'],
        pair_receipts=counts,geometry=geometry,baseline_reused_units=500,baseline_new_solves=0,environment=guard.status,
        run_inventory=dict(path=str(inventory),bytes=inventory.stat().st_size,sha256=sha(inventory)),
        run_disk=disk(run),active_writers=[],new_forwards=0,optimizer_updates=0,checkpoint_writes=0)
    artifact(run,'delivery_verification',result)
    files=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]
    text(out/'RESULT_FILES.txt','File inventory (this index is excluded from its own entries)\n\n'+'\n'.join(x['path']+'\t'+str(x['bytes'])+'\t'+x['sha256'] for x in files)+'\n')
    allfiles=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size) for p in sorted(out.rglob('*')) if p.is_file()]
    total=sum(x['bytes'] for x in allfiles)
    if total>50*2**20 or any(x['bytes']>10*2**20 for x in allfiles):raise GateFailure('Result size contract exceeded')
    save(run/'delivery_receipt.json',dict(**result,results=str(out),result_files=allfiles,result_total_bytes=total,result_total_MiB=total/2**20,
        result_hash_inventory=dict(path=str(out/'RESULT_FILES.txt'),sha256=sha(out/'RESULT_FILES.txt'))))
    print(json.dumps(dict(status='completed',results=str(out),files=allfiles,total_bytes=total,total_MiB=total/2**20,verified_geometry=geometry,last_verified_beijing=now()),ensure_ascii=False,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args();main(ROOT/'runs'/('phase4_diag_'+a.run_id))
