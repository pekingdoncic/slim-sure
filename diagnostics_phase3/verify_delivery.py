"""Final delivery verification. Reads evidence and writes only Phase 3 delivery records."""
from common import *
from runner import guard_for
from report_phase3 import disk
import argparse, subprocess

def main(run):
    run=Path(run)
    guard=guard_for(run,'delivery')
    stage_states={s:read(run/s/'status.json') for s in ['A','B','C','A4','report']}
    if any(s['status']!='completed' or s.get('returncode')!=0 for s in stage_states.values()):
        raise GateFailure('A phase or report has not completed')
    active=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():continue
        try:cmd=(p/'cmdline').read_bytes()
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if str(run).encode() in cmd and (b'runner.py' in cmd or b'pose_units.py' in cmd):
            active.append(dict(pid=int(p.name),command=cmd.replace(b'\0',b' ').decode(errors='replace')))
    if active:raise GateFailure('Diagnostic writer still active: '+str(active))
    verified_pairs={}
    for stage,kind in [('A','epe'),('B','validation'),('C','epe')]:
        counts=0
        for record in records(kind):
            receipt=committed(run,stage,record)
            if not receipt or receipt.get('status')!='committed':raise GateFailure('Missing committed '+stage+' pair')
            counts+=1
        verified_pairs[stage]=counts
        guard.check()
    plan=read(run/'A4/geometry_plan.json')
    counts=dict(formal=0,independent_repeat=0,solver_failures=0)
    checked_inputs={}
    for n,(key,u) in enumerate(plan['units'].items()):
        folder=Path(plan['output'])/u['directory']
        p=folder/('seed_%d.json'%u['seed'])
        q=folder/('seed_%d_commit.json'%u['seed'])
        row=read(p);receipt=read(q)
        if row['status']!='committed' or receipt['status']!='committed' or sha(p)!=receipt['sha256']:
            raise GateFailure('Invalid committed geometry: '+key)
        if row['unit']!=key or row['identity']!=receipt['identity'] or row['identity']['pair_id']!=u['pair_id'] or row['identity']['seed']!=u['seed']:
            raise GateFailure('Geometry identity mismatch: '+key)
        if row['prediction_sha256']!=u['prediction_sha256']:raise GateFailure('Geometry prediction mismatch')
        pred=Path(u['prediction'])
        if str(pred) not in checked_inputs:checked_inputs[str(pred)]=sha(pred)
        if checked_inputs[str(pred)]!=u['prediction_sha256']:raise GateFailure('Geometry input changed')
        counts['independent_repeat' if u.get('independent_repeat') else 'formal']+=1
        if row['solver_status']=='geometry_failure':
            counts['solver_failures']+=1
            if row['pose_error_deg'] is not None:raise GateFailure('Failure error encoding')
        elif row['solver_status']=='success':
            finite(np.asarray([row['pose_error_deg'],row['R_error_deg'],row['t_error_deg']]),'geometry result')
        else:raise GateFailure('Unknown geometry result')
        if n%500==0:guard.check()
    if counts['formal']!=8500 or counts['independent_repeat']!=25:raise GateFailure('Incomplete geometry count')
    integrity=read(run/'final_integrity.json')
    for name,value in integrity['diagnostic_code_sha256'].items():
        if sha(ROOT/'diagnostics_phase3'/name)!=value:raise GateFailure('Code changed since report: '+name)
    if set(integrity['diagnostic_code_sha256'])!={p.name for p in (ROOT/'diagnostics_phase3').iterdir() if p.suffix in ['.py','.json']}:
        raise GateFailure('Diagnostic file set changed since report')
    files={}
    for item in read(run/'artifact_manifest.json'):
        for name in item['paths']:
            p=Path(name)
            if not p.exists():raise GateFailure('Missing report artifact: '+name)
            if p.is_file():files[str(p.relative_to(run))]=dict(bytes=p.stat().st_size,sha256=sha(p))
    for name in ['Phase3_Diagnostics_Report.md','Phase3_Diagnostics_Report.json','Phase3_Diagnostics_Report.csv','artifact_manifest.json','artifact_manifest.csv','final_integrity.json','execution_summary.json']:
        p=run/name;files[name]=dict(bytes=p.stat().st_size,sha256=sha(p))
    for path,value in read(run/'input_refs.json')['frozen_paths'].items():
        if sha(path)!=value:raise GateFailure('Frozen reference changed')
    current=subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain=v1','--untracked-files=normal'],text=True).strip()
    if current!=integrity['worktree']:raise GateFailure('Worktree changed after report')
    guard.check()
    state=read(run/'stage_status.json')
    state.update(status='completed',last_verified_beijing=now(),tasks={s:'completed' for s in stage_states},
                 active_diagnostic_writers=[],active_monitoring=False,
                 monitoring_end_reason='All requested stages, report, and receipt verification completed; no task remains running',
                 delivery_receipt=str(run/'delivery_receipt.json'),resume_needed=False)
    save(run/'stage_status.json',state)
    files['stage_status.json']=dict(bytes=(run/'stage_status.json').stat().st_size,sha256=sha(run/'stage_status.json'))
    for s in stage_states:
        name=s+'/status.json';p=run/name
        files[name]=dict(bytes=p.stat().st_size,sha256=sha(p))
    save(run/'delivery_artifact_sha256.json',files)
    result=dict(status='completed',time_beijing=now(),boot_id=guard.boot,identity_sha=EXPECTED,
        HEAD=integrity['HEAD'],branch=integrity['branch'],worktree=current,
        verified_pair_receipts=verified_pairs,geometry_receipts=counts,unique_geometry_prediction_files=len(checked_inputs),
        verified_report_artifacts=len(files),artifact_hash_manifest=str(run/'delivery_artifact_sha256.json'),
        output_disk=disk(run),environment=guard.status,active_diagnostic_writers=active,
        optimizer_updates=0,checkpoint_writes=0,expansion_executed=False,active_monitoring=False,
        report=str(run/'Phase3_Diagnostics_Report.md'))
    save(run/'delivery_receipt.json',result)
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args();main(args.run)
