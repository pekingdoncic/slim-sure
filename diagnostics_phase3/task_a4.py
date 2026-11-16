"""A4 serial-after-C driver with a 20-unit measured pilot, then three bounded CPU workers."""
from common import *
import subprocess,contextlib
from pose_prepare import prepare
from pose_statistics import summarize,verified_unit
from reliability_v1.metrics import pose_auc
PYTHON='/home/bupt/miniconda3/envs/hybrid-eval/bin/python'
def spawn_jobs(run,plan,keys,name):
    out=Path(run)/'A4';path=out/(name+'_jobs.json')
    if path.exists() and read(path)!=keys:raise GateFailure('Frozen worker job list changed')
    if not path.exists():save(path,keys)
    cmd=[PYTHON,'-B','-u',str(ROOT/'diagnostics_phase3/pose_units.py'),'--plan',str(out/'geometry_plan.json'),'--jobs',str(path),'--worker-name',name]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
    log=(out/(name+'.log')).open('ab')
    child=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env)
    save(out/(name+'_launch.json'),dict(command=cmd,pid=child.pid,started=now(),CUDA_VISIBLE_DEVICES=''))
    return child,log
def wait_jobs(run,jobs,phase):
    out=Path(run)/'A4'
    while any(p.poll() is None for p,f in jobs):
        progress={}
        for p,f in jobs:
            if p.poll() not in (None,0):raise RuntimeError('Geometry worker failed with returncode '+str(p.returncode))
        for path in (out/'geometry').glob('*_heartbeat.json'):
            progress[path.stem]=read(path)
        heartbeat(run,'A4',phase=phase,workers=progress)
        time.sleep(5)
    for p,f in jobs:
        f.close()
        if p.returncode:raise RuntimeError('Geometry worker failed with returncode '+str(p.returncode))
def main(run):
    run=Path(run);out=run/'A4'
    if read(run/'A/A0_reproduction.json').get('status')!='pass':raise GateFailure('A0 must pass for A4')
    if (run/'C/status.json').exists() and read(run/'C/status.json').get('status')=='running':raise RuntimeError('C must finish before A4')
    if (run/'B/status.json').exists() and read(run/'B/status.json').get('status')=='running':raise RuntimeError('B must finish before A4')
    scale=dict(zeros=pose_auc(np.zeros(5)),above_all_thresholds=pose_auc(np.full(5,100.)),failures=pose_auc(np.full(5,np.inf)))
    if any(v!=100 for v in scale['zeros'].values()) or any(v!=0 for k in ['above_all_thresholds','failures'] for v in scale[k].values()):raise GateFailure('AUC scale test failed')
    save(out/'AUC_scale_preflight.json',dict(status='pass',scale='0-100 already; no conversion multiplier',**scale))
    recs=records('epe');plan=prepare(run,recs);jobs=[]
    try:
        timing=plan['initial_timing_units']
        if not all(verified_unit(plan,k) is not None for k in timing):
            jobs=[spawn_jobs(run,plan,timing,'timing20')];wait_jobs(run,jobs,'first 20 prespecified formal geometry units');jobs=[]
        sampled=[verified_unit(plan,k) for k in timing];times=[r['geometry_seconds'] for r in sampled]
        remaining=sum(verified_unit(plan,k) is None for k in plan['formal_unit_order'])
        eta=dict(time_beijing=now(),timed_units=20,mean_geometry_seconds=float(np.mean(times)),median_geometry_seconds=float(np.median(times)),max_geometry_seconds=max(times),
            remaining_formal_units=remaining,cpu_workers=3,estimated_remaining_serial_solver_seconds=remaining*float(np.mean(times)),
            estimated_remaining_wall_seconds_at_3_workers=remaining*float(np.mean(times))/3,
            assumptions='Measured first 20 fixed spread-out formal units; 3 CPU workers; estimate excludes remaining input/commit/bootstrap overhead and is not a tail-latency guarantee.',
            samples=[dict(unit=k,scene=plan['units'][k]['scene'],method=plan['units'][k]['method'],rate=plan['units'][k]['rate'],seed=plan['units'][k]['seed'],seconds=r['geometry_seconds']) for k,r in zip(timing,sampled)])
        if not (out/'first20_timing_and_ETA.json').exists():save(out/'first20_timing_and_ETA.json',eta)
        print(json.dumps(dict(event='first20_complete',**{k:v for k,v in eta.items() if k!='samples'})),flush=True)
        # The partition is fixed independently of observed runtime or solver outcomes.
        formal=[k for k in plan['formal_unit_order'] if k not in set(timing)]
        keys=formal+plan['independent_repeat_units']
        for worker in range(3):jobs.append(spawn_jobs(run,plan,keys[worker::3],'cpu_worker_'+str(worker)))
        wait_jobs(run,jobs,'formal geometry + independent baseline rechecks');jobs=[]
    finally:
        # Children share this task worker's private process group for supervisor storage-fault stop.
        for child,log in jobs:
            if child.poll() is None:child.terminate()
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            log.close()
    return summarize(run,plan,recs)
