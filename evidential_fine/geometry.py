"""CPU canonical geometry for A/B full and epistemic95; baseline is immutable Phase3."""
from .common import *
from .statistics import select
import signal
sys.path.insert(0,str(ROOT/'diagnostics_phase3'))
from pose_statistics import verified_unit,mean_auc
from reliability_v1.geometry_worker import prediction_identity
def source_json(rel):
    index=read(P3/'delivery_artifact_sha256.json');p=P3/rel
    if rel not in index or sha(p)!=index[rel]['sha256']:raise IdentityFailure('Historical geometry aggregate checksum')
    return read(p)
def execute(run):
    run=Path(run);out=run/'geometry';out.mkdir(exist_ok=True)
    if (run/'degradation.json').exists() and 'Phase5 pose' in read(run/'degradation.json').get('skipped',[]):
        result=dict(status='skipped',reason='Preregistered measured-duration degradation',new_units=0);save(out/'summary.json',result);return result
    routes=[s for s in ['A','B'] if (run/('final_'+s)/'summary.json').exists()]
    if not routes:return dict(status='不可判定',reason='No completed route')
    protocol=geometry_protocol()
    if protocol!=source_json('A4/protocol.json'):raise IdentityFailure('Canonical geometry changed')
    oldplan=source_json('A4/geometry_plan.json');oldprep={x['pair_id']:x for x in source_json('A4/prediction_preparation.json')['rows']}
    save(out/'protocol.json',protocol);units={};baseline=[];names=[s+'_'+rate for s in routes for rate in ['all','95']]
    for r in records('epe'):
        a=get_unit(run,'final_'+routes[0],r,True)
        native=dict(points0_orig=a['native_source']*a['scale0'],points1_orig=a['native_target']*a['scale1'],K0=a['K0'],K1=a['K1'],T_0to1=a['T'])
        if prediction_identity(native,protocol)!=oldprep[r['pair_id']]['predictions']['baseline_100']['identity']:raise IdentityFailure('Phase5 native coordinate/order hash mismatch')
        for seed in range(5):
            x=verified_unit(oldplan,pairname(r['pair_id'])+'/baseline_100/'+str(seed))
            if x is None:raise IdentityFailure('Missing committed native baseline')
            baseline.append(dict(**base(r),seed=seed,error=x['pose_error_deg'],status=x['solver_status'],inliers=x['num_inliers'],matches=x['num_points']))
        for route in routes:
            a=get_unit(run,'final_'+route,r,True)
            for label in ['all','95']:
                ix=np.arange(len(a['cp'])) if label=='all' else select(a,-a['epistemic'],.95)[0]
                combo=route+'_'+label;pred=dict(points0_orig=a['source'][ix]*a['scale0'],points1_orig=a['target'][ix]*a['scale1'],K0=a['K0'],K1=a['K1'],T_0to1=a['T'],pair_id=np.array(r['pair_id']))
                p=out/'predictions'/combo/(pairname(r['pair_id'])+'.npz')
                if not p.exists():npzout(p,pred)
                elif prediction_identity(npzload(p),protocol)!=prediction_identity(pred,protocol):raise IdentityFailure('Geometry prediction changed')
                for seed in range(5):
                    key=pairname(r['pair_id'])+'/'+combo+'/'+str(seed);units[key]=dict(**base(r),combo=combo,seed=seed,prediction=str(p),prediction_sha256=sha(p),directory=pairname(r['pair_id'])+'/'+combo,independent_repeat=False)
    plan=dict(output=str(out/'units'),protocol=str(out/'protocol.json'),protocol_sha256=digest(protocol),units=units,formal_unit_order=list(units),formal_units=len(units),combos=names)
    if (out/'plan.json').exists() and read(out/'plan.json')!=plan:raise IdentityFailure('Frozen geometry plan changed')
    save(out/'plan.json',plan);save(out/'baseline_reuse.json',dict(status='pass',units=500,new_solves=0,rows=baseline))
    jobs=[]
    try:
        for i in range(3):
            keys=list(units)[i::3];j=out/f'jobs_{i}.json'
            if j.exists() and read(j)!=keys:raise IdentityFailure('Geometry worker partition changed')
            save(j,keys);cmd=['/home/bupt/miniconda3/envs/hybrid-eval/bin/python','-B','-u',str(ROOT/'diagnostics_phase3/pose_units.py'),'--plan',str(out/'plan.json'),'--jobs',str(j),'--worker-name',f'cpu_{i}']
            log=(out/f'worker_{i}.log').open('ab')
            p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1'))
            save(out/f'launch_{i}.json',dict(command=cmd,pid=p.pid,started=now()));jobs.append((p,log))
        while any(p.poll() is None for p,f in jobs):
            if any(p.poll() not in [None,0] for p,f in jobs):raise RuntimeError('Geometry worker failure')
            beat(run,'report',phase='geometry',workers=[dict(pid=p.pid,returncode=p.poll()) for p,f in jobs]);time.sleep(5)
        if any(p.returncode for p,f in jobs):raise RuntimeError('Geometry worker failure')
    finally:
        for p,f in jobs:
            if p.poll() is None:p.terminate()
            try:p.wait(10)
            except subprocess.TimeoutExpired:p.kill();p.wait()
            f.close()
    recs=records('epe');lookup={r['pair_id']:i for i,r in enumerate(recs)};errors={n:np.full((100,5),np.nan) for n in ['native']+names}
    for x in baseline:errors['native'][lookup[x['pair_id']],x['seed']]=x['error'] if x['status']=='success' else np.inf
    rows=[]
    for k,u in units.items():
        x=verified_unit(plan,k)
        if x is None:raise RuntimeError('Missing formal geometry unit')
        errors[u['combo']][lookup[u['pair_id']],u['seed']]=x['pose_error_deg'] if x['solver_status']=='success' else np.inf
        rows.append(dict(**base(u),combo=u['combo'],seed=u['seed'],error=x['pose_error_deg'],error_for_auc=x['pose_error_deg'] if x['solver_status']=='success' else '+inf',solver_status=x['solver_status'],inliers=x['num_inliers'],matches=x['num_points'],seconds=x['geometry_seconds']))
    means={n:mean_auc(e).tolist() for n,e in errors.items()};scenes=sorted({r['scene'] for r in recs});ixs=[np.array([i for i,r in enumerate(recs) if r['scene']==s]) for s in scenes]
    rng=np.random.default_rng(2026);boot={n:[] for n in names}
    for _ in range(2000):
        idx=np.concatenate([ixs[i] for i in rng.integers(len(scenes),size=len(scenes))]);b=mean_auc(errors['native'][idx])
        for n in names:boot[n].append(mean_auc(errors[n][idx])-b)
    unit_summary={}
    for name in ['native']+names:
        selected=baseline if name=='native' else [x for x in rows if x['combo']==name]
        success=[x for x in selected if x.get('solver_status',x.get('status'))=='success']
        unit_summary[name]=dict(units=len(selected),pairs=len({x['pair_id'] for x in selected}),successes=len(success),executed_failures=len(selected)-len(success),
            matches=distributions(np.array([x['matches'] for x in selected])),inliers_all=distributions(np.array([x['inliers'] for x in selected])),
            inliers_success=distributions(np.array([x['inliers'] for x in success])))
    result=dict(status='completed',new_units=len(rows),reused_native_units=500,unit_summary=unit_summary,missing=0,executed_failures=sum(x['solver_status']!='success' for x in rows),
        AUC_5_10_20=means,paired_differences={n:dict(mean_difference=(np.array(means[n])-means['native']).tolist(),ci95=np.quantile(np.asarray(v),[.025,.975],axis=0).T.tolist(),pairs=100,scenes=15) for n,v in boot.items()},
        per_scene={s:{n:mean_auc(e[ix]).tolist() for n,e in errors.items()} for s,ix in zip(scenes,ixs)},protocol=protocol)
    save(out/'summary.json',result);csvout(out/'geometry_units.csv',rows);return result
