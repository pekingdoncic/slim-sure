"""Persistent CPU-only single-unit adapter around the frozen canonical geometry implementation."""
from pathlib import Path
import sys,os,json,argparse,time,hashlib,datetime
import numpy as np
ROOT=Path('/home/bupt/projects/SLiM-master');sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from reliability_v1.geometry_worker import ARRAY_KEYS,atomic_json,file_sha,validate_protocol,prediction_identity
def read(p):return json.loads(Path(p).read_text())
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--jobs',required=True);parser.add_argument('--worker-name',required=True)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CUDA must be hidden for geometry')
    plan=read(args.plan);jobs=read(args.jobs);protocol=read(plan['protocol']);validate_protocol(protocol)
    sys.path.insert(0,str(Path(protocol['hybrid_root']).parent))
    from hybrid_match.evaluation.pose import estimate_pose,backend_information
    from hybrid_match.evaluation.metrics import pose_error
    out=Path(plan['output']);atomic_json(out/(args.worker_name+'_backend.json'),backend_information('LO-RANSAC'))
    cachepath=None;pred=None;identity=None;sourcehash=None;new_calls=0;reused=0
    for index,key in enumerate(jobs):
        unit=plan['units'][key];folder=out/unit['directory'];folder.mkdir(parents=True,exist_ok=True)
        target=folder/('seed_%d.json'%unit['seed']);receipt=folder/('seed_%d_commit.json'%unit['seed'])
        path=Path(unit['prediction'])
        if str(path)!=cachepath:
            if file_sha(path)!=unit['prediction_sha256']:raise RuntimeError('Prediction input changed')
            with np.load(path,allow_pickle=False) as z:pred={k:z[k] for k in ARRAY_KEYS}
            identity=prediction_identity(pred,protocol);sourcehash=unit['prediction_sha256'];cachepath=str(path)
        one_identity=dict(identity,pair_id=unit['pair_id'],seed=unit['seed'])
        if target.exists() and receipt.exists():
            meta=read(receipt);row=read(target)
            if meta['sha256']!=file_sha(target) or row['identity']!=one_identity or row['status']!='committed':raise RuntimeError('Committed geometry changed')
            reused+=1;continue
        atomic_json(out/(args.worker_name+'_heartbeat.json'),dict(status='running',unit=key,pair_id=unit['pair_id'],seed=unit['seed'],time_beijing=now(),unix=time.time(),pid=os.getpid(),new_calls=new_calls,reused=reused))
        started=time.perf_counter()
        with (out/(args.worker_name+'_attempts.jsonl')).open('a') as f:f.write(json.dumps(dict(unit=key,status='started',time_beijing=now()))+'\n');f.flush();os.fsync(f.fileno())
        result=estimate_pose(pred['points0_orig'],pred['points1_orig'],pred['K0'],pred['K1'],protocol['geometry'],unit['seed'])
        if result['status']=='success':
            rotation,translation=pose_error(pred['T_0to1'],result['R'],result['t'])
            if not np.isfinite([rotation,translation]).all():raise FloatingPointError('Undefined valid GT pose error')
            error=max(rotation,translation)
        elif result['status']=='geometry_failure':rotation=translation=error=None
        else:raise RuntimeError('Unexpected canonical solver status')
        row=dict(schema='phase3_canonical_geometry_unit_v1',status='committed',identity=one_identity,prediction_sha256=sourcehash,
            solver_status=result['status'],failure_reason=result['failure_reason'],pose_error_deg=error,R_error_deg=rotation,t_error_deg=translation,
            num_inliers=int(result['num_inliers']),num_points=len(pred['points0_orig']),geometry_seconds=result['geometry_seconds'],
            unit_wall_seconds=time.perf_counter()-started,backend_invoked=bool(result['backend_invoked']),backend_seed=result['backend_seed'],
            backend_seed_controlled=result['backend_seed_controlled'],seed_control=result['seed_control'],resolved_geometry=result['resolved_geometry'],
            R=result['R'].tolist() if result['R'] is not None else None,t=result['t'].tolist() if result['t'] is not None else None,
            time_beijing=now(),unit=key,independent_repeat=unit.get('independent_repeat',False))
        atomic_json(target,row);atomic_json(receipt,dict(status='committed',sha256=file_sha(target),identity=one_identity))
        new_calls+=1
        with (out/(args.worker_name+'_attempts.jsonl')).open('a') as f:f.write(json.dumps(dict(unit=key,status='committed',geometry_seconds=result['geometry_seconds'],time_beijing=now()))+'\n');f.flush()
    atomic_json(out/(args.worker_name+'_heartbeat.json'),dict(status='completed',time_beijing=now(),pid=os.getpid(),new_calls=new_calls,reused=reused,total_units=len(jobs)))
    print(json.dumps(dict(status='completed',new_calls=new_calls,reused=reused)))
if __name__=='__main__':main()
