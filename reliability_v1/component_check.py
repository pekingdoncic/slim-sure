"""Persist synthetic component calls, using exactly the production functions."""
import copy,json,os,shutil,sqlite3,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from .runtime import atomic_json,save_checkpoint,load_checkpoint,state_hash,sha
from .scene_protocol import freeze_plan,parent_indices,skip_guard
from .scalar_lr import ScalarStore,fit_cumulative,lr_logits
from .comparison import evaluate
from .phase2_runtime import report

def run_checks(root,out):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=False);calls=[]
    class Dataset:
        def __init__(self):
            self.path=out/"synthetic_pool.jsonl";self.split="train";self.connection=sqlite3.connect(":memory:")
            self.connection.execute("CREATE TABLE pairs(scene TEXT,split TEXT)");self.records=[]
            for scene,n in [("a",7),("b",300),("c",155)]:
                for i in range(n):
                    self.records.append(dict(pair_id=f"{scene}_{i:04d}",scene=scene,direction="0to1"))
                    self.connection.execute("INSERT INTO pairs VALUES(?,?)",(scene,"train"))
            self.path.write_text("".join(json.dumps(r)+"\n" for r in self.records))
        def __len__(self):return len(self.records)
        def __getitem__(self,i):return self.records[i]
    data=Dataset();plan0,_=freeze_plan(data,out/"epoch_0.json",0);plan1,_=freeze_plan(data,out/"epoch_1.json",1)
    calls.append(dict(function="scene_protocol.freeze_plan",epochs=[0,1],visits=[len(plan0["visits"]),len(plan1["visits"])],unique_within_epoch=True))
    torch.manual_seed(2026);model=torch.nn.Linear(3,1);optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    selected=parent_indices(np.ones(2000,bool),np.arange(2000),2026,1,"synthetic")
    cursor=dict(epoch=1,next_pos=3,next_pair_id=plan1["visits"][3]["pair_id"],selected_raw_ids=selected.tolist())
    ptr=out/"resume.json";save_checkpoint(ptr,model,optimizer,cursor,{"scope":"synthetic_component_check"})
    def update():
        optimizer.zero_grad();x=torch.tensor(np.stack([selected/2000,selected/4000,selected/6000],axis=1),dtype=torch.float32)
        loss=model(x).square().mean();loss.backward();optimizer.step();return float(loss)
    a=update();params=copy.deepcopy(model.state_dict());state=load_checkpoint(ptr,model,optimizer,{"scope":"synthetic_component_check"})
    assert state["position"]==cursor
    b=update();assert a==b
    for k in params:torch.testing.assert_close(params[k],model.state_dict()[k],atol=0,rtol=0)
    calls.append(dict(function="save_checkpoint/load_checkpoint",status="pass",next_loss=a,restored_loss=b,parameter_max_abs=0.,formal_optimizer_updates=0,synthetic_optimizer_updates=2))
    store=ScalarStore(out/"lr_records",{"scope":"synthetic_component_check"})
    rng=np.random.default_rng(2026);means=[];squares=[]
    generated=[]
    for visit in range(100):
        z=rng.normal(size=(1024,3));x=np.stack([1/(1+np.exp(-z[:,0])),np.log1p(np.exp(z[:,1])),np.log1p(np.exp(z[:,2]))],axis=-1).astype(np.float32)
        y=(3*x[:,0]-2*x[:,1]+.8*x[:,2]>.6).astype(np.float32)
        generated.append((x,y));means.append(x.mean(0,dtype=np.float64));squares.append((x.astype(float)**2).mean(0))
    mean=np.mean(means,axis=0);std=np.sqrt(np.mean(squares,axis=0)-mean*mean);stats=dict(mean=mean.tolist(),scale=std.tolist(),scope="synthetic_train_only")
    atomic_json(out/"synthetic_standardization.json",stats)
    fit_results=[]
    for visit,(x,y) in enumerate(generated):
        store.commit(visit//50,visit%50,f"synthetic_{visit}","0to1",np.arange(1024),x,y)
        if visit in (49,99):fit_results.append(fit_cumulative(store,visit//50,stats,out/f"lr_epoch_{visit//50}.json"))
    replay=store.commit(0,0,"synthetic_0","0to1",np.arange(1024),*generated[0]);assert replay["reused"]
    signature=store.signature(1);assert signature["visits"]==100 and signature["parents"]==102400
    calls.append(dict(function="ScalarStore.commit/fit_cumulative",status="pass",fits=[{k:v for k,v in f.items() if k not in ("identity",)} for f in fit_results],
                      serialized=signature,bytes_per_parent=signature["bytes"]/signature["parents"],scope="synthetic benchmark, bounded 100 visits, not actual training"))
    records=[dict(pair_id=f"synthetic_eval_{i}",scene="a" if i<2 else "b",direction="0to1") for i in range(3)]
    def predict(record):
        return dict(raw_id=np.array([10,20,30,40]),head=np.array([2.,1.,-1.,-2.]),lr=np.array([1.,.5,-.5,-1.]),mconf=np.array([.9,.8,.2,.1]),
                    labels=np.array([1,1,0,0]),low=np.array([1,0,1,1]),child_parent_id=np.array([10,10,20,30]),child_epe=np.array([1.,3.,2.,7.]),child_valid=np.ones(4,bool))
    fixture=out/"report_fixture";coarse=evaluate(records,predict,fixture/"formal_train/validation_00",{"scope":"synthetic"},"coarse")
    final=evaluate(records,predict,fixture/"final_epe",{"scope":"synthetic"},"final_epe")
    evaluate(records,lambda r: (_ for _ in ()).throw(AssertionError("replay recomputed")),fixture/"final_epe",{"scope":"synthetic"},"final_epe")
    atomic_json(fixture/"formal_train/status.json",dict(status="committed",state=dict(epoch=10,best_epoch=0,scope="synthetic_fake_epoch_counter_only")))
    pointer=json.loads(ptr.read_text());shutil.copyfile(ptr,fixture/"formal_train/best.json");shutil.copyfile(ptr.parent/pointer["payload"],fixture/"formal_train"/pointer["payload"])
    report({},fixture)
    calls.append(dict(function="comparison.evaluate/coarse+final_epe and phase2_runtime.report",status="pass",pairs=3,scope="synthetic only; no native fine forwards",report=str(fixture/"phase2_report.json")))
    # Exercise the actual CLI supervisor with harmless fake child executables.
    fake=out/"supervisor_fixture";fake.mkdir();(fake/"run").mkdir()
    config=fake/"config.json";atomic_json(config,dict(repo_root=str(fake),run_dir=str(fake/"run")))
    child=fake/"reliability_phase2.py"
    child.write_text("import sys,json\nfrom pathlib import Path\nr=Path(__file__).parent/'run'\ns=sys.argv[1]\nwith (r/'calls.jsonl').open('a') as f:f.write(json.dumps(s)+'\\n')\nif s=='full_train':\n (r/'formal_train').mkdir(exist_ok=True)\n (r/'formal_train/status.json').write_text(json.dumps({'status':'committed','scope':'fake child no model'}))\nif s=='final_epe':sys.exit(7)\n")
    base=[sys.executable,str(root/"reliability_phase2.py"),"pipeline","--config",str(config)]
    before=sorted(str(p) for p in (fake/"run").rglob("*"));dry=subprocess.run([*base,"--dry-run"],capture_output=True,text=True,check=True)
    assert before==sorted(str(p) for p in (fake/"run").rglob("*"))
    (out/"launch_dry_run.json").write_text(dry.stdout)
    failed=subprocess.run(base,capture_output=True,text=True,timeout=60)
    (out/"supervisor_failure.log").write_text(failed.stdout+failed.stderr)
    assert failed.returncode!=0
    failure=json.loads((fake/"run/launch_state.json").read_text());assert failure["stage"]=="final_epe"
    first=[json.loads(s) for s in (fake/"run/calls.jsonl").read_text().splitlines()];assert first==["full_train","final_epe"]
    child.write_text(child.read_text().replace("if s=='final_epe':sys.exit(7)","if s=='final_epe':\n (r/'final_epe').mkdir(exist_ok=True)\n (r/'final_epe/state.json').write_text(json.dumps({'status':'committed','scope':'fake child'}))"))
    resumed=subprocess.run([*base,"--resume"],capture_output=True,text=True,timeout=60)
    (out/"supervisor_resume.log").write_text(resumed.stdout+resumed.stderr);assert resumed.returncode==0
    sequence=[json.loads(s) for s in (fake/"run/calls.jsonl").read_text().splitlines()];assert sequence==["full_train","final_epe","final_epe","report"]
    calls.append(dict(function="actual phase2 CLI supervisor",status="pass",scope="synthetic fake workers; formal model updates zero",calls=sequence,failure=failure,resume_skipped_completed_training=True,dry_run_side_effects=False))
    # Use production stop counter at the exact 20/21 boundary.
    guard={}
    for i in range(20):guard=skip_guard(guard,"invalid")
    assert not guard["pause"];guard=skip_guard(guard,"invalid");assert guard["pause"]
    calls.append(dict(function="skip_guard",status="pass",at20="continue",at21="pause"))
    result=dict(status="pass",calls=calls,formal_optimizer_updates=0,real_final_epe_executed=False,scope="isolated synthetic production-component invocations")
    atomic_json(out/"component_calls.json",result);store.close();data.connection.close();return result
