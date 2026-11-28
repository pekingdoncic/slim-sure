import csv
import json
import os
import subprocess
import sys
from pathlib import Path
import pytest
import torch
from experiments.lowtex_pro5000_hr_aug.data.dataset_adapter import split_check,TrainPool,manifest
from experiments.lowtex_pro5000_hr_aug.training.options import parser,plan
from experiments.lowtex_pro5000_hr_aug.training.pipeline import Pipeline
from experiments.lowtex_pro5000_hr_aug.training.optimizer import optimizer,freeze_phase
from experiments.lowtex_pro5000_hr_aug.utils.run_io import EXP,ROOT,locations,write,read,config,atomic_path
from experiments.lowtex_pro5000_hr_aug.evaluation.summarize_results import summarize


def test_official_pool_split_and_sampler_resume(tmp_path):
    pool=split_check()
    assert pool["legal_rows"]==sum(b["legal_rows"] for b in pool["bins"])
    assert {r["pair_id"] for r in manifest("texturepoor_smoke_val")}<={r["pair_id"] for r in manifest("texturepoor_val_pairs")}
    d=TrainPool(tmp_path,tmp_path,1024,config("P1"))
    d.size=127
    seq=[d.row_at(i) for i in range(254)]
    assert len(set(seq[:127]))==len(set(seq[127:]))==127
    e=TrainPool(tmp_path,tmp_path,1216,config("P4"))
    e.size=127
    assert [e.row_at(i) for i in range(80,190)]==seq[80:190]


def test_full_budget_and_isolated_roots():
    args=parser().parse_args(["--mode","full","--full-epochs","1"])
    sides,_,steps=plan(args)
    assert sides==[1024,1216] and sum(steps)*16>=manifest("train_pool")["legal_rows"]
    with pytest.raises(ValueError,match="cover"):
        plan(parser().parse_args(["--mode","full","--steps-1024","1","--steps-1216","1"]))
    assert locations("smoke","one","P1")[0]!=locations("full","one","P1")[0]
    assert locations("smoke","one","P1")[0]!=locations("smoke","two","P1")[0]
    with pytest.raises(ValueError):
        locations("smoke","../other","P0")


def test_optimizer_checkpoint_transition_matches_uninterrupted(tmp_path):
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.core=torch.nn.Module()
            self.core.feature_backbone=torch.nn.Sequential(torch.nn.Linear(2,2))
            self.head=torch.nn.Linear(2,1)
            self.cfg=config("P3")
        def forward(self,x):
            return self.head(self.core.feature_backbone(x))
    def update(model,opt,scheduler,step):
        warm=freeze_phase(model,step,4)
        opt.zero_grad()
        model(torch.ones(2,2)).square().mean().backward()
        opt.step()
        scheduler.step()
        return warm
    model=Toy()
    opt,sched=optimizer(model,4)
    assert update(model,opt,sched,0)
    dst=tmp_path/"checkpoint_last.ckpt"
    with atomic_path(dst) as path:
        torch.save(dict(model=model.state_dict(),opt=opt.state_dict(),scheduler=sched.state_dict(),cursor=16),path)
    assert not update(model,opt,sched,1)
    restored=Toy()
    op,sc=optimizer(restored,4)
    state=torch.load(dst,weights_only=False)
    restored.load_state_dict(state["model"])
    op.load_state_dict(state["opt"])
    sc.load_state_dict(state["scheduler"])
    assert state["cursor"]==16
    update(restored,op,sc,1)
    for a,b in zip(model.parameters(),restored.parameters()):
        torch.testing.assert_close(a,b)


def test_shell_arguments_and_receipt_resume(tmp_path,monkeypatch):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHON=sys.executable)
    command=["bash",str(EXP/"scripts/run_full_pipeline.sh"),"--python",sys.executable,"--help"]
    help_text=subprocess.check_output(command,cwd=ROOT,env=env,text=True)
    for option in ("--resume-pipeline","--monitor","--dev-only","--steps-1024","--scannet-root","--weights"):
        assert option in help_text
    p=Pipeline.__new__(Pipeline)
    p.report=tmp_path
    p.progress=tmp_path/"progress.json"
    p.opts={}
    p.source=p.identity="unit"
    p.cache=tmp_path/"cache"
    p.state={"stages":{}}
    expected=tmp_path/"complete.json"
    write(expected,{"result":True})
    p.state["stages"]["cpu"]={"status":"PASS"}
    assert p.stage("cpu",cmd=["does-not-exist"],expected=expected)
    # A missing expected artifact forces execution, even with a previous PASS.
    assert p.stage("cpu",cmd=[sys.executable,"-c","from pathlib import Path; import sys; Path(sys.argv[1]).write_text('{}')",
                             str(tmp_path/"new.json")],expected=tmp_path/"new.json")
    assert read(p.progress)["stages"]["cpu"]["exit_code"]==0


def test_report_preserves_missing_and_blocked(tmp_path):
    root=tmp_path/"output"
    report=tmp_path/"report"
    write(root/"P1/id/test/scannet.json",{"status":"BLOCKED","reason":"missing index"})
    out=summarize("smoke","id",root,report)
    with (out/"final_comparison.csv").open() as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==5 and rows[0]["megadepth_status"]=="NOT_RUN"
    assert rows[1]["scannet_status"]=="BLOCKED" and rows[1]["scannet_auc5"]==""
    assert "Smoke" in (out/"EXECUTIVE_SUMMARY.md").read_text()
    for name in ("NEGATIVE_RESULTS.md","MODEL_SELECTION.md","training_coverage.csv","figures"):
        assert (out/name).exists()


def test_source_paths_portable():
    import re
    windows=re.compile(r"[A-Z]:[\\/][A-Za-z]",re.I)
    unix="/ho"+"me/bu"+"pt/"
    paths=[p for d in ("training","models","losses","data","scripts","evaluation","utils","configs","manifests")
           for p in (EXP/d).rglob("*") if p.suffix in (".py",".sh",".yaml",".json")]
    for path in paths:
        data=path.read_text()
        assert not windows.search(data) and unix not in data,path
