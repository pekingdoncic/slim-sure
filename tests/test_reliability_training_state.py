"""CPU state tests for trainer/checkpoint APIs, without model/GPU runs.

Formal prerequisites use a mock here because these tiny state-machine fixtures
do not represent a full audit. test_reliability_formal_gate exercises the real
fail-closed gate independently; this suite cannot authorize formal training.
"""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from reliability_v1.runtime import (atomic_json,save_checkpoint,load_checkpoint,state_hash,seed_all,append_json)
from reliability_v1.trainer import full_train,permutation


class CPUStateContracts(unittest.TestCase):
    def setUp(self):
        self.stack=ExitStack()
        self.formal_gate=self.stack.enter_context(patch(
            "reliability_v1.formal_gate.verify_formal_prerequisites",
            return_value={"status":"verified","scope":"synthetic_state_test_only"}))
        self.stack.enter_context(patch("torch.cuda.get_rng_state_all",return_value=[]))
        self.stack.enter_context(patch("torch.cuda.set_rng_state_all"))
        self.stack.enter_context(patch("torch.cuda.manual_seed_all"))
        torch.set_num_threads(1)

    def tearDown(self):
        self.stack.close()

    def runtime(self,path,epochs=1):
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        seed_all(2026)
        head=torch.nn.Linear(2,1)
        rt=SimpleNamespace(run=path,config={"seed":2026,"epochs":epochs},head=head,
            optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001),
            calls={"backbone":0,"head_forward":0,"head_backward":0},load_seconds=0.,
            check_frozen=lambda:True)
        atomic_json(path/"engineering_acceptance.json",{"status":"pass","identity":{"case":"state_test"}})
        return rt

    def records(self,n):
        return [{"pair_id":"pair_%05d"%i,"scene":"train_scene","index":i} for i in range(n)]

    def test_checkpoint_restores_rng_optimizer_and_next_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            rt=self.runtime(tmp)
            x=torch.tensor([[1.,2.],[-2.,1.]])
            def update():
                rt.optimizer.zero_grad(set_to_none=True)
                target=torch.rand(2,1)+float(np.random.rand())+random.random()
                loss=(rt.head(x)-target).square().mean()
                loss.backward()
                grads=[p.grad.clone() for p in rt.head.parameters()]
                rt.optimizer.step()
                return loss.item(),grads,[p.detach().clone() for p in rt.head.parameters()]
            update()
            cursor={"epoch":2,"next_pos":17,"global_step":33,"order":[2,0,1]}
            path=Path(tmp)/"last.json"
            save_checkpoint(path,rt.head,rt.optimizer,cursor,{"case":"state_test"})
            expected=update()
            restored=load_checkpoint(path,rt.head,rt.optimizer,{"case":"state_test"})
            actual=update()
            self.assertEqual(restored["position"],cursor)
            self.assertEqual(expected[0],actual[0])
            for a,b in zip(expected[1]+expected[2],actual[1]+actual[2]):
                self.assertTrue(torch.equal(a,b))

    def test_identity_and_checksum_fail_before_parameter_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            rt=self.runtime(tmp);path=Path(tmp)/"last.json"
            save_checkpoint(path,rt.head,rt.optimizer,{"epoch":0,"next_pos":0},{"case":"state_test"})
            before=state_hash(rt.head)
            with self.assertRaisesRegex(RuntimeError,"identity"):
                load_checkpoint(path,rt.head,rt.optimizer,{"case":"wrong"})
            self.assertEqual(before,state_hash(rt.head))
            payload=Path(tmp)/json.loads(path.read_text())["payload"]
            with payload.open("ab") as f:f.write(b"corrupt")
            with self.assertRaisesRegex(RuntimeError,"checksum"):
                load_checkpoint(path,rt.head,rt.optimizer,{"case":"state_test"})
            self.assertEqual(before,state_hash(rt.head))

    def test_complete_epochs_exceed_legacy_pair_and_step_caps(self):
        records=self.records(3001);val=self.records(7);visited=[];validations=[]
        def step(rt,record,epoch,position,log_path):
            visited.append((epoch,position,record["pair_id"]))
            return {"updated":True,"loss":.5,"seconds":.001,"resources":{"allocated":0,"reserved":0,"peak_allocated":0,"peak_reserved":0,"rss_bytes":64*1024**2,"rss_peak_bytes":64*1024**2}},None
        def validation(rt,rows,out,checkpoint_id,identity):
            validations.append([r["pair_id"] for r in rows])
            return {"pair_bce":.5}
        with tempfile.TemporaryDirectory() as tmp,patch("reliability_v1.trainer.train_step",side_effect=step),patch("reliability_v1.trainer.validate",side_effect=validation):
            rt=self.runtime(tmp,epochs=2)
            final=full_train(rt,records,val,{"case":"state_test"})
            self.formal_gate.assert_called_once_with(rt.config,rt.run,{"case":"state_test"})
            self.assertEqual(final["attempted_pairs"],6002)
            self.assertEqual(final["global_step"],6002)
            self.assertEqual(final["epoch"],2)
            self.assertEqual(final["best_epoch"],0)  # Equal BCE keeps the earlier epoch.
            self.assertEqual(len(validations),3)
            self.assertTrue(all(len(v)==7 for v in validations))
            for epoch in range(2):
                observed=[pid for e,_,pid in visited if e==epoch]
                self.assertEqual(len(observed),3001)
                self.assertEqual(set(observed),{r["pair_id"] for r in records})
                self.assertEqual(observed,[records[int(i)]["pair_id"] for i in permutation(3001,2026,epoch)])
                coverage=json.loads((Path(tmp)/"full_train"/("epoch_%02d_coverage.json"%epoch)).read_text())
                self.assertEqual(coverage["coverage"],1)
                self.assertEqual(coverage["unique_pairs_seen"],3001)

    def test_interrupted_cursor_replays_only_uncommitted_suffix(self):
        records=self.records(1107);identity={"case":"state_test"}
        def cpu_step(rt,record,epoch,position,log_path):
            rt.optimizer.zero_grad(set_to_none=True)
            x=torch.tensor([[1.,(record["index"]%13)/13.]])
            target=torch.rand(1,1)+float(np.random.rand())+random.random()
            loss=(rt.head(x)-target).square().mean()
            loss.backward();rt.optimizer.step()
            append_json(log_path,{"pair_id":record["pair_id"],"position":position})
            return {"updated":True,"loss":loss.item(),"seconds":.001,"resources":{"allocated":0,"reserved":0,"peak_allocated":0,"peak_reserved":0,"rss_bytes":64*1024**2,"rss_peak_bytes":64*1024**2}},None
        with tempfile.TemporaryDirectory() as tmp,patch("reliability_v1.trainer.validate",return_value={"pair_bce":.5}):
            reference=self.runtime(Path(tmp)/"reference")
            with patch("reliability_v1.trainer.train_step",side_effect=cpu_step):
                full_train(reference,records,records[:4],identity)
            expected={k:v.detach().clone() for k,v in reference.head.state_dict().items()}
            interrupted=self.runtime(Path(tmp)/"resume")
            def fail_after_checkpoint(rt,record,epoch,position,log_path):
                if position==1037:raise RuntimeError("simulated process interruption")
                return cpu_step(rt,record,epoch,position,log_path)
            with patch("reliability_v1.trainer.train_step",side_effect=fail_after_checkpoint):
                with self.assertRaisesRegex(RuntimeError,"simulated"):
                    full_train(interrupted,records,records[:4],identity)
            saved=load_checkpoint(interrupted.run/"full_train/last.json",interrupted.head,interrupted.optimizer,identity)
            self.assertEqual(saved["position"]["next_pos"],1000)
            resumed=self.runtime(interrupted.run)
            with patch("reliability_v1.trainer.train_step",side_effect=cpu_step):
                final=full_train(resumed,records,records[:4],identity,resume=True)
            self.assertEqual(final["attempted_pairs"],1107)
            self.assertEqual(final["global_step"],1107)
            for key,value in resumed.head.state_dict().items():
                self.assertTrue(torch.equal(value,expected[key]),key)
            physical=[json.loads(line) for line in (resumed.run/"full_train/physical_visits.jsonl").read_text().splitlines()]
            self.assertEqual(len(physical),1107+37)
            counts={i:sum(r["position"]==i for r in physical) for i in (999,1000,1036,1037,1106)}
            self.assertEqual(counts,{999:1,1000:2,1036:2,1037:1,1106:1})

    def test_stale_engineering_receipt_blocks_after_formal_gate(self):
        # Isolate the trainer's second identity check after prerequisite approval.
        with tempfile.TemporaryDirectory() as tmp:
            rt=self.runtime(tmp)
            with self.assertRaisesRegex(RuntimeError,"stale"):
                full_train(rt,self.records(5),self.records(2),{"case":"different"})
            self.formal_gate.assert_called_once_with(rt.config,rt.run,{"case":"different"})


if __name__=="__main__":unittest.main()
