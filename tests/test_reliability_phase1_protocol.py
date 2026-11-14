import copy,json,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from reliability_v1.comparison import ranks,spearman,sparsification,parent_errors,scene_bootstrap,final_source_errors,evaluate
from reliability_v1.scene_protocol import uniform_indices,plan_indices,parent_indices,skip_guard,scene_alerts
from reliability_v1.scalar_lr import ScalarStore,fit_cumulative,lr_logits
from reliability_v1.runtime import save_checkpoint,load_checkpoint,atomic_json

class Phase1Contracts(unittest.TestCase):
    def test_spearman_order_and_undefined(self):
        self.assertAlmostEqual(spearman([1,2,3],[1,2,3]),1)
        self.assertAlmostEqual(spearman([3,2,1],[1,2,3]),-1)
        for a,b in [([1,1],[1,2]),([1,2],[1,1]),([1],[2]),([],[])]:self.assertIsNone(spearman(a,b))
    def test_ause_oracle_inversion_rounding_and_ties(self):
        e=np.arange(1,8,dtype=float);ids=np.arange(7)[::-1]
        ideal=sparsification(-e,e,ids);bad=sparsification(e,e,ids)
        self.assertAlmostEqual(ideal["AUSE"],0);self.assertGreater(bad["AUSE"],0)
        self.assertEqual(ideal["counts"][0],1);self.assertEqual(ideal["counts"][4],4)
        self.assertEqual(ideal["curve"][-1],bad["curve"][-1])
        tied=sparsification(np.ones(7),e,ids)
        self.assertEqual(tied["curve"][0],7.)
    def test_pair_auc_ap_single_class_and_constant_scores(self):
        for y in [[],[1],[1,1],[0,0]]:
            r=ranks(y,np.ones(len(y)));self.assertIsNone(r["AP"]);self.assertIsNone(r["score_AUROC"])
        r=ranks([0,1,0,1],[1,1,1,1]);self.assertAlmostEqual(r["score_AUROC"],.5);self.assertAlmostEqual(r["AP"],.5)
        self.assertEqual(ranks([0,1],[0,1])["AP"],1.)
    def test_parent_median_and_missing_counts(self):
        x=parent_errors([1,2,3],[1,1,2,2],[2,6,5,7],[1,1,0,0])
        np.testing.assert_array_equal(x["parent_ids"],[1]);np.testing.assert_array_equal(x["error"],[4])
        self.assertEqual(x["counts"]["parents_without_fine_output"],1)
        self.assertEqual(x["counts"]["parents_with_no_valid_final_gt"],1)
        self.assertEqual(x["counts"]["multi_child_parents"],2)
        self.assertEqual(x["counts"]["invalid_final_gt_children"],2)
    def test_scene_paired_bootstrap_and_common_pairs(self):
        rows=[dict(scene="a",methods={"head":{"AP":.9},"lr":{"AP":.1}}) for _ in range(9)]
        rows+=[dict(scene="b",methods={"head":{"AP":.1},"lr":{"AP":.3}}),
               dict(scene="c",methods={"head":{"AP":None},"lr":{"AP":.7}})]
        result=scene_bootstrap(rows,"AP","head","lr")
        self.assertAlmostEqual(result["mean_difference"],.3);self.assertEqual(result["pair_count"],10)
        np.testing.assert_allclose(result["ci95"],[-.2,.8]);self.assertEqual(result,scene_bootstrap(rows,"AP","head","lr"))
        self.assertIsNone(scene_bootstrap(rows[:9],"AP","head","lr")["ci95"])
    def test_sampling_full_small_scene_and_epoch_redraw(self):
        ranges=[dict(scene="a",start=0,count=7),dict(scene="b",start=7,count=10000)]
        first=plan_indices(ranges,0);second=plan_indices(ranges,1)
        self.assertEqual(len(first),107);self.assertEqual(len(set(first)),107);self.assertTrue(set(range(7))<=set(first))
        self.assertEqual(first,plan_indices(ranges,0));self.assertNotEqual(first,second)
        self.assertEqual(len(uniform_indices(1000000000,100,2026)),100)
    def test_sampling_checkpoint_restores_next_update(self):
        with tempfile.TemporaryDirectory() as t:
            torch.manual_seed(2026);model=torch.nn.Linear(3,1);opt=torch.optim.AdamW(model.parameters(),lr=.01)
            def step(ids):
                opt.zero_grad();x=torch.tensor(np.stack([ids,ids*.1,ids*.01],-1),dtype=torch.float32)
                loss=(model(x).flatten()-.5).square().mean();loss.backward();opt.step();return float(loss)
            raw=np.arange(2000);valid=np.ones(2000,bool);cursor=dict(epoch=1,next_pos=2)
            ids=parent_indices(valid,raw,2026,cursor["epoch"],"pair")
            path=Path(t)/"checkpoint.json";save_checkpoint(path,model,opt,cursor,{"test":"isolated"})
            a=step(ids);params=copy.deepcopy(model.state_dict())
            loaded=load_checkpoint(path,model,opt,{"test":"isolated"});self.assertEqual(loaded["position"],cursor)
            next_ids=parent_indices(valid,raw,2026,loaded["position"]["epoch"],"pair")
            np.testing.assert_array_equal(ids,next_ids);b=step(next_ids);self.assertEqual(a,b)
            for key in params:torch.testing.assert_close(params[key],model.state_dict()[key],atol=0,rtol=0)
    def test_skip_boundary_and_no_window_extension(self):
        state={}
        for i in range(100):state=skip_guard(state,"invalid" if i<20 else None)
        self.assertFalse(state["pause"]);state=skip_guard(state,"invalid");self.assertFalse(state["pause"])
        state={}
        for i in range(21):state=skip_guard(state,"invalid")
        self.assertTrue(state["pause"]);self.assertEqual(state["planned_visits"],21)
    def test_label_alert_extreme_rules(self):
        rows=[dict(scene="a",positive=0,valid=100,ignore=1900,total=2000) for _ in range(5)]
        rows += [dict(scene="b",positive=50,valid=100,ignore=0,total=100) for _ in range(5)]
        result=scene_alerts(rows);self.assertIn("ignore_at_least_95pct",result["a"]["review_reasons"]);self.assertIn("positive_extreme",result["a"]["review_reasons"])
    def test_lr_pair_equal_cumulative_and_idempotent_resume(self):
        stats=dict(mean=[0,0,0],scale=[1,1,1]);rng=np.random.default_rng(2026)
        x=rng.normal(size=(80,3)).astype(np.float32);y=(x[:,0]-2*x[:,1]+.5*x[:,2]>0).astype(np.float32)
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);store=ScalarStore(root/"records",{"test":"scalar"})
            store.commit(0,0,"p0","0to1",np.arange(40),x[:40],y[:40])
            replay=store.commit(0,0,"p0","0to1",np.arange(40),x[:40],y[:40]);self.assertTrue(replay["reused"])
            store.commit(1,0,"p1","0to1",np.arange(40,80),x[40:],y[40:])
            self.assertEqual(store.signature(0)["visits"],1);self.assertEqual(store.signature(1)["visits"],2)
            result=fit_cumulative(store,1,stats,root/"fit.json")
            self.assertTrue(result["converged"]);self.assertGreater(result["parameters"][0],0);self.assertLess(result["parameters"][1],0);self.assertGreater(result["parameters"][2],0)
            self.assertGreater(((lr_logits(x,stats,result)>0)==y).mean(),.95)
            self.assertEqual(result,fit_cumulative(store,1,stats,root/"fit.json"))
            store.close()
    def test_pair_weight_not_parent_weight(self):
        stats=dict(mean=[0,0,0],scale=[1,1,1])
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);s=ScalarStore(root/"s",{})
            s.commit(0,0,"positive","0to1",np.arange(100),np.zeros((100,3)),np.ones(100))
            s.commit(0,1,"negative","0to1",np.arange(2),np.zeros((2,3)),np.zeros(2))
            fit=fit_cumulative(s,0,stats,root/"lr.json");self.assertAlmostEqual(fit["parameters"][3],0,places=6);s.close()
    def test_final_source_projection_not_old_coarse_source(self):
        plan=dict(scale_input_to_orig_xy=[1.,1.],orig_hw=[832,832],resized_hw=[832,832])
        transform=np.eye(4);transform[0,3]=8.
        data={"fine_coord_0":torch.tensor([[16.,16.]]),"fine_coord_1":torch.tensor([[24.,16.]]),
              "_gt":dict(plan0=plan,plan1=plan,depth0=np.ones((832,832)),depth1=np.ones((832,832)),
                          K0=np.eye(3),K1=np.eye(3),T_0to1=transform,depth_absolute_tolerance=.01,depth_relative_tolerance=.05)}
        error,valid,gt=final_source_errors(data);self.assertTrue(valid[0]);self.assertAlmostEqual(error[0],0)
        data["fine_coord_0"]=torch.tensor([[8.,16.]]);wrong,_,_=final_source_errors(data);self.assertAlmostEqual(wrong[0],8)
    def test_evaluation_actual_load_join_output_resume(self):
        records=[dict(pair_id="p0",scene="a",direction="0to1"),dict(pair_id="p1",scene="b",direction="0to1")]
        def prediction(r):
            return dict(raw_id=np.array([10,20,30]),head=np.array([2.,-2.,1.]),lr=np.array([1.,-1.,0.]),mconf=np.array([.9,.2,.5]),labels=np.array([1,0,-1]),low=np.array([1,1,0]),
                        child_parent_id=np.array([10,10,20]),child_epe=np.array([1.,3.,5.]),child_valid=np.array([1,1,1]))
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);coarse=evaluate(records,prediction,root/"coarse",{"test":"eval"})
            self.assertEqual(coarse["status"],"committed");self.assertEqual(coarse["pooled"]["parent_count"],4)
            final=evaluate(records,prediction,root/"final",{"test":"eval"},"final_epe")
            self.assertEqual(final["rows"][0]["counts"]["evaluable_parents"],2)
            replay=evaluate(records,lambda r: (_ for _ in ()).throw(AssertionError("must reuse commit")),root/"final",{"test":"eval"},"final_epe")
            self.assertEqual(final,replay)
    def test_launch_dry_run_is_read_only(self):
        from reliability_phase2 import dry_plan
        value=dry_plan(dict(repo_root="/a",run_dir="/b"),Path("/cfg"),"pipeline",True)
        self.assertEqual(value["stages"],["full_train","final_epe","report"]);self.assertTrue(value["dry_run"]);self.assertEqual(value["formal_optimizer_updates"],0)

    def test_formal_readiness_rejects_smoke_partial_and_overlap(self):
        from reliability_v1.phase2_runtime import validate_readiness_evidence,REQUIRED_CHECKS
        from reliability_v1.scene_protocol import PROTOCOL
        import copy
        receipt=dict(status="ready",scope="phase1_full_data_scene_protocol",checks={k:dict(status="pass") for k in REQUIRED_CHECKS})
        summary=dict(audit_complete=True,committed_indices=368,expected_indices=368,counts={},train_scenes=["a"],val_scenes=["b"],excluded_test_scenes=["0015","0022"])
        freeze=dict(status="pass",protocol=PROTOCOL,epoch_plans=[{}]*10,final_epe_pairs=100)
        review=dict(status="pass",unresolved_scenes=[]);cost=dict(status="pass",remaining_storage_peak_bytes=123)
        validate_readiness_evidence(receipt,summary,freeze,review,cost)
        for change in [dict(scope="engineering_smoke"),dict(checks={}),dict(status="blocked")]:
            with self.assertRaises(RuntimeError):validate_readiness_evidence(dict(receipt,**change),summary,freeze,review,cost)
        for change in [dict(committed_indices=367),dict(val_scenes=["a"]),dict(train_scenes=["0015"]),dict(counts=dict(missing_indices=1))]:
            with self.assertRaises(RuntimeError):validate_readiness_evidence(receipt,dict(summary,**change),freeze,review,cost)
        with self.assertRaises(RuntimeError):validate_readiness_evidence(receipt,summary,freeze,dict(status="blocked"),cost)

    def test_formal_invalid_position_is_retained_io_is_fatal(self):
        from reliability_v1.phase2_runtime import supervised_visit
        from types import SimpleNamespace
        import tempfile
        from pathlib import Path
        def fail(record,**kwargs):raise RuntimeError("Frozen data checksum mismatch: synthetic")
        runtime=SimpleNamespace(forward=fail)
        with tempfile.TemporaryDirectory() as d:
            result=supervised_visit(runtime,dict(pair_id="pair"),0,17,None,Path(d))
            self.assertFalse(result["updated"]);self.assertTrue((Path(d)/"00_000017.json").exists())
            runtime.forward=lambda *a,**k: (_ for _ in ()).throw(AssertionError("invalid visit recomputed"))
            self.assertEqual(result,supervised_visit(runtime,dict(pair_id="pair"),0,17,None,Path(d)))
            runtime.forward=lambda *a,**k: (_ for _ in ()).throw(OSError(5,"synthetic system IO"))
            with self.assertRaises(OSError):supervised_visit(runtime,dict(pair_id="pair"),0,18,None,Path(d))
            self.assertFalse((Path(d)/"00_000018.json").exists())

    def test_failed_lr_fit_is_not_a_reusable_phase_commit(self):
        from reliability_v1.scalar_lr import ScalarStore,fit_cumulative
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);store=ScalarStore(root/"records",{"scope":"synthetic_failed_fit"})
            x=np.column_stack([np.linspace(-2,2,32),np.zeros(32),np.zeros(32)]).astype(np.float32)
            store.commit(0,0,"p","0to1",np.arange(32),x,(x[:,0]>0).astype(np.float32))
            stats={"mean":[0,0,0],"scale":[1,1,1]}
            with self.assertRaises(RuntimeError):fit_cumulative(store,0,stats,root/"fit.json",maxiter=1)
            result=fit_cumulative(store,0,stats,root/"fit.json")
            self.assertTrue(result["converged"]);self.assertEqual(result["status"],"committed")
            store.close()

if __name__=="__main__":unittest.main()
