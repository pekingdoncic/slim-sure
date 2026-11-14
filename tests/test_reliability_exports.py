"""CPU-only atomic export/controller contracts; worker processes are mocked."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from reliability_v1.exports import (FilePredictor,save_prediction,load_prediction,worker_identity,MODES)
from reliability_v1.geometry_worker import atomic_json,digest,file_sha
from reliability_v1.evaluation import measure_costs


class ExportContracts(unittest.TestCase):
    def setup_files(self,tmp):
        root=Path(tmp);run=root/"run";run.mkdir()
        config=root/"config.json"
        atomic_json(config,{"repo_root":str(root),"run_dir":str(run)})
        atomic_json(run/"texture_calibration.json",{"status":"committed","threshold":.1})
        atomic_json(run/"watchdog.json",{"measured":True,"pair_timeout_seconds":120,"cold_load_timeout_seconds":300})
        identity={"data":{"test":"cpu"},"code":{"test":"cpu"}}
        payload=run/"best.payload.pt";payload.write_bytes(b"opaque-checkpoint-test-payload")
        pointer=run/"best.json"
        atomic_json(pointer,{"payload":payload.name,"sha256":file_sha(payload),"identity_sha":digest(identity)})
        return root,config,identity,pointer

    def test_prediction_roundtrip_and_checksum(self):
        prediction={"pair_id":"pair","coarse_match_id":np.array([[0,1,4],[0,3,9]],np.int64),
                    "labels":np.array([1,-1],np.int8),"scores":np.array([.7,.2],np.float32),
                    "timing":{"model":np.float64(.2)},"memory":{"peak":123}}
        with tempfile.TemporaryDirectory() as tmp:
            save_prediction(tmp,prediction,{"id":"x"})
            result=load_prediction(tmp,{"id":"x"})
            self.assertFalse(result["executed_forward"])
            for key in ("coarse_match_id","labels","scores"):
                self.assertEqual(result[key].dtype,prediction[key].dtype)
                np.testing.assert_array_equal(result[key],prediction[key])
            with self.assertRaises(ValueError):load_prediction(tmp,{"id":"different"})
            with (Path(tmp)/"arrays.npz").open("ab") as f:f.write(b"bad")
            with self.assertRaises(ValueError):load_prediction(tmp,{"id":"x"})

    def test_one_worker_exports_three_modes_then_reuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,config,identity,pointer=self.setup_files(tmp)
            launches=[]
            class FakePopen:
                pid=12345
                def __init__(self,command,**kwargs):
                    launches.append(command)
                    request=json.loads(Path(command[command.index("--request")+1]).read_text())
                    out=Path(request["output"]);paths=[]
                    for mode in MODES:
                        save_prediction(out/mode,{"pair_id":request["record"]["pair_id"],"labels":np.array([1]),
                            "points0_orig":np.array([[1.,2.]]),"mode":mode},dict(request["identity"],mode=mode))
                        paths.extend(mode+"/"+name for name in ("arrays.npz","metadata.json","commit.json"))
                    atomic_json(out/"commit.json",{"status":"committed","identity":request["identity"],
                        "sha256":{name:file_sha(out/name) for name in paths}})
                def poll(self):return 0
                def wait(self,timeout=None):return 0
            predictor=FilePredictor(config,identity,root/"exports",pointer)
            record={"pair_id":"unique-one","scene":"train-independent-val"}
            with patch("reliability_v1.exports.subprocess.Popen",FakePopen):
                results=[predictor(record,mode) for mode in MODES]
                self.assertEqual([r["mode"] for r in results],list(MODES))
                self.assertEqual(len(launches),1)
                predictor(record,"A0")
                self.assertEqual(len(launches),1)
            state=json.loads((root/"exports/controller_state.json").read_text())
            self.assertEqual(state["successful_export_workers"],1)
            self.assertEqual(state["maximum_unique_pairs_per_worker"],1)

    def test_numeric_failure_is_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,config,identity,pointer=self.setup_files(tmp);launches=[]
            class BadPopen:
                pid=12345
                def __init__(self,command,**kwargs):
                    launches.append(command)
                    attempt=Path(command[command.index("--attempt-dir")+1])
                    atomic_json(attempt/"failure.json",{"retryable":False,"error":"nonfinite model coordinate"})
                def poll(self):return 1
                def wait(self,timeout=None):return 1
            predictor=FilePredictor(config,identity,root/"exports",pointer)
            with patch("reliability_v1.exports.subprocess.Popen",BadPopen):
                with self.assertRaises(RuntimeError):
                    predictor({"pair_id":"failure","scene":"val"},"A0")
            self.assertEqual(len(launches),1)

    def test_one_transient_retry_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,config,identity,pointer=self.setup_files(tmp);launches=[]
            class TransientPopen:
                pid=12345
                def __init__(self,command,**kwargs):
                    launches.append(command)
                    attempt=Path(command[command.index("--attempt-dir")+1])
                    atomic_json(attempt/"failure.json",{"retryable":True,"error":"interrupted system call"})
                def poll(self):return 1
                def wait(self,timeout=None):return 1
            predictor=FilePredictor(config,identity,root/"exports",pointer)
            with patch("reliability_v1.exports.subprocess.Popen",TransientPopen):
                with self.assertRaises(RuntimeError):
                    predictor({"pair_id":"transient","scene":"val"},"A0")
                self.assertEqual(len(launches),2)
                with self.assertRaises(RuntimeError):
                    predictor({"pair_id":"transient","scene":"val"},"A0")
                self.assertEqual(len(launches),2)

    def test_timing_dispatch_does_not_time_cached_predictions(self):
        class Timed:
            def __call__(self,*args):raise AssertionError("Cached prediction must not be timed")
            def measure_costs(self,records,out):
                return {"count":len(records),"delegated":True}
        records=[{"pair_id":str(i),"scene":"s"} for i in range(12)]
        result=measure_costs(records,Timed(),"/unused",synchronize=None)
        self.assertEqual(result,{"count":10,"delegated":True})

    def test_interrupt_terminates_worker_group(self):
        import os
        import signal
        from unittest.mock import Mock
        for cause in ("keyboard","sigterm"):
            with self.subTest(cause=cause),tempfile.TemporaryDirectory() as tmp:
                root,config,identity,pointer=self.setup_files(tmp);created=[]
                class RunningPopen:
                    pid=987654321
                    def __init__(self,command,**kwargs):
                        self.code=None;created.append(kwargs)
                    def poll(self):return self.code
                    def wait(self,timeout=None):
                        self.code=-15;return self.code
                predictor=FilePredictor(config,identity,root/"exports",pointer)
                def interrupt(_):
                    if cause=="sigterm":signal.getsignal(signal.SIGTERM)(signal.SIGTERM,None)
                    raise KeyboardInterrupt("test interrupt")
                original_handler=signal.getsignal(signal.SIGTERM)
                with patch("reliability_v1.exports.subprocess.Popen",RunningPopen),patch("reliability_v1.exports.time.sleep",side_effect=interrupt),patch("reliability_v1.exports.os.killpg") as kill:
                    with self.assertRaises(KeyboardInterrupt):
                        predictor({"pair_id":"interrupt","scene":"val"},"A0")
                    kill.assert_called_once_with(987654321,signal.SIGTERM)
                self.assertTrue(created[0]["start_new_session"])
                self.assertEqual(created[0]["env"]["SLIM_EXPORT_PARENT_PID"],str(os.getpid()))
                self.assertEqual(signal.getsignal(signal.SIGTERM),original_handler)
                interruptions=list((root/"exports").glob("pairs/*/attempt_0/interruptions/*.json"))
                self.assertEqual(len(interruptions),1)
                self.assertFalse(json.loads(interruptions[0].read_text())["automatic_retry"])

    def test_parent_death_race_checked_before_gpu_load(self):
        import signal
        from types import SimpleNamespace
        from unittest.mock import Mock
        from reliability_v1.export_worker import arm_parent_death_signal
        library=SimpleNamespace(prctl=Mock(return_value=0))
        with patch.dict("os.environ",{"SLIM_EXPORT_PARENT_PID":"100"}),patch("os.getppid",side_effect=[100,101]),patch("ctypes.CDLL",return_value=library):
            with self.assertRaisesRegex(RuntimeError,"arming"):
                arm_parent_death_signal()
        library.prctl.assert_called_once_with(1,int(signal.SIGTERM),0,0,0)



if __name__=="__main__":unittest.main()
