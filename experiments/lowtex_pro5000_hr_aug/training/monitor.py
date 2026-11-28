import json
import subprocess
import time
from .options import parser
from ..utils.run_io import locations,read


def monitor(args):
    _,report = locations(args.mode,args.run_id)
    while True:
        print(time.strftime("%Y-%m-%d %H:%M:%S"),report,flush=True)
        subprocess.run(["nvidia-smi","--query-gpu=index,uuid,name,memory.free,memory.used,utilization.gpu","--format=csv"])
        for name in ("progress","live_status"):
            path = report/(name+".json")
            if path.exists():
                data = read(path)
                print(name,json.dumps(data if name=="live_status" else {k:data.get(k) for k in ("stage","status","updated")},ensure_ascii=False))
        log = report/"pipeline.log"
        if log.exists():
            with log.open("rb") as f:
                f.seek(max(0,log.stat().st_size-8192))
                print("\n".join(f.read().decode(errors="replace").splitlines()[-5:]),flush=True)
        state=read(report/"progress.json") if (report/"progress.json").exists() else {}
        if state.get("status") in {"PASS","PARTIAL","FAIL"}:
            break
        time.sleep(5)
