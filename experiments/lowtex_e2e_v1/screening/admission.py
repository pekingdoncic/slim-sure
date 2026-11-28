import argparse
import subprocess
import sys
from .data import EXP,OUT,read,save,sha

def main():
    p=argparse.ArgumentParser();p.add_argument("--exp",required=True,choices=("exp0","exp1a","rru_matched","single","crop","joint"));a=p.parse_args()
    cfg=read(OUT/"training_profile.json")
    if cfg!=read(OUT/"selected_config.json"):raise ValueError("Shared throughput profile changed")
    if cfg["bs"]==2 and not read(OUT/"batch_equivalence.json")["pass"]:raise ValueError("Missing true B2 verification")
    d=EXP/"outputs/fast_screening_v1/smoke"/cfg["id"]/a.exp
    done=read(d/"complete.json")
    if done["step"]!=20 or done["visits"]!=80 or done["min_free_bytes"]<1024**3:raise ValueError("Smoke incomplete or memory margin failed")
    if done["checkpoint_sha256"]!=sha(d/"checkpoint_last.pt"):raise ValueError("Smoke checkpoint digest changed")
    arg=str(d.relative_to(EXP))
    if not (d/"timing.json").exists():
        subprocess.run([sys.executable,"-B","-m","experiments.lowtex_e2e_v1.screening.timing","--smoke","--run",arg],check=True)
    if not (d/"actual_gradients.json").exists():
        subprocess.run([sys.executable,"-B","-m","experiments.lowtex_e2e_v1.screening.gradient_audit","--run",arg],check=True)
    assert {(r["side"],r["route"]) for r in read(d/"timing.json")["results"]}>={(640,"native"),(832,"native")}
    for step in (0,20):
        for side in (640,832):
            v=read(d/"validation"/f"step_{step:08d}"/"native"/f"r{side}"/"accepted.json")
            assert v["scope"]=="smoke" and set(v["scores"])=={"texturepoor","megadepth"}
    save(OUT/"admissions"/(a.exp+".json"),dict(profile=cfg["id"],checkpoint=done["checkpoint_sha256"],
         gradient_audit=read(d/"actual_gradients.json"),stage="screening",experiment=a.exp))
if __name__=="__main__":main()
