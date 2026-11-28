import fcntl, json, os, subprocess, time, traceback
from .data import EXP,OUT,prepare,read,save
from .bench_matrix import matrix

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/"workflow.lock").open("a")
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:print("Fast screening workflow already active",flush=True);return
    if not (OUT/"legacy_r832/transition.json").is_file():raise RuntimeError("R832 transition has not completed")
    prepare()
    if not (OUT/"selected_config.json").exists():matrix()
    # Launch a fresh process per run so validation, memory high water and failures are isolated.
    from .pipeline import execute
    execute()

if __name__=="__main__":
    try:main()
    except Exception as exc:
        save(OUT/"state.json",dict(stage="workflow",status="BLOCKED",error=repr(exc),traceback=traceback.format_exc(),pid=os.getpid()))
        raise
