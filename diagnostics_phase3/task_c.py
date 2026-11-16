"""C GPU extraction followed by CPU-only analysis; no parameter update."""
from common import *
import gc,subprocess
from c_extract import extract,optional_two_step
from c_statistics import analyze
def main(run):
    import torch
    run=Path(run);out=run/'C';recs=records('epe');runtime=None
    if (run/'B/status.json').exists() and read(run/'B/status.json').get('status')=='running':raise RuntimeError('B is still running; GPU tasks must be serial')
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
    if gpu and gpu!='No running processes found':raise RuntimeError('GPU is already used: '+gpu)
    try:
        runtime,stats,fit=start_runtime(run,'C')
        for index,r in enumerate(recs):
            receipt=committed(run,'C',r)
            if receipt is None:
                a,result,ref=extract(runtime,stats,fit,r,run)
                receipt=commit(run,'C',r,result,a,ref);del a,result;gc.collect()
            heartbeat(run,'C',phase='full forward / C0',committed_pairs=index+1,total_pairs=100)
        optional_path=out/'optional_two_step_equivalence.json'
        if not optional_path.exists():
            optional_two_step(runtime,stats,fit,run,recs)
        check=check_runtime(runtime);save(out/'frozen_check.json',check)
    finally:
        if runtime is not None:
            try:save(out/'frozen_check.json',check_runtime(runtime))
            finally:del runtime;gc.collect();torch.cuda.empty_cache()
    return analyze(run,recs)
