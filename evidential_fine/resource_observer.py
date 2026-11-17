"""Read-only resource observer; watchdog enforcement remains in each stage supervisor."""
import os,sys,time,json,subprocess,datetime,shutil
from pathlib import Path
sys.dont_write_bytecode=True
run=Path(sys.argv[1]);out=run/'observation';out.mkdir(exist_ok=True)
def size(folder):
    total=0
    for root,dirs,files in os.walk(folder):
        for f in files:
            try:total+=(Path(root)/f).stat().st_size
            except FileNotFoundError:pass
    return total
def read(p):
    try:return json.loads(p.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}
context=read(run/'context.json');T=Path(context['T_run']);peaks={}
(out/'observer.pid').write_text(str(os.getpid())+'\n')
while True:
    mem={k:int(v.strip().split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())}
    sample=dict(time_beijing=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),unix=time.time(),pid=os.getpid(),run_apparent_bytes=size(run),T_apparent_bytes=size(T),
        host_c_free_bytes=shutil.disk_usage('/mnt/c').free,linux_free_bytes=shutil.disk_usage(run).free,mem_available=mem['MemAvailable'],swap_used=mem['SwapTotal']-mem['SwapFree'],
        stage=read(run/'stage_status.json'))
    peaks['P5_observed_peak_bytes']=max(peaks.get('P5_observed_peak_bytes',0),sample['run_apparent_bytes'])
    peaks['T_observed_peak_bytes']=max(peaks.get('T_observed_peak_bytes',0),sample['T_apparent_bytes'])
    peaks['host_minimum_free_bytes']=min(peaks.get('host_minimum_free_bytes',sample['host_c_free_bytes']),sample['host_c_free_bytes'])
    with (out/'resources.jsonl').open('a') as f:f.write(json.dumps(sample)+'\n');f.flush()
    p=out/'observed_peaks.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(**peaks,sample_interval_seconds=30,scope='Observed snapshots; sub-interval checkpoint atomic temporary peak is separately bounded',latest=sample),indent=2));os.replace(tmp,p)
    if (out/'stop').exists() or sample['stage'].get('status')=='completed' and sample['stage'].get('stage')=='report':break
    time.sleep(30)
(out/'exit.json').write_text(json.dumps(dict(status='completed',time_beijing=datetime.datetime.now().astimezone().isoformat(),**peaks),indent=2))
