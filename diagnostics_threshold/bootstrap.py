"""Create preregistered T and Phase 5 roots; no experiment outcomes are read here."""
from shared import *
from reliability_v1.environment_monitor import FAULT,NEW_WARNING,system_sample
def main():
    ident=identity_gate();stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%dT%H%M%S')
    run=ROOT/'runs'/('phaseT_'+stamp);run.mkdir(exist_ok=False)
    p5id=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%dT%H%M%S')
    p5=ROOT/'runs'/('phase5_'+p5id);p5.mkdir(exist_ok=False)
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    initial=dict(status='pass',boot_id=boot,time_beijing=now(),resources=system_sample(),linux_free_bytes=shutil.disk_usage(run).free)
    faults=[];warnings=[];n=0;cursor=None
    proc=subprocess.Popen(['journalctl','-k','-b','--no-pager','-o','json'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    with (run/'kernel_boot_review.jsonl').open('w') as f:
        for line in proc.stdout:
            x=json.loads(line);n+=1;cursor=x['__CURSOR'];assert x['_BOOT_ID']==boot.replace('-','')
            f.write(line)
            if FAULT.search(x.get('MESSAGE','')):faults.append(x)
            if NEW_WARNING.search(x.get('MESSAGE','')):warnings.append(x)
    proc.wait();assert proc.returncode==0
    initial.update(cursor=cursor,kernel_records=n,current_boot_faults=faults,current_boot_warnings=warnings)
    save(run/'initial_environment.json',initial);save(p5/'initial_environment.json',initial)
    if faults:raise IdentityFailure('Current boot storage faults: globally stop')
    validation=records();final=records('epe');audit=read(SOURCE/'data_audit_summary.json')
    tr=set(audit['train_scenes']);va=set(audit['val_scenes']);excluded=set(audit['excluded_test_scenes'])|{'0015','0022'}
    assert len(tr)==138 and len(va)==15 and not tr&va and not (tr|va)&excluded
    assert len({r['scene'] for r in final})==15
    assert {r['pair_id'] for r in final}<={r['pair_id'] for r in validation}
    specimen=final[0];p=Path(specimen['image0'])
    with p.open('rb') as f:block=f.read(256*1024)
    for folder in [run,p5]:
        q=folder/'bounded_probe.tmp';data=os.urandom(1024*1024)
        with q.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        assert q.read_bytes()==data;q.unlink();fd=os.open(folder,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
    # Estimate from actual compressed historical pair arrays, plus wide-candidate and training log/checkpoint bounds.
    cbytes=sum(p.stat().st_size for p in (P3/'C/pairs').glob('*.npz'))
    bbytes=sum(p.stat().st_size for p in (P3/'B/pairs').glob('*.npz'))
    estimated=max(8*2**30,cbytes*30+bbytes*4)
    storage=dict(phase3_C_payload_bytes=cbytes,phase3_B_payload_bytes=bbytes,estimated_added_peak_bytes=estimated,
        reserve_bytes=2*2**30,basis='30 x actual 100-pair trajectory payload + 4 x 750-pair coarse payload; minimum 8 GiB includes candidate grids, training logs, selftest features, two small heads/Adam states and atomic temporary copies',
        linux_free_bytes=shutil.disk_usage(run).free,host_c_free_bytes=shutil.disk_usage('/mnt/c').free,
        backing_path=r'C:\Users\bupt\AppData\Local\wsl\{2ffa190c-d4f2-4f32-98e4-e3c6b0721fed}\ext4.vhdx',
        swap_path=r'C:\Users\bupt\AppData\Local\Temp\017C467D-EB94-47CD-A037-D966E1A773EA\swap.vhdx',host_volume_counted_once=True,time_beijing=now())
    assert min(storage['linux_free_bytes'],storage['host_c_free_bytes'])>=estimated+2*2**30
    settings=[dict(setting=s,threshold=t,texture_budget=s=='S5',fraction=.2 if s=='S5' else 0) for s,t in zip(['S0','S1','S2','S3','S4','S5'],[.65,.55,.45,.30,None,.65])]
    protocol=dict(identity=EXPECTED,created=now(),T_settings=settings,T_order=[x['setting'] for x in settings],T_timing=dict(warmup=1,repeats=3,statistic='per-pair median; scene-equal ratios',synchronization='CUDA before/after',exclude='GT/statistics/file writes'),
        final_pairs_sha=sha(SOURCE/'fixed_final_epe.json'),validation_sha=sha(SOURCE/'fixed_validation.json'),
        plans={f'epoch_{i:02d}':sha(SOURCE/'plans'/f'epoch_{i:02d}.json') for i in range(2)},
        GT1=dict(comparison='S2-S0 low-texture source-grid any-parent usable EPE<2 recall',positive_lower_gt=.05,small_upper_lt=.02),
        GT2=dict(positive_ge=.60,negative_le=.40,denominator='S2 newly added low parents with valid median final EPE'),
        GT3=dict(equal_N0_N1_each_pair_required=True,recall_ci_lower_gt=0,time_ratio_le=1.1),
        Phase5=dict(order=['E0','A','B','final'],epochs=2,visits_per_epoch=13800,validation_per_epoch=750,final_pairs=100,
            formal_seeds={'A':2026,'B':2026},selftest_seed=2027,selftest_pairs='first 8 distinct pairs in frozen epoch 0 order',selftest_updates=200,
            A=dict(R=8,K=32,cap=1024,shared_sides=True,coarse_project=128,fine_quadrant_project=128,side_layers=[128,128],fusion_layers=[128,128],axis_hidden=64,normalization='LayerNorm',activation='GELU',window='native 8x8 fine cells; 4x4 quadrant means TL,TR,BL,BR'),
            B=dict(R=3,K=12,cap=2048,conv_channels=[64,64],kernel=3,padding=1,normalization='GroupNorm 8 groups',activation='GELU',linear=128,axis_hidden=64,window='native bilinear 6x6 fine cells at t0'),
            optimizer=dict(name='AdamW',lr=.001,weight_decay=.0001,betas=[.9,.999],eps=1e-8,scheduler=None),head_precision='AMP linear/conv; all decoding and NIG float32',scaler='torch.cuda.amp.GradScaler; restored in checkpoint',
            stable_floors=dict(nu=1e-6,beta=1e-6,alpha_minus_one=1e-4),
            loss='specified normalized NIG NLL + abs(u-y)*(2nu+alpha), equal weight; samples/axes mean',
            supervision='GT valid AND Linf offset strictly <R; all valid GT for evaluation',
            selftest_criteria='last20 loss <first20; mean EPE at update200 <zero-offset on frozen point set; all finite',
            official_reset='destroy selftest head, optimizer, scaler, RNG; seed 2026 and cursor0',
            checkpoint='head+AdamW+scaler+all RNG+cursor; atomic pointer; epoch2 fixed; no best selection',
            G8='B-t1 lt1: lower>0 superior; else lower>=-.01 noninferior; upper<-.01 worse; otherwise insufficient',
            G9='rho(-epistemic,EPE)-rho(fine,EPE): upper<0 better; lower>0 worse; otherwise insufficient',
            G10='A-native parent median lt2 lower>=-.02 AND actual fine-stage latency lower'),
        aggregation='per pair -> mean within scene -> scene equal; paired common units; 2000 whole-scene bootstrap seed2026',
        ties='frozen metrics unchanged; concrete selection ties by stable parent ID then original child ordinal',
        degradation_order=['T S3/S4 pose','T S1','Phase5 pose','T segmented timing','A efficiency timing'],
        required_preserved=['identity/resources','T S0/S2/S5 GT1-GT3','P5 E0 B training G8/G9','A training G10 precision'],
        geometry=dict(backend='canonical LO-RANSAC',pixel_threshold=.5,confidence=.99999,seeds=list(range(5)),workers=3),
        inference_GT_use='none; only supervision/offline evaluation/oracle',no_A4=True,no_MegaDepth1500=True)
    for folder,kind,rid in [(run,'phaseT',stamp),(p5,'phase5',p5id)]:
        output=ROOT/'experiment_results'/(kind+'_'+rid);output.mkdir(exist_ok=False)
        context=dict(run=str(folder),run_id=rid,results=str(output),T_run=str(run),phase5_run=str(p5),created=now(),identity=EXPECTED)
        save(folder/'context.json',context);save(folder/'frozen_identity.json',ident);save(folder/'preregistered_protocol.json',protocol);save(folder/'storage_budget.json',storage)
        save(folder/'startup.json',dict(time_beijing=now(),boot_id=boot,branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
            HEAD=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),worktree=subprocess.check_output(['git','status','--porcelain=v1'],text=True),
            input_probe=dict(path=str(p),bytes=len(block),sha256=hashlib.sha256(block).hexdigest()),fsync_readback='pass',split='pass',excluded=list(sorted(excluded))))
        save(folder/'stage_status.json',dict(status='prepared',identity=EXPECTED))
        save(output/'preregistered_protocol.json',protocol)
    print(json.dumps(dict(T_run=str(run),phase5_run=str(p5),storage=storage)),flush=True)
if __name__=='__main__':main()
