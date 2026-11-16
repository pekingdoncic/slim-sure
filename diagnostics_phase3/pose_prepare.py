"""A4 prediction preparation: all fine-output parents, no GT-based filtering."""
from common import *
from reliability_v1.geometry_worker import ARRAY_KEYS,array_sha,prediction_identity
from reliability_v1.evaluation import canonical_protocol
RATES=(1.,.9,.8,.7,.5)
FILTERS=('head','lr','mconf','random')
COMBOS=[('baseline',1.)]+[(m,f) for f in [.9,.8,.7,.5] for m in FILTERS]
def combo_name(method,rate):return 'baseline_100' if method=='baseline' else method+'_'+str(int(round(rate*100)))
def occupancy(points,hw,grid=32):
    h,w=map(int,hw);p=np.asarray(points)
    inside=(p[:,0]>=0)&(p[:,1]>=0)&(p[:,0]<w)&(p[:,1]<h)
    cells=np.floor(p[inside]/grid).astype(np.int64);n=len(np.unique(cells,axis=0));den=int(np.ceil(w/grid)*np.ceil(h/grid))
    return dict(occupied_cells=n,valid_grid_cells=den,occupancy=n/den,points_inside=int(inside.sum()),points_outside=int((~inside).sum()),grid_pixels=grid,valid_hw=[h,w])
def prepare(run,recs):
    from PIL import Image
    run=Path(run);out=run/'A4';protocol=canonical_protocol();save(out/'protocol.json',protocol)
    sys.path.insert(0,str(Path(protocol['hybrid_root']).parent))
    from hybrid_match.evaluation.metrics import symmetric_epipolar_distance
    preparations=[];units={};unit_order=[]
    for i,r in enumerate(recs):
        name=pairname(r['pair_id']);folder=out/'predictions'/name;folder.mkdir(parents=True,exist_ok=True);commitfile=folder/'commit.json'
        if commitfile.exists():
            receipt=read(commitfile)
            if receipt['pair_id']!=r['pair_id'] or receipt['protocol_sha256']!=digest(protocol):raise GateFailure('A4 prediction identity changed')
            for v in receipt['predictions'].values():
                if sha(folder/v['payload'])!=v['sha256']:raise GateFailure('A4 committed prediction checksum changed')
        else:
            p,ref=cache(r);raw=p['raw_id'];cp=p['child_parent_id'];outputids=np.unique(cp);positions_out=positions(raw,outputids)
            low=p['low'].astype(bool);parent_low=low[positions_out];count=len(outputids)
            ordered={}
            for method in METHODS:ordered[method]=outputids[np.lexsort((outputids,-p[method][positions_out]))]
            random_seed=int(digest(dict(seed=2026,pair_id=r['pair_id'],purpose='fixed_parent_random_prefix'))[:8],16)
            ordered['random']=np.random.default_rng(random_seed).permutation(outputids)
            source=p['final_source_input']*p['input_to_orig0'];target=p['final_target_input']*p['input_to_orig1']
            K0=np.asarray(r['K0']);K1=np.asarray(r['K1']);T=np.asarray(r['pose1'])@np.linalg.inv(np.asarray(r['pose0']))
            full=dict(points0_orig=source,points1_orig=target,K0=K0,K1=K1,T_0to1=T)
            hashes={}
            for method in FILTERS:
                selected=ordered[method][:int(np.ceil(count))];childmask=np.isin(cp,selected)
                hashes[method]={k:array_sha(v[childmask] if k in ('points0_orig','points1_orig') else v) for k,v in full.items()}
            if any(h!=hashes['head'] for h in hashes.values()):raise GateFailure('100% ordered coordinate baseline differs by method')
            epipolar=symmetric_epipolar_distance(source,target,K0,K1,T)
            shapes=[]
            for side in [0,1]:
                image=Path(r['image'+str(side)])
                if sha(image)!=r['file_sha256']['image'+str(side)]:raise GateFailure('Image changed before coverage header read')
                with Image.open(image) as im:w,h=im.size
                shapes.append((h,w))
            input_hw=[(int(round(shapes[s][0]/p['input_to_orig'+str(s)][1])),int(round(shapes[s][1]/p['input_to_orig'+str(s)][0]))) for s in [0,1]]
            predrefs={};metrics={}
            for method,rate in COMBOS:
                combo=combo_name(method,rate)
                keep=outputids if method=='baseline' else ordered[method][:int(np.ceil(rate*count))]
                childmask=np.isin(cp,keep) # preserves original native child order, independent of parent ranking.
                points={k:(v[childmask] if k in ('points0_orig','points1_orig') else v) for k,v in full.items()}
                for k in ['points0_orig','points1_orig']:finite(points[k],k)
                path=folder/(combo+'.npz');npzout(path,dict(**points,pair_id=np.array(r['pair_id'])))
                predrefs[combo]=dict(payload=path.name,sha256=sha(path),identity=prediction_identity(points,protocol))
                kept_low=int(np.isin(keep,outputids[parent_low]).sum())
                n=int(childmask.sum());good=int((epipolar[childmask]<1e-4).sum())
                metrics[combo]=dict(method=method,rate=rate,original_output_parents=count,retained_parents=len(keep),children=n,
                    epipolar_correct=good,epipolar_correct_fraction=good/n if n else None,nonfinite_epipolar_distances=int((~np.isfinite(epipolar[childmask])).sum()),
                    low_texture_before=int(parent_low.sum()),low_texture_retained=kept_low,low_texture_retention=kept_low/int(parent_low.sum()) if parent_low.any() else None,
                    coverage_input={str(side):occupancy(p['final_source_input' if side==0 else 'final_target_input'][childmask],input_hw[side]) for side in [0,1]},
                    coverage_original={str(side):occupancy(points['points0_orig' if side==0 else 'points1_orig'],shapes[side]) for side in [0,1]})
            receipt=dict(status='committed',**base(r),source=ref,predictions=predrefs,metrics=metrics,protocol_sha256=digest(protocol),
                baseline_method_coordinate_hashes=hashes,random_seed=random_seed,random_parent_permutation=ordered['random'].tolist(),
                valid_input_hw=input_hw,valid_original_hw=shapes,geometry_filter_population='All raw parents with any fine output, no GT/label/EPE selection')
            save(commitfile,receipt)
        preparations.append(receipt)
        for method,rate in COMBOS:
            combo=combo_name(method,rate);pred=receipt['predictions'][combo]
            for seed in range(5):
                key=name+'/'+combo+'/'+str(seed);unit_order.append(key)
                units[key]=dict(**base(r),method=method,rate=rate,combo=combo,seed=seed,prediction=str(folder/pred['payload']),
                    prediction_sha256=pred['sha256'],directory=name+'/'+combo,independent_repeat=False)
        heartbeat(run,'A4',phase='prepare fixed filtered geometry inputs',prepared_pairs=i+1,total_pairs=100)
    first_by_scene={s:sorted([r for r in recs if r['scene']==s],key=lambda r:r['pair_id'])[0] for s in sorted({r['scene'] for r in recs})}
    scenes=sorted(first_by_scene);timing=[]
    for k in range(20):
        r=first_by_scene[scenes[k%len(scenes)]];method,rate=COMBOS[k%len(COMBOS)];combo=combo_name(method,rate)
        key=pairname(r['pair_id'])+'/'+combo+'/'+str(k%5)
        if key in timing:raise AssertionError('Duplicate timing unit')
        timing.append(key)
    repeat=[]
    for scene in scenes[:5]:
        r=first_by_scene[scene];name=pairname(r['pair_id'])
        for seed in range(5):
            source=name+'/baseline_100/'+str(seed);key='repeat/'+source;unit=dict(units[source])
            unit.update(directory='repeats/'+name+'/baseline_100',independent_repeat=True,baseline_unit=source)
            units[key]=unit;repeat.append(key)
    if len(unit_order)!=8500 or len(timing)!=20 or len(repeat)!=25:raise AssertionError('Geometry plan size differs')
    plan=dict(status='frozen',output=str(out/'geometry'),protocol=str(out/'protocol.json'),protocol_sha256=digest(protocol),units=units,
        formal_unit_order=unit_order,initial_timing_units=timing,independent_repeat_units=repeat,formal_units=8500,independent_units=25,
        seeds=list(range(5)),primary_retention=.8,first20_selection='Fixed before solves: scene round robin, combination round robin, seed k mod 5; all count toward formal results',
        random_control='One pair-specific seeded parent permutation; all rates use prefixes; child order restored before canonical seed permutation',
        geometry_adapter='pose_units.py reuses geometry_worker identity/protocol and the same frozen hybrid estimate_pose/pose_error; one-unit scheduling only')
    path=out/'geometry_plan.json'
    if path.exists():
        if read(path)!=plan:raise GateFailure('Existing geometry plan changed')
    else:save(path,plan)
    save(out/'prediction_preparation.json',dict(status='completed',pairs=100,rows=preparations,baseline_hash_equality=True))
    return plan
