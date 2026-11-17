"""Real fine-stage replacements; all actual outputs, no GT work in timed regions."""
from .common import *
from .native import prefix,inputs
import torch
def path(rt,heads,common,method):
    d=dict(common);model=rt.model
    with torch.no_grad():
        if method in ['native1','native4','B']:
            with torch.cuda.amp.autocast(enabled=rt.config['amp']):
                model._fine_correlation(d);model._get_fine_coord_test(d)
                d['fine_parent_id']=d['coarse_match_id'][d['m_idx_it']]
                if method!='B':
                    before=model.refine_iters;model.refine_iters=1 if method=='native1' else 4
                    try:model._refinement(d)
                    finally:model.refine_iters=before
                    return d['fine_coord_0'],d['fine_coord_1'],d['fine_parent_id']
        route=method;features=inputs(rt,d,route)
        with torch.cuda.amp.autocast(enabled=False):
            # Preserve the existing full-evaluation4096 head batches on GPU.
            # Every actual output remains included; no training cap is applied.
            chunks=[heads[route](tuple(x[i:i+4096] for x in features))['mu']
                    for i in range(0,len(features[0]),4096)]
            mu=torch.cat(chunks) if chunks else features[0].new_empty((0,2))
            p={'mu':mu}
        source=d['coarse_coord_0']+4 if route=='A' else d['intermediate_coord_0']
        target=(d['coarse_coord_1']+4 if route=='A' else d['intermediate_coord_1'])+p['mu']
        return source,target,d['coarse_match_id'] if route=='A' else d['fine_parent_id']
def proof(rt,heads,d,method):
    count={'fine_correlation':0,'RRU':0};old=rt.model._fine_correlation
    def fine(data):count['fine_correlation']+=1;return old(data)
    def rru(*args):count['RRU']+=1
    rt.model._fine_correlation=fine;hook=rt.model.recurrent_refine_unit.register_forward_hook(rru)
    try:out=path(rt,heads,d,method)
    finally:hook.remove();rt.model._fine_correlation=old
    expected={'native1':(1,1),'native4':(1,4),'A':(0,0),'B':(1,0)}[method]
    assert (count['fine_correlation'],count['RRU'])==expected,(method,count)
    return count
def measure(run,rt,heads):
    run=Path(run);out=run/'efficiency';out.mkdir(exist_ok=True)
    skipped=read(run/'degradation.json').get('skipped',[]) if (run/'degradation.json').exists() else []
    methods=['native1','native4']+[r for r in ['A','B'] if r in heads and not (r=='A' and 'A efficiency timing' in skipped)]
    rows=[]
    for index,r in enumerate(records('epe')):
        saved=get_unit(run,'efficiency',r)
        if saved is not None:rows.append(saved);continue
        beat(run,'evaluate',phase='efficiency',pair_index=index,pair_id=r['pair_id'])
        loaded=rt.load(r);up=[]
        d=dict(loaded);prefix(rt,d,'A');del d
        for rep in range(3):
            d=dict(loaded);torch.cuda.synchronize();t=time.perf_counter();prefix(rt,d,'A');torch.cuda.synchronize();up.append(time.perf_counter()-t)
            if rep!=2:del d
        checks={};timing={};maxdiff={}
        for method in methods:
            checks[method]=proof(rt,heads,d,method);result=path(rt,heads,d,method);del result # warmup
            samples=[]
            for rep in range(3):
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();initial=torch.cuda.memory_allocated();t=time.perf_counter()
                result=path(rt,heads,d,method);torch.cuda.synchronize();elapsed=time.perf_counter()-t
                samples.append(dict(seconds=elapsed,base_allocated=initial,allocated_peak=torch.cuda.max_memory_allocated(),reserved_peak=torch.cuda.max_memory_reserved(),outputs=len(result[0])))
                if rep==2:
                    src,trg,cp=[x.float().cpu().numpy() if x.dtype.is_floating_point else x.cpu().numpy() for x in result]
                    if method in heads:
                        ref=get_unit(run,'final_'+method,r,True)
                        assert np.array_equal(cp,ref['cp']) and np.array_equal(src,ref['source'])
                        delta=float(np.max(np.abs(trg-ref['target']),initial=0))
                        maxdiff[method]=delta
                        if delta>1e-5:raise RuntimeError('Full evaluation versus replacement coordinate mismatch')
                    elif method in ['native1','native4']:
                        reference_route='B' if 'B' in heads else 'A'
                        ref=get_unit(run,'final_'+reference_route,r,True)
                        assert np.array_equal(cp,ref['native_cp']) and np.array_equal(src,ref['native_source'])
                        expected=ref['native_one_target'] if method=='native1' else ref['native_target']
                        delta=float(np.max(np.abs(trg-expected),initial=0));maxdiff[method]=delta
                        if delta>1e-5:raise RuntimeError('Timed native reference differs from direct full-evaluation reference')
                del result
            timing[method]=dict(repeats=samples,median_seconds=float(np.median([x['seconds'] for x in samples])))
        row=dict(**base(r),timing=timing,upstream=dict(repeats=up,median_seconds=float(np.median(up))),execution_proof=checks,
            max_output_deviation=maxdiff,boundary='Start: post-interaction coarse descriptors, matched coarse parents and upsampled fine maps. End: actual output source/target coordinates; includes mandatory extraction/pooling/correlation/head/decode. Shared upstream excluded and separately measured.',resources=resource_sample())
        commit(run,'efficiency',r,row,dict(pair_id=np.array(r['pair_id'])));rows.append(row);del d,loaded;gc.collect()
    rs=[dict(**base(r),methods={m:{'seconds':r['timing'][m]['median_seconds']} for m in methods}) for r in rows]
    comp=[(m,'native4') for m in methods if m!='native4']+([('B','native1')] if 'B' in methods else [])
    summary=scene_summary(rs,['seconds'],methods,comp)
    save(out/'summary.json',dict(status='completed',pairs=len(rows),summary=summary,skipped=[x for x in ['A'] if x not in methods],frozen=rt.check_frozen()))
    return read(out/'summary.json')
