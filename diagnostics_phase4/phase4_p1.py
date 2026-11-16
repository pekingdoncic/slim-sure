from common4 import *
BINS=['[0,4]','(4,8]','(8,12]','>12']
COUNT_KEYS=['parents','without_fine','fine_without_valid_GT','valid_final_parents','final_lt1','final_lt2','initial_lt2','children_valid','initial_child_ge2','rescued','initial_child_lt2','lost']
def ratios(x,total):
    return dict(share=ratio(x['parents'],total),final_lt1_fraction=ratio(x['final_lt1'],x['valid_final_parents']),final_lt2_fraction=ratio(x['final_lt2'],x['valid_final_parents']),
        initial_lt2_fraction=ratio(x['initial_lt2'],x['valid_final_parents']),rescue_fraction=ratio(x['rescued'],x['initial_child_ge2']),loss_fraction=ratio(x['lost'],x['initial_child_lt2']))
def main(run):
    cap=read(Path(run)/'P0/summary.json')['capabilities']
    if not cap['P1']:
        obj=dict(status='不可判定',reason='P0 coarse residual / child association unavailable');artifact(run,'P1_residual_bins',obj);return obj
    rows=[]
    for i,r in enumerate(records('epe')):
        v=own_read(run,'P1',r)
        if v is None:
            a=data(run,r);raw=a['raw_id'];cp=a['cp'];valid=a['valid'];last=a['EPE'][-1];initial=a['EPE'][0]
            output=np.isin(raw,cp);ev=np.zeros(len(raw),bool);e0=np.full(len(raw),np.nan);et=e0.copy();stats=np.zeros((len(raw),5),int)
            order=np.argsort(cp,kind='stable');groups=np.split(order,np.flatnonzero(np.diff(cp[order]))+1);lookup={int(p):j for j,p in enumerate(raw)}
            for inds in groups:
                j=lookup[int(cp[inds[0]])];g=inds[valid[inds]]
                if not len(g):continue
                ev[j]=True;e0[j]=np.median(initial[g]);et[j]=np.median(last[g])
                stats[j]=[len(g),int((initial[g]>=2).sum()),int(((initial[g]>=2)&(last[g]<2)).sum()),int((initial[g]<2).sum()),int(((initial[g]<2)&(last[g]>=2)).sum())]
            part=[]
            anchors=[('corner',a['corner_residual'],a['labels']>=0)]
            if 'center_residual' in a:anchors.append(('center',a['center_residual']-4,a['center_labels']>=0))
            for anchor,residual,resvalid in anchors:
                finite(residual[resvalid],anchor+' valid residual')
                b=np.searchsorted([4,8,12],np.max(np.abs(residual),axis=1),side='left')
                for region,regionmask in [('all',np.ones(len(raw),bool)),('low_texture',a['parent_low'])]:
                    total=int((resvalid&regionmask).sum())
                    for index,name in enumerate(BINS):
                        mask=resvalid&regionmask&(b==index);good=mask&ev;s=stats[mask].sum(axis=0)
                        x=dict(parents=int(mask.sum()),without_fine=int((mask&~output).sum()),fine_without_valid_GT=int((mask&output&~ev).sum()),
                            valid_final_parents=int(good.sum()),final_lt1=int((good&(et<1)).sum()),final_lt2=int((good&(et<2)).sum()),initial_lt2=int((good&(e0<2)).sum()),
                            **dict(zip(['children_valid','initial_child_ge2','rescued','initial_child_lt2','lost'],s.tolist())))
                        if x['parents']!=x['without_fine']+x['fine_without_valid_GT']+x['valid_final_parents']:raise AssertionError('P1 denominator conservation')
                        part.append(dict(**base(r),anchor=anchor,region=region,bin=name,coarse_valid_total=total,**x,**ratios(x,total)))
            v=dict(rows=part);own_commit(run,'P1',r,v)
        rows.extend(v['rows']);heartbeat(run,'P1',committed_pairs=i+1,total_pairs=100)
    out=[];sens={}
    for anchor in sorted({r['anchor'] for r in rows}):
        for region in ['all','low_texture']:
            total=sum(r['coarse_valid_total'] for r in rows if r['anchor']==anchor and r['region']==region and r['bin']==BINS[0])
            for b in BINS:
                rs=[r for r in rows if r['anchor']==anchor and r['region']==region and r['bin']==b]
                x={k:sum(r[k] for r in rs) for k in COUNT_KEYS}
                out.append(dict(anchor=anchor,region=region,bin=b,coarse_valid_total=total,**x,**ratios(x,total)))
                sens[anchor+'/'+region+'/'+b]=scene_avg(rs,list(ratios(x,total)))
    summary=dict(status='completed',pooled=out,scene_equal_sensitivity=sens,center_status='完成' if any(r['anchor']=='center' for r in rows) else '不可判定',
        interpretation='Success-rate association only; does not prove geometric overlap of the two windows.')
    artifact(run,'P1_residual_bins',summary,out);save(Path(run)/'P1/pair_rows.json',rows);save(Path(run)/'P1/summary.json',summary)
    return dict(status='completed',pairs=100,rows=len(out),center_status=summary['center_status'])
