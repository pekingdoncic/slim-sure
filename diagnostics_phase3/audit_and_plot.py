"""Independent invariants for the derived diagnostic tables; optional standard plots."""
from common import *
import ast
def main(run):
    run=Path(run);checks=[]
    modules=list((ROOT/'diagnostics_phase3').glob('*.py'))
    for p in modules:ast.parse(p.read_text())
    checks.append(dict(check='all diagnostic modules parse',count=len(modules),status='pass'))
    a=read(run/'A/A0_reproduction.json');assert a['status']=='pass' and a['counts']['evaluable_parents']==149102 and a['counts']['valid_fine_children']==361254
    checks.append(dict(check='A0 exact frozen counts and metrics',status='pass'))
    b=read(run/'B/label_agreement.json')['pooled']
    assert sum(b[k] for k in ['both_valid','corner_only','center_only','both_invalid'])==b['total']
    matrix=np.asarray(b['matrix']);assert matrix.sum()==b['both_valid'] and np.trace(matrix)==b['agree']
    for field in ['anchor_displacement_bins','local_scale_bins']:
        data=read(run/'B'/(field+'.json'))
        for key,rows in data.items():
            if key in ['all','all/linf','all/l2']:
                assert sum(r['count'] for r in rows)==b['both_valid']
    checks.append(dict(check='B pooled validity, confusion and bin denominator conservation',status='pass'))
    r4=read(run/'B/R4_label_final_EPE_comparison.json')
    for row in r4['rows']:
        for m in ['corner','center']:assert np.asarray(row['methods'][m]['matrix_true_rows_pred_columns']).sum()==row['valid']
    assert read(run/'A/summary.json')['R4']=='completed'
    checks.append(dict(check='R4 exact common-parent denominator and A2 backfill',status='pass'))
    stage=read(run/'C/stage_predictive_rows.json');reference={r['pair_id']:r for r in stage['common_BC/parent/coarse_linf']}
    actual_parents=actual_children=0
    for r in records('epe'):
        d=arrays_of(run,'C',committed(run,'C',r));mask=d['child_valid'].astype(bool)&(d['B_labels'][d['child_parent_position']]>=0)
        parents=np.unique(d['child_parent_id'][mask]);n=len(parents);actual_parents+=n;actual_children+=int(mask.sum())
        assert reference[r['pair_id']]['valid']==n
        for t in range(5):
            parent=next(x for x in stage['common_BC/parent/t'+str(t)] if x['pair_id']==r['pair_id'])
            child=next(x for x in stage['common_BC/child/t'+str(t)] if x['pair_id']==r['pair_id'])
            assert parent['valid']==n and child['valid']==int(mask.sum())
    checks.append(dict(check='C all-stage fixed common parent and child population',status='pass',parents=actual_parents,children=actual_children))
    outcomes=read(run/'C/rescues_losses_gain_classes.json')['pooled_and_per_scene']
    for row in outcomes:
        assert row['count']==row['further_gain']+row['stable_error']+row['worsened']
        assert row['count']==row['initial_lt2']+row['initial_ge2']
        assert row['rescued']<=row['initial_ge2'] and row['lost']<=row['initial_lt2']
    checks.append(dict(check='C rescue/loss and refinement-gain count conservation',status='pass'))
    plan=read(run/'A4/geometry_plan.json')
    assert len(plan['formal_unit_order'])==len(set(plan['formal_unit_order']))==8500
    assert len(set(plan['initial_timing_units']))==20 and set(plan['initial_timing_units'])<=set(plan['formal_unit_order'])
    timing=[plan['units'][k] for k in plan['initial_timing_units']]
    checks.append(dict(check='A4 first20 fixed units are formal, spread over scenarios',status='pass',
        scenes=len({r['scene'] for r in timing}),methods=sorted({r['method'] for r in timing}),rates=sorted({r['rate'] for r in timing}),seeds=sorted({r['seed'] for r in timing})))
    save(run/'independent_statistical_checks.json',dict(status='pass',time_beijing=now(),checks=checks))
    plotstatus={}
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        stats=read(run/'C/stage_predictive_power.json');keys=['common_BC/parent/'+x for x in ['coarse_linf','coarse_l2','t0','t1','t4']]
        ds=[stats[k]['paired_differences']['head-mconf']['Spearman'] for k in keys]
        values=np.array([r['mean_difference'] for r in ds]);limits=np.array([r['ci95'] for r in ds])
        fig,ax=plt.subplots(figsize=(8,3.8));ys=np.arange(len(keys))
        ax.errorbar(values,ys,xerr=np.array([values-limits[:,0],limits[:,1]-values]),fmt='o',color='#235789',capsize=4)
        ax.axvline(0,color='#777777',linestyle='--');ax.set_yticks(ys,['Coarse L-infinity','Coarse L2','Fine selection t=0','Refinement t=1','Refinement t=4']);ax.invert_yaxis()
        ax.set_xlabel('Spearman(head) - Spearman(mconf); negative favors head')
        ax.set_title('Common parents: first observed reversal at fine selection')
        ax.grid(axis='x',alpha=.2);fig.tight_layout();folder=run/'figures';folder.mkdir(exist_ok=True)
        for suffix in ['png','svg']:fig.savefig(folder/('stage_advantage.'+suffix),dpi=180)
        plt.close(fig);plotstatus=dict(status='completed',files=[str(folder/'stage_advantage.png'),str(folder/'stage_advantage.svg')])
    except ImportError as exc:plotstatus=dict(status='skipped',reason=repr(exc))
    save(run/'plot_status.json',plotstatus)
    print(json.dumps(dict(status='pass',checks=len(checks),plot=plotstatus)))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();main(Path(a.run))
