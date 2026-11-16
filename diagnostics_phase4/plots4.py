"""Publication-style CPU plots from committed summary JSON; no new inference."""
from common4 import *
def main(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=results(run);figdir=out/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
    made=[]
    def write(fig,name):
        fig.tight_layout()
        for ext in ['png','svg']:fig.savefig(figdir/(name+'.'+ext),bbox_inches='tight')
        plt.close(fig);made.append(name)
    p1=read(out/'P1_residual_bins.json')['pooled']
    v=[r for r in p1 if r['anchor']=='corner' and r['region']=='all']
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    x=np.arange(4);labels=[r['bin'] for r in v]
    bottom=np.zeros(4)
    for k,label,col in [('without_fine','No fine output','#9aa1a8'),('fine_without_valid_GT','Fine output, no valid GT','#e8b66d'),('valid_final_parents','Valid final EPE','#4b8baa')]:
        vals=np.array([r[k] for r in v]);ax[0].bar(x,vals,bottom=bottom,label=label,color=col);bottom+=vals
    ax[0].set_xticks(x,labels);ax[0].set_xlabel('Coarse L-infinity residual [input pixels]');ax[0].set_ylabel('Parents');ax[0].legend(fontsize=8)
    for k,label,style in [('initial_lt2_fraction','t=0 median EPE < 2','o--'),('final_lt2_fraction','t=4 median EPE < 2','o-'),('final_lt1_fraction','t=4 median EPE < 1','s-')]:
        ax[1].plot(x,[r[k] for r in v],style,label=label)
    ax[1].set_xticks(x,labels);ax[1].set_ylim(0,1);ax[1].set_ylabel('Fraction of evaluable parents');ax[1].legend(fontsize=8)
    fig.suptitle('P1: coarse residual association; no window-overlap inference',y=1.03);write(fig,'P1_residual_bins')
    p2=read(out/'P2_ranking.json')['summaries']['all/child']['scene_equal']
    fig,ax=plt.subplots(figsize=(7,4.5))
    colors=dict(mconf='#555555',head_p='#999999',fine='#2878a3',S_m='#d58024',S_h='#6b58a6',random='#aaa466')
    for m,d in p2.items():
        ax.scatter(d['Spearman'],d['AUSE'],label=m,s=65,color=colors[m]);ax.annotate(m,(d['Spearman'],d['AUSE']),xytext=(5,5),textcoords='offset points')
    ax.set_xlabel('Spearman (lower is better)');ax.set_ylabel('AUSE [input pixels] (lower is better)');ax.grid(alpha=.2)
    ax.set_title('P2: scene-equal ranking on valid-GT children');write(fig,'P2_score_tradeoff')
    p=read(out/'P2b_within_window.json')['summaries']
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,scope,metric,title in [(axes[0],'all/actual','valid_and_correct_probability','All-candidate strategy: valid AND EPE < 2'),(axes[1],'all/common_valid','precision_lt2','Common-valid windows: EPE < 2'),(axes[2],'all/common_valid','median_EPE','Common-valid: pair distribution median EPE')]:
        d=p[scope]['scene_equal'];names=list(d);vals=[d[m][metric] for m in names]
        ax.bar(names,vals,color=[colors.get(m,'#56a572') for m in names]);ax.set_title(title,fontsize=9);ax.grid(axis='y',alpha=.2)
        if metric!='median_EPE':ax.set_ylim(0,1)
        else:ax.set_ylabel('Input pixels')
    fig.text(.5,-.03,'Random median: equal window mass, 1/v_i per candidate; not the median of window medians.',ha='center',fontsize=9)
    write(fig,'P2b_within_window')
    p=read(out/'P3_filter_quality.json')['table'];d={r['combo']:r for r in p}
    names=['fine','S_m','S_h','mconf','random']
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,key,title in [(axes[0],'epipolar_precision','Epipolar precision (all retained children)'),(axes[1],'precision_lt1','EPE < 1 (retained valid-GT children)'),(axes[2],'GT_evaluable_rate','GT evaluability (all retained children)')]:
        ax.bar(names,[d[m+'_80'][key] for m in names],color=[colors[m] for m in names]);ax.set_ylim(0,1);ax.set_title(title,fontsize=9);ax.grid(axis='y',alpha=.2)
    fig.suptitle('P3: 80% retained per pair before any GT check',y=1.03);write(fig,'P3_filter_quality')
    q=read(out/'P4_step_quality.json')['table'];pose=read(out/'P4_pose.json')['pooled_seed_mean'];g=read(out/'G7.json')
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    titles=['Pose AUC@5 delta','EPE < 1 delta [percentage points]','Epipolar precision delta [percentage points]']
    metrics=['AUC@5','precision_lt1','epipolar_precision']
    for ax,metric,title in zip(axes,metrics,titles):
        points=sorted(g['steps'],key=lambda r:r['step']);f=1 if metric=='AUC@5' else 100
        y=np.array([r['conditions'][metric]['difference'] for r in points])*f
        low=np.array([r['conditions'][metric]['ci95'][0] for r in points])*f;high=np.array([r['conditions'][metric]['ci95'][1] for r in points])*f
        x=np.array([r['step'] for r in points]);ax.errorbar(x,y,yerr=[y-low,high-y],fmt='o',capsize=3,color='#28638c')
        ax.axhline(0,color='gray',lw=1);ax.axhline(points[0]['conditions'][metric]['tolerance']*f,color='#b6463f',ls='--',label='Non-inferiority margin')
        ax.set_xticks([0,1,2,3]);ax.set_xlabel('Replayed refinement steps');ax.set_title(title,fontsize=9);ax.legend(fontsize=7)
    fig.suptitle('P4: step minus historical t4; 95% whole-scene paired intervals',y=1.03);write(fig,'P4_truncation')
    artifact(run,'figure_manifest',dict(status='completed',figures=[dict(name=n,png='figures/'+n+'.png',svg='figures/'+n+'.svg') for n in made]))
    return made
