"""Finalize the completed T report from committed CPU evidence only."""
import sys,json,csv,hashlib,re,datetime
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path('/home/bupt/projects/SLiM-master')
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def tab(h,rows):return '\n'.join(['|'+'|'.join(h)+'|','|'+'|'.join(['---']*len(h))+'|']+['|'+'|'.join(str(v) for v in row)+'|' for row in rows])
def main(run):
    run=Path(run);out=Path(read(run/'context.json')['results']);versions=run/'report_versions';versions.mkdir(exist_ok=True)
    for path in [out/'PhaseT_Report.md',out/'artifact_manifest.json',run/'handoff.json']:
        target=versions/(path.name+'.initial')
        if not target.exists():target.write_bytes(path.read_bytes())
    funnel=list(csv.DictReader((out/'grid_funnel.csv').open()));rows=[];aggregates={}
    names=['source_pass','target_pass','both_pass','correct_parent','correct_parent_fine','correct_parent_good','any_parent','any_good','has_evaluable_child']
    for setting in ['S0','S1','S2','S3','S4','S5']:
        for region in ['low_texture','other_texture']:
            for stage in names:
                values=[x for x in funnel if x['setting']==setting and x['region']==region and x['stage']==stage];sc=defaultdict(list)
                num=sum(int(x['numerator']) for x in values);den=sum(int(x['denominator']) for x in values)
                for x in values:
                    if int(x['denominator']):sc[x['scene']].append(int(x['numerator'])/int(x['denominator']))
                av=sum(sum(v)/len(v) for v in sc.values())/len(sc) if sc else None
                x=dict(setting=setting,region=region,stage=stage,numerator=num,denominator=den,pooled=num/den if den else None,scene_equal=av,evaluable_pairs=sum(int(x['denominator'])>0 for x in values),scenes=len(sc))
                rows.append(x);aggregates[setting+'/'+region+'/'+stage]=x
    save(out/'grid_funnel_summary.json',rows)
    with (out/'grid_funnel_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    runtime=[]
    for setting in ['S0','S1','S2','S3','S4','S5']:
        data=[read(x) for x in (run/setting/'pairs').glob('*.result.json')]
        timing=[rep for x in data for rep in x['timing']['repeats']]
        r=dict(setting=setting,pairs=len(data),timed_forwards=len(timing),allocated_peak=max(x['allocated_peak'] for x in timing),reserved_peak=max(x['reserved_peak'] for x in timing),
            coarse_cap_pairs=sum(x['counts']['coarse_budget_cap_hit'] for x in data),fine_cap_limited_pairs=sum(x['counts']['fine_budget_cap_limited'] for x in data),
            coarse_prethreshold_topk=sum(x['counts']['coarse_prethreshold_topk'] for x in data),fine_prethreshold_topk=sum(x['counts']['fine_prethreshold_topk_total'] for x in data))
        for name in ['launch','status','cleanup']:
            p=run/setting/(name+'.json')
            if p.exists():r[name]=read(p)
        runtime.append(r)
    save(out/'runtime_summary.json',runtime)
    quality5=[];groups5=defaultdict(list)
    for setting in ['S0','S1','S2','S3','S4','S5']:
        for receipt in sorted((run/setting/'pairs').glob('*.json')):
            if receipt.name.endswith('.result.json'):continue
            c=read(receipt);a_info=c['files']['arrays'];payload=receipt.parent/a_info['name']
            if sha(payload)!=a_info['sha256']:raise RuntimeError('T committed array checksum mismatch')
            with np.load(payload,allow_pickle=False) as a:
                e=a['parent_epe'];low=a['low'].astype(bool);added=a['added'].astype(bool)
                for region,mask in [('all',np.ones(len(e),bool)),('low_texture',low),('added',added),('added_low_texture',added&low)]:
                    valid=mask&np.isfinite(e);den=int(valid.sum());num=int((valid&(e<5)).sum())
                    row=dict(setting=setting,region=region,pair_id=c['pair_id'],scene=c['scene'],parents=int(mask.sum()),valid_parent_denominator=den,numerator_lt5=num,
                        proportion_lt5=num/den if den else None,equal5=int((valid&(e==5)).sum()))
                    quality5.append(row);groups5[setting+'/'+region].append(row)
    stats5={}
    for key,values in groups5.items():
        scenes=defaultdict(list)
        for v in values:
            if v['proportion_lt5'] is not None:scenes[v['scene']].append(v['proportion_lt5'])
        den=sum(v['valid_parent_denominator'] for v in values);num=sum(v['numerator_lt5'] for v in values)
        stats5[key]=dict(numerator=num,denominator=den,pooled=num/den if den else None,scene_equal=sum(sum(v)/len(v) for v in scenes.values())/len(scenes) if scenes else None,
            evaluable_pairs=sum(v['valid_parent_denominator']>0 for v in values),scenes=len(scenes),equal5=sum(v['equal5'] for v in values))
    with (out/'parent_lt5.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,list(quality5[0]));writer.writeheader();writer.writerows(quality5)
    save(out/'parent_lt5_summary.json',stats5)

    d=aggregates
    loss=dict(denominator=d['S0/low_texture/both_pass']['denominator'],both_pass=d['S0/low_texture/both_pass']['numerator'],correct_parent=d['S0/low_texture/correct_parent']['numerator'],
        loss_both_to_correct=d['S0/low_texture/both_pass']['numerator']-d['S0/low_texture/correct_parent']['numerator'],
        scene_equal_both_pass=d['S0/low_texture/both_pass']['scene_equal'],scene_equal_correct_parent=d['S0/low_texture/correct_parent']['scene_equal'],
        conclusion='Largest loss on the correct-grid path occurs after both true grids survive norm filtering, before the correct grid pair becomes a native coarse parent. It is not principally explained by norm filtering on this visible-center denominator.',
        scope='100-pair R832 frozen validation; 7954 low-texture visible source grids across87 pairs/15 scenes. Any-parent usable recall is a separate branch.')
    save(out/'largest_low_texture_loss.json',loss)
    report=(versions/'PhaseT_Report.md.initial').read_text()
    report=re.sub(r'(?m)(\|[^\n]+\|)\n\n(?=\|)',r'\1\n',report)
    section=('\n\n## 补充分母、资源与结果解释\n\n'+tab(['低纹理S0阶段','分子','固定分母','pooled比例','场景等权比例'],[[x['stage'],x['numerator'],x['denominator'],round(x['pooled'],6),round(x['scene_equal'],6)] for x in rows if x['setting']=='S0' and x['region']=='low_texture'])+
        '\n\n在固定7954个可见低纹理源格分母上，7728个正确格对两侧均通过范数筛选，但仅3576个实际成为正确粗父点，流失4152个。对应场景等权比例从0.945433降到0.531922，正确格路径最大损失在两侧通过之后的粗匹配转化。主召回允许任意目标父格，S0为0.538336，不能和“正确父点产生好子点”分支混用。此结论仅针对当前可见源格分母，不推断被遮挡或不可投影区域。\n\n'+
        'GT1的S2−S0主召回为+3.060个百分点，95%区间[-0.029,8.161]个百分点，证据不足。GT2的134/171=78.363%只覆盖3108新增低纹理父点中的5.502%，不能称全部新增父点大多正确。GT3预算严格相同，但召回差区间下界未大于0；时间比1.0614通过≤1.10条件，联合判据不通过。\n\n'+
        tab(['设置','完成对','计时前向','allocated峰GiB','reserved峰GiB','粗上限触顶对','细预算限制对'],[[x['setting'],x['pairs'],x['timed_forwards'],round(x['allocated_peak']/2**30,4),round(x['reserved_peak']/2**30,4),x['coarse_cap_pairs'],x['fine_cap_limited_pairs']] for x in runtime])+
        '\n\n每对每档warmup1次加正式计时3次，共1800次正式计时前向、600次warmup；全部600设置单元完成。既有细预算是每父点top-k分配后再相对阈值，并非简单全局6000截断。触顶前后计数见[candidate_budget.csv](candidate_budget.csv)。CUDA段计时仅作描述，统计hook与GT不计入。\n\n'+
        '逐档真实命令、PID、退出状态和恢复清理：[runtime_summary.json](runtime_summary.json)。本次新增2500位姿求解，原生500单元核验后复用；无T训练或checkpoint。T开始11:59:29，全部报告/几何于12:16:59完成；后续仅整理已提交CPU证据。\n\n'+
        '低纹理沿用Phase2训练集校准的Sobel局部能量阈值0.0001920461654663086；源粗格的标签由子点继承，不由GT或本次成绩定义。[逐阶段汇总](grid_funnel_summary.json)、[主要流失判断](largest_low_texture_loss.json)。通用bootstrap返回的“single training seed”元数据来自复用函数，T本身没有训练种子；T区间的抽样单位是场景。\n\n原始报告和原清单保存在run/report_versions/，本次补充不重跑或覆盖任何有效pair缓存。'
    )
    section+='\n\nStrict parent-median EPE<5px counts and scene-equal values: [parent_lt5.csv](parent_lt5.csv), [summary](parent_lt5_summary.json). Equality at5px is counted separately; this is not computed by complementing >5px. Four strata are all, low texture, added, and added low texture.\n'
    (out/'PhaseT_Report.md').write_text(report+section+'\n\n'+(out/'texture_definition.md').read_text()+'\n')
    (out/'README.md').write_text('# Phase T results\n\n[完整报告](PhaseT_Report.md) · [主表](main_table.csv) · [判据](decisions.json) · [正确格去向](grid_funnel_summary.csv) · [资源和命令](runtime_summary.json) · [产物清单](artifact_manifest.json)\n')
    artifacts=[]
    for root in [out,run,ROOT/'diagnostics_threshold']:
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.name not in ['artifact_manifest.json','handoff.json','delivery_receipt.json','writer.lock'] and '__pycache__' not in p.parts and 'report_versions' not in p.parts:
                artifacts.append(dict(path=str(p),size=p.stat().st_size,sha256=sha(p)))
    save(out/'artifact_manifest.json',artifacts)
    handoff=read(run/'handoff.json');handoff.update(manifest_sha=sha(out/'artifact_manifest.json'),CPU_delivery_refreshed=datetime.datetime.now().astimezone().isoformat())
    save(run/'handoff.json',handoff)
    save(run/'delivery_receipt.json',dict(status='completed',time_beijing=datetime.datetime.now().astimezone().isoformat(),report_sha=sha(out/'PhaseT_Report.md'),manifest_sha=sha(out/'artifact_manifest.json'),handoff_sha=sha(run/'handoff.json'),
        report=str(out/'PhaseT_Report.md'),command=sys.argv,numeric_units_recomputed=0,presentation_and_committed_CPU_summary_only=True))
    print(json.dumps(dict(status='completed',report=str(out/'PhaseT_Report.md'),largest_loss=loss),ensure_ascii=False))
if __name__=='__main__':main(sys.argv[1])
