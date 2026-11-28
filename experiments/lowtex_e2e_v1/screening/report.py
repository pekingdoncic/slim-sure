import csv, json, math, os
import numpy as np
from .data import EXP,OUT,N,U,read,save
MODELS=["exp0","exp1a","rru_matched","single","crop","joint"]
DESC={"exp0":"官方权重原始 RRU4，端到端","exp1a":"原始 RRU4，仅粗正样本 LT/other 分母独立加权",
      "rru_matched":"重新初始化原生 RRU4，10% 头部热身","single":"单尺度一步头，10% 热身",
      "crop":"多尺度 Crop-Fuse 一步头，10% 热身","joint":"有条件组合纹理损失与多尺度一步头，匹配热身"}

def run_path(model):return EXP/"outputs/fast_screening_v1/screening"/model

def emit_csv(p,header,rows):
    tmp=p.with_suffix(".tmp")
    with tmp.open("w",newline="") as f:
        writer=csv.DictWriter(f,header,extrasaction="ignore");writer.writeheader();writer.writerows(rows)
    tmp.replace(p)

def emit_text(p,lines):
    tmp=p.with_name(p.name+".tmp");tmp.write_text("\n".join(lines)+"\n");tmp.replace(p)

def scores(model,side=640,route="native"):
    d=run_path(model)
    if not (d/"complete.json").exists():return None
    step=read(d/"complete.json")["step"]
    result=read(d/"validation"/f"step_{step:08d}"/route/f"r{side}"/"accepted.json")["scores"]
    path=d/"timing.json"
    if path.exists():
        timing=next(v for v in read(path)["results"] if v["side"]==side and v["route"]==route)
        result["texturepoor"]["screening"].update(matcher_ms_median=timing["median_ms"],refine_ms_median=timing["refine_ms"],
                      peak_allocated_bytes=timing["allocated"],peak_reserved_bytes=timing["reserved"])
    return result

def judge(name,control):
    a,b=scores(name),scores(control)
    if not a or not b or not all((run_path(n)/"timing.json").exists() for n in (name,control)):return {"candidate":name,"control":control,"disposition":"PENDING","advance":False}
    md,base=a["megadepth"],b["megadepth"]
    scene_ids=sorted(md["per_scene"]); assert scene_ids==sorted(base["per_scene"])
    delta=[md["per_scene"][s]["lt_recall"]-base["per_scene"][s]["lt_recall"] for s in scene_ids
           if md["per_scene"][s]["lt_recall"] is not None and base["per_scene"][s]["lt_recall"] is not None]
    if not delta:return dict(candidate=name,control=control,disposition="INCONCLUSIVE",advance=False,reason="empty paired LT denominator")
    rng=np.random.default_rng(66)
    ci=np.quantile(rng.choice(delta,(2000,len(delta)),replace=True).mean(1),[.025,.975])
    dl=float(np.mean(delta))
    auc=[a[k]["scene_equal"]["auc@10"]-b[k]["scene_equal"]["auc@10"] for k in ("megadepth","texturepoor")]
    lt,ref=md["screening"]["lt"],base["screening"]["lt"]
    lat=a["texturepoor"]["screening"]["matcher_ms_median"]/b["texturepoor"]["screening"]["matcher_ms_median"]
    ratio=a["texturepoor"]["screening"]["refine_ms_median"]/max(b["texturepoor"]["screening"]["refine_ms_median"],1e-9)
    other=md["scene_equal"]["other_recall"]-base["scene_equal"]["other_recall"]
    if name=="exp1a":
        mech=lt["strict_recall"]>ref["strict_recall"] and lt["gt_rank"]<ref["gt_rank"] and other>=-.01
    else:
        mech=lt["epe"]<ref["epe"] and lt["precision1"]>=ref["precision1"] and lt["precision2"]>=ref["precision2"] and ratio<=1 and lat<=1
    advance=bool(dl>=.01 and ci[0]>0 and min(auc)>=-1 and lat<=1.1 and mech)
    return dict(candidate=name,control=control,disposition="PROMISING_ONE_CYCLE" if advance else "INCONCLUSIVE",
                advance=advance,lt_recall_delta=dl,lt_ci=ci.tolist(),auc10_delta_pp=auc,other_recall_delta=other,
                latency_ratio=lat,refine_ratio=ratio,mechanism_same_direction=bool(mech))

def render():
    OUT.mkdir(parents=True,exist_ok=True)
    gates=[judge("exp1a","exp0"),judge("crop","single"),judge("crop","rru_matched")]
    if scores("joint"):gates.extend([judge("joint","crop"),judge("joint","rru_matched")])
    save(OUT/"selection.json",gates)
    comp=[];coverage=[];mdlines=["# 4080 单周期筛选报告","",
        "R640 训练使用全部官方合法有向索引，R640/R832 开发评测单列。百分号只用于 AUC；召回、精度、覆盖率均用 0–1。最终测试场景封存。",
        "", "## 当前结果",""]
    st=read(OUT/"state.json") if (OUT/"state.json").exists() else {}
    cfg=read(OUT/"selected_config.json") if (OUT/"selected_config.json").exists() else {}
    failure=read(OUT/"failures.json") if (OUT/"failures.json").exists() else {}
    for name in MODELS:
        d=run_path(name); complete=(d/"complete.json").is_file()
        prog=read(d/"progress.json") if (d/"progress.json").exists() else {}
        if complete:prog=read(d/"complete.json")
        status="COMPLETE" if complete else "BLOCKED" if name in failure else "RUNNING" if prog or st.get("mode")=="screening" and st.get("experiment")==name else "NOT_RUN"
        if name=="joint" and (OUT/"joint_decision.json").exists() and not read(OUT/"joint_decision.json")["run"]:status="SKIPPED"
        coverage.append(dict(experiment=name,status=status,planned_rows=N,planned_unordered=U,**prog))
        mdlines.append(f"- {name}，{DESC[name]}。{status}，已提交更新 {prog.get('step','—')}，训练访问 {prog.get('visits','—')}。")
        if name in failure:mdlines.append("  失败记录 "+str(failure[name]))
        for label in (("native","rru1") if name in ("exp0","rru_matched") else ("native",)):
            for side in (640,832):
                row=dict(Experiment=name,variant=label,eval_resolution=side,status=status,
                         training_time_s=prog.get("train_seconds"),training_active_wall_s=prog.get("active_wall_seconds"),
                         train_peak_allocated_bytes=(prog.get("peak_bytes") or [None,None])[0],train_peak_reserved_bytes=(prog.get("peak_bytes") or [None,None])[1])
                a=scores(name,side,label)
                if a:
                    t,m=a["texturepoor"],a["megadepth"];l=m["screening"]["lt"];o=m["screening"]["other"]
                    for dataset,z in [("TexturePoor",t),("MegaDepth",m)]:
                        for deg in (5,10,20):row[f"{dataset}_AUC{deg}"]=z["scene_equal"][f"auc@{deg}"]
                        row[dataset+"_coverage0"]=z["scene_equal"]["coverage0_mean"];row[dataset+"_coverage1"]=z["scene_equal"]["coverage1_mean"]
                    row.update(LT_recall=m["scene_equal"]["lt_recall"],other_recall=m["scene_equal"]["other_recall"],LT_EPE=l["epe"],
                               LT_precision1=l["precision1"],LT_precision2=l["precision2"],LT_GT_rank=l["gt_rank"],
                               LT_correct_coarse=l["strict_coarse"],LT_strict_recall=l["strict_recall"],other_correct_coarse=o["strict_coarse"],
                               other_GT_rank=o["gt_rank"],inference_ms=t["screening"]["matcher_ms_median"],refine_ms=t["screening"]["refine_ms_median"],
                               VRAM_allocated_bytes=t["screening"]["peak_allocated_bytes"],VRAM_reserved_bytes=t["screening"]["peak_reserved_bytes"])
                comp.append(row)
    c=["Experiment","variant","eval_resolution","status",*[f"{dom}_AUC{k}" for dom in ("TexturePoor","MegaDepth") for k in (5,10,20)],
       "LT_recall","other_recall","LT_EPE","LT_precision1","LT_precision2","LT_GT_rank","LT_correct_coarse","LT_strict_recall","other_correct_coarse","other_GT_rank",
       "TexturePoor_coverage0","TexturePoor_coverage1","MegaDepth_coverage0","MegaDepth_coverage1","inference_ms","refine_ms","VRAM_allocated_bytes","VRAM_reserved_bytes",
       "training_time_s","training_active_wall_s","train_peak_allocated_bytes","train_peak_reserved_bytes"]
    comparisons={("exp1a","native"):("exp0","native"),("single","native"):("rru_matched","native"),
                 ("crop","native"):("single","native"),("joint","native"):("crop","native"),
                 ("rru_matched","rru1"):("rru_matched","native"),("exp0","rru1"):("exp0","native")}
    bykey={(z["Experiment"],z["variant"],z["eval_resolution"]):z for z in comp}
    metrics=[k for k in c if k not in ("Experiment","variant","eval_resolution","status")]
    for row in comp:
        ctrl=comparisons.get((row["Experiment"],row["variant"]))
        if ctrl:
            row["matched_control"]="/".join(ctrl)
            other=bykey[(*ctrl,row["eval_resolution"])]
            for k in metrics:
                if k in ("training_time_s","training_active_wall_s") and row["variant"]=="rru1":continue
                if row.get(k) is not None and other.get(k) is not None and row["status"]==other["status"]=="COMPLETE":
                    row["delta_"+k]=row[k]-other[k]
    emit_csv(OUT/"FINAL_COMPARISON.csv",c+["matched_control"]+["delta_"+k for k in metrics],comp)
    emit_csv(OUT/"TRAINING_COVERAGE.csv",["experiment","status","planned_rows","planned_unordered","step","visits","unique_rows","unique_unordered",
                    "repeated_rows","repeated_unordered","invalid_supervision","zero_valid_fine_gt","ordered_coverage","unordered_coverage","bins_visited"],coverage)
    legacy=read(OUT/"legacy_r832/verified.json")
    lp=legacy["progress"]
    mdlines.extend(["","## 保留的 R832 历史基线","",
        f"正常 checkpoint 在 {lp['step']} 步写入，学习率 {legacy['lr']}，访问 {lp['cursor']} 对，有向覆盖 {100*lp['coverage']:.4f}%，无向覆盖 {100*lp['unordered_coverage']:.4f}%。历史训练平均 {legacy['pairs_per_second']:.3f} 对/秒，峰值 allocated/reserved 分别为 {legacy['peak_bytes'][0]/2**30:.3f}/{legacy['peak_bytes'][1]/2**30:.3f} GiB。",
        "各节点两域开发指标、checkpoint SHA-256、正常退出验证在 legacy_r832/verified.json 和 transition.json。历史结果使用不同分辨率和预算。"])
    mdlines.extend(["","## 配置、实测与证据","",f"当前选型 {cfg.get('id','测速中')}。训练 R640，effective batch 4，BF16，冻结 BN，clip 1。",
                   "DATA 与预算见 protocols/fast_screening_v1.md、frozen_inputs.json。每组身份、初始化、源文件哈希、训练日志、梯度/参数变化、恢复记录、两域逐对结果均在 outputs/fast_screening_v1 对应子目录。",
                   "TRAINING_THROUGHPUT.csv 包含独立 GPU 实测。TRAINING_COVERAGE.csv 使用真实 checkpoint。FINAL_COMPARISON.csv 的训练耗时以秒表示，正式选型计时在独立进程上使用 100 个固定 TexturePoor 开发对、30 次预热、3 轮重复；推理为 matcher 中位毫秒，VRAM 为字节、推理和训练分别给出。空值表示尚无该次测量。","",
                   "## 初步趋势与不确定性",""])
    for g in gates:mdlines.append(f"- {g['candidate']} 对 {g['control']}：{g['disposition']}。"+(f" LT 召回差 {100*g['lt_recall_delta']:.3f} 个百分点，配对 95% 区间 {g['lt_ci']}，两域 AUC@10 差 {g['auc10_delta_pp']}，全模型延迟比 {g['latency_ratio']:.3f}。" if "lt_ci" in g else ""))
    for x in coverage:
        if x["status"]=="COMPLETE":
            rr=next(t for t in comp if t["Experiment"]==x["experiment"] and t["variant"]=="native" and t["eval_resolution"]==640)
            mdlines.append(f"- {x['experiment']} 完整覆盖 {x['unique_rows']} 条、{x['unique_unordered']} 个无向对，训练更新耗时 {x['train_seconds']/3600:.2f} 小时；R640 低纹理召回 {rr['LT_recall']:.4f}，低纹理 EPE {rr['LT_EPE']:.4f}px，全模型中位 {rr['inference_ms']:.2f} ms。")
    mdlines.append("单轮结论用于筛选。尚未验证的假设包括多轮后的排序、跨随机种子泛化和封存场景上的性能。")
    emit_text(OUT/"FAST_SCREENING_REPORT.md",mdlines)
    emit_text(OUT/"MODEL_SELECTION.md",["# 开发集选模","",*mdlines[mdlines.index("## 初步趋势与不确定性")+2:]])
    rec=[g["candidate"] for g in gates if g["advance"]]
    completed=all(x["status"] in ("COMPLETE","SKIPPED","BLOCKED") for x in coverage)
    emit_text(OUT/"NEXT_STEPS.md",["# 后续建议","",("建议人工复核并优先考虑 extended 的模型："+", ".join(sorted(set(rec)))) if rec else
               ("本轮未出现通过全部筛选门槛的模型，优先查阅负结果与置信区间，再决定 3–5 轮投入。" if completed else "主要组的一次全池覆盖和两域终验证仍在进行。待实测结果齐备后决定 3–5 轮投入。"),
               "本工作流到筛选汇总结束。长期 full、多种子扩展及 PRO 5000 实验分开安排。"])
    return gates

if __name__=="__main__":render()
