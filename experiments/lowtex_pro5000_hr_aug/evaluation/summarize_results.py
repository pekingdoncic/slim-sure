"""Render a run from its receipts, summaries and per-update measurements, including partial runs."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean,median
from ..utils.run_io import EXP,config,locations,read

MODELS = ("P0","P1","P2","P3","P4")
ARCH = {
    "P0":("官方零样本高分辨率参照","官方权重","models/slim_wrapper.py CCSMatcher.forward → src/slim.py SLiM.forward","原生范数筛选、双向概率、细窗与 RRU-4；无训练。"),
    "P1":("衡量相同训练预算本身的作用","P0","CCSMatcher.training_loss / losses/sparse_matching_loss.py native_focal","B×1×R×R → 192×R/8×R/8；全合法行列竞争，原 focal、细窗 focal 和四步回归更新原模型。"),
    "P2":("检验跨视图上下文能否提高特征区分度","P1","models/ccfi.py CCFI.forward","1/8 掩码池化至 1/32，96 维两图关联和门控残差回写 192 维描述子。原粗细损失回传 CCFI 与主干。"),
    "P3":("衡量正确候选保留率与稀疏计算成本","P2","models/block_router.py BlockRouter / models/sparse_correlation.py _EdgeDot / models/sparse_dual_softmax.py dual_logp","双向块并集→有界 halo→E 条去重边；全 E 条边归一化和 MNN。几何多正块监督更新 CCFI，边 focal 更新关联特征。"),
    "P4":("检验池内多候选能否转为严格正确父点","P3","models/hypothesis_graph.py HypothesisGraph / losses/hard_negative_ranking.py ranking","每源格 M 个 64 维投影候选经有界图聚合，标量残差回写 E 条边；原分母上重做归一化。可靠竞争负例的 margin loss 回传图、CCFI 与主干。")
}


def csv_file(path,rows,columns):
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def fmt(x):
    return "—" if x is None else f"{x:.5g}" if isinstance(x,float) else str(x)


def table(rows,cols):
    lines=["| "+" | ".join(cols)+" |","|"+"---|"*len(cols)]
    return "\n".join(lines+["| "+" | ".join(fmt(r.get(c)) for c in cols)+" |" for r in rows])


def maybe(path):
    return read(path) if path.is_file() else {}


def log_rows(path):
    # Retried work after an interruption may replay updates. Only the committed path contributes.
    values={}
    if path.exists():
        last=None
        with path.open() as fp:
            for line in fp:
                try:
                    r=json.loads(line)
                except json.JSONDecodeError:
                    continue
                values[r["step"]]=dict(step=r["step"],loss=r["loss"],seconds_per_step=r["seconds_per_step"])
                last=r
        if last:
            values[last["step"]]=last
    return [values[k] for k in sorted(values)]


def paired_interval(paths,model,baseline):
    import numpy as np
    from collections import defaultdict
    def load_pairs(name):
        path=paths.get(name)
        if path is None or not path.is_file():
            return {}
        with path.open() as f:
            return {r["pair_id"]:r for r in map(json.loads,f) if "lt" in r}
    a,b=load_pairs(model),load_pairs(baseline)
    if not a or set(a)!=set(b):
        return None
    groups=defaultdict(list)
    for pid,r in a.items():
        den=r["lt"]["den"]
        other=b[pid]
        if r["scene"]!=other["scene"] or den!=other["lt"]["den"]:
            raise ValueError(f"paired LT denominator mismatch at {pid}")
        if den:
            groups[r["scene"]].append((r["lt"]["success"]-other["lt"]["success"])/den)
    if len(groups)<2:
        return None
    values=np.array([np.mean(groups[s]) for s in sorted(groups)])
    sample=np.random.default_rng(66).choice(values,size=(2000,len(values)),replace=True).mean(1)
    return dict(delta=float(values.mean()),lower=float(np.quantile(sample,.025)),upper=float(np.quantile(sample,.975)),
                scenes=len(values),pairs=sum(map(len,groups.values())))


def figures(directory,scaling):
    if not scaling:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for x,y in (("side","auc10"),("side","lt_recall"),("side","latency_ms"),("side","reserved_gib"),
                ("latency_ms","auc10"),("reserved_gib","auc10")):
        fig,ax=plt.subplots(figsize=(6,4))
        points=[]
        for m in MODELS:
            rows=sorted([r for r in scaling if r["model"]==m and r.get(x) is not None and r.get(y) is not None],key=lambda r:r["side"])
            if rows:
                ax.plot([r[x] for r in rows],[r[y] for r in rows],marker="o",label=m)
                points.extend((r[x],r[y]) for r in rows)
        if x!="side" and points:
            best=-float("inf")
            front=[]
            for a,b in sorted(points,key=lambda p:(p[0],-p[1])):
                if b>best:
                    front.append((a,b))
                    best=b
            ax.plot(*zip(*front),ls="--",color="black",label="Pareto")
        if points:
            ax.set(xlabel=x,ylabel=y)
            ax.legend()
            fig.tight_layout()
            fig.savefig(directory/f"{y}_by_{x}.png",dpi=160)
        plt.close(fig)


def summarize(mode,run_id,output_root=None,report_root=None):
    _,default=locations(mode,run_id)
    report=report_root or default
    final=report/"final"
    fig=final/"figures"
    fig.mkdir(parents=True,exist_ok=True)
    progress=maybe(report/"progress.json")
    freeze=maybe(report/"model_freeze.json").get("models",{})
    formal,ablation,eff,scaling,cover,speed,details,negative=[],[],[],[],[],[],[],[]
    improvements={}
    pairs_paths={}
    stage_rows=[]
    for name,v in progress.get("stages",{}).items():
        stage_rows.append(dict(stage=name,status=v.get("status"),exit_code=v.get("exit_code"),
                               started=v.get("started"),ended=v.get("ended"),command=v.get("command")))
    for m in MODELS:
        root=output_root/m/run_id if output_root else locations(mode,run_id,m)[0]
        side=freeze.get(m,{}).get("resolution",int(progress.get("options",{}).get("resolution_plan","1024").split(",")[-1]))
        chosen=freeze.get(m,{})
        source=root/f"R{side}"
        fr={"model":m}
        for domain in ("texturepoor","megadepth","scannet","hpatches"):
            path=root/"test"/(domain+".json")
            res=maybe(path)
            fr[domain+"_status"]=res.get("status","NOT_RUN")
            fr[domain+"_source"]=str(path.relative_to(root)) if res else None
            stat=res.get("scene_equal" if domain=="texturepoor" else "global_metrics",{})
            for k in ((3,5,10) if domain=="hpatches" else (5,10,20)):
                fr[f"{domain}_auc{k}"]=stat.get(f"auc@{k}")
        formal.append(fr)
        before=maybe(source/"before/megadepth.json")
        md=maybe(source/"selected/megadepth.json") or maybe(source/"after/megadepth.json")
        tp=maybe(source/"selected/texturepoor.json") or maybe(source/"after/texturepoor.json")
        dev=md.get("scene_equal",{})
        pairs_paths[m]=source/md.get("pair_records","after/megadepth_pairs.jsonl")
        row={k:dev.get(k) for k in ("lt_recall","other_recall","lt_epe","lt_precision","lt_coarse_recall","lt_rank","lt_before_rank",
             "lt_recall1","lt_recall5","lt_recall10","lt_route_recall","lt_halo_recall","lt_topm_recall","lt_topm_conversion",
             "lt_feature_norm","lt_margin","lt_pool_rate")}
        row.update(model=m,side=side,md_dev_auc10=dev.get("auc@10"),tp_dev_auc10=tp.get("scene_equal",{}).get("auc@10"),
                   status=md.get("status","NOT_RUN"),before_lt=before.get("scene_equal",{}).get("lt_recall"))
        for p in sorted((root/"efficiency").glob("R*/benchmark.json")):
            v=read(p)
            if v.get("status")!="PASS":
                continue
            tim=v["timing_ms"]["Total"]
            item=dict(model=m,side=v["side"],params=v["params"],new_params=v["new_params"],partial_flops=v["covered_flops"],
                      latency_ms=tim["median"],mean_ms=tim["mean"],p95_ms=tim["p95"],reserved_gib=v["peak_reserved_gib"],
                      allocated_gib=v["peak_allocated_gib"],process_gib=v["driver"]["peak_process_gib"],
                      auc10=v["scene_equal"]["auc@10"],lt_recall=v["scene_equal"].get("lt_recall"),
                      edges=v["global_metrics"].get("edges"),source=str(p.relative_to(root)),run_id=run_id)
            item.update(precision=v["precision"],gpu=v["driver"]["name"],uuid=v["driver"]["uuid"],
                        utilization=v["driver"].get("utilization_mean"),unknown_operators=len(v["uncounted_operators"]),
                        flop_audit=v.get("profiler_error") or v.get("convention"))
            for stage,times in v["timing_ms"].items():
                for metric,ms in times.items():
                    item[stage+"_"+metric+"_ms"]=ms
            eff.append(item)
            scaling.append(item)
            if v["side"]==side:
                row.update({k:item[k] for k in ("params","partial_flops","latency_ms","reserved_gib")})
        ablation.append(row)
        training=[]
        for path in sorted(root.glob("R*/training_summary.json")):
            v=read(path)
            logs=[x for x in log_rows(path.parent/"training.jsonl") if x["step"]<=v["step"]]
            cv=dict(model=m,side=v["side"],run_id=run_id,**v["coverage"],backbone_steps=v["backbone_steps"],
                    source=str(path.relative_to(root)))
            cover.append(cv)
            times=[r["seconds_per_step"] for r in logs]
            sp=dict(model=m,side=v["side"],updates=len(times),seconds=sum(times) if times else None,
                    pairs_per_s=v["config"]["effective_batch_size"]/mean(times) if times else None,
                    seconds_per_step=median(times) if times else None,precision=v["precision"],micro=v["job"]["micro"],
                    peak_reserved_gib=v["memory"]["reserved_gib"],source=str(path.relative_to(root)),run_id=run_id)
            speed.append(sp)
            training.append({"resolution":v["side"],"coverage":cv,"throughput":sp,"first_loss":logs[0]["loss"] if logs else None,
                             "last_loss":logs[-1]["loss"] if logs else None,"best_step":v.get("best_step"),
                             "last_components":logs[-1].get("metrics",{}) if logs else {}})
        why,baseline,code,flow=ARCH[m]
        details.extend([f"## {m}","",f"{why}。对照为 {baseline}。实现位于 `{code}`。{flow}","",
                        f"选定权重及哈希 {json.dumps(chosen,ensure_ascii=False)}","",
                        "训练参数、预算与收敛记录","", "```json",json.dumps(dict(config=config(m),training=training),ensure_ascii=False,indent=2),"```","",
                        "开发指标","",table([row],list(row)),"", "正式测试","",table([fr],list(fr)),""])
    cols=["model"]+[f"{d}_{v}" for d in ("texturepoor","megadepth","scannet") for v in ("status","auc5","auc10","auc20")]
    for name,rows,default_cols in (("final_comparison",formal,cols),("ablation_results",ablation,["model"]),
         ("efficiency_results",eff,["model","side","latency_ms","reserved_gib"]),("resolution_scaling",scaling,["model","side","auc10","lt_recall"]),
         ("training_coverage",cover,["model","side","row_coverage","physical_coverage"]),("training_throughput",speed,["model","side","pairs_per_s"])):
        keys=list(dict.fromkeys([*default_cols,*(k for r in rows for k in r),"run_id"]))
        csv_file(final/(name+".csv"),[dict(r,run_id=run_id) for r in rows],keys)
    header=f"Run `{run_id}` · {mode} · {progress.get('status','NOT_RUN')}\n\n"
    comparison=header+table(formal,cols)+"\n"
    (final/"final_comparison.md").write_text("# 正式测试\n\n"+comparison)
    questions=[("P2","P1","lt_rank","CCFI 低纹理 GT 平均排名",False),("P3","P2","lt_pool_rate","BBR 正确候选覆盖率",True),
               ("P3","P2","latency_ms","BBR 开销（同步中位毫秒）",False),("P4","P3","lt_recall1","LHG 正确候选排第一的比例",True),
               ("P4","P1","lt_recall","低纹理成功源格",True),("P4","P1","other_recall","其他纹理成功源格",True),
               ("P4","P1","md_dev_auc10","MegaDepth 开发位姿 AUC@10",True),("P4","P1","tp_dev_auc10","TexturePoor 开发位姿 AUC@10",True),
               ("P4","P1","params","参数量",False),("P4","P1","reserved_gib","同卡峰值保留显存",False)]
    rowmap={r["model"]:r for r in ablation}
    observations=[]
    for m,b,k,label,higher in questions:
        a,z=rowmap[m].get(k),rowmap[b].get(k)
        delta=a-z if a is not None and z is not None else None
        statement=f"{label}，{m}−{b} = {fmt(delta)}。"
        observations.append(statement)
        if mode!="smoke" and delta is not None and (delta<0 if higher else delta>0):
            negative.append(statement)
    for m in MODELS[2:]:
        for key in ("lt_recall","other_recall","md_dev_auc10","tp_dev_auc10"):
            a,b=rowmap[m].get(key),rowmap["P1"].get(key)
            improvements[f"{m}-P1 {key}"]=a-b if a is not None and b is not None else None
    formal_deltas=[]
    for r in formal[2:]:
        for domain in ("texturepoor","megadepth","scannet"):
            ref=formal[1].get(domain+"_auc10")
            value=r.get(domain+"_auc10")
            formal_deltas.append(dict(model=r["model"],dataset=domain,metric="AUC@10",reference="P1",
                delta_points=value-ref if value is not None and ref is not None else None))
    retention=[]
    for m,b in (("P2","P1"),("P3","P2"),("P4","P3")):
        a,z=rowmap[m],rowmap[b]
        keys=("lt_recall","md_dev_auc10","tp_dev_auc10","latency_ms")
        interval=paired_interval(pairs_paths,m,b) if mode!="smoke" else None
        if mode=="smoke" or interval is None or any(a.get(k) is None or z.get(k) is None for k in keys):
            decision="待完整可比开发与效率结果"
        else:
            meets=(interval["delta"]>=.01 and interval["lower"]>0 and a["md_dev_auc10"]-z["md_dev_auc10"]>=-1 and
                   a["tp_dev_auc10"]-z["tp_dev_auc10"]>=-1 and a["latency_ms"]<=1.1*z["latency_ms"])
            decision="进入独立种子复验" if meets else "暂缓叠加，保留负结果和原对照"
        retention.append(dict(module=m,baseline=b,decision=decision,
                              lt_scene_bootstrap_95=([interval["lower"],interval["upper"]] if interval else None)))
    assessment=("Smoke 仅用于工程验收。科学判定待正式训练及冻结评测。" if mode=="smoke" else
                "开发差值为单次运行的描述性观测。正式测试的逐域结果与覆盖率见主表。")
    selection=header+json.dumps(freeze,ensure_ascii=False,indent=2)
    issues=[]
    for stage in stage_rows:
        if stage["status"] in ("FAIL","BLOCKED"):
            issues.append(f"{stage['stage']} · {stage['status']} · exit {stage['exit_code']}")
    for m in MODELS:
        root=output_root/m/run_id if output_root else locations(mode,run_id,m)[0]
        for path in (root/"test").glob("*.json"):
            info=maybe(path)
            if info.get("status")=="BLOCKED":
                issues.append(f"{m} {path.stem}: {info.get('reason')}")
    failure=maybe(report/"latest_error.json")
    if failure and progress.get("errors"):
        tail=str(failure.get("traceback","")).splitlines()
        issues.append(f"{failure.get('stage')} · {tail[-1] if tail else 'see latest_error.json'}")
    issue_text="\n\n".join(issues) or "本次未记录阶段性工程错误。"
    (final/"MODEL_SELECTION.md").write_text("# 模型选择\n\n"+selection+"\n\n"+table(retention,["module","baseline","lt_scene_bootstrap_95","decision"])+"\n")
    (final/"NEGATIVE_RESULTS.md").write_text("# 负结果与工程阻塞\n\n"+header+assessment+"\n\n"+
               "\n".join(negative or ["本次可用数据中没有可列出的不利差值。"])+"\n\n"+issue_text+"\n")
    nxt=("修复最新错误并按原 run_id 恢复。" if progress.get("errors") else "补齐标记 BLOCKED 的数据集。" if progress.get("blocked")
         else "使用相同划分和独立训练种子复核模块收益与代价。" if mode=="full" else "依据冻结的开发指标和预算决定是否进入下一训练阶段。")
    (final/"NEXT_STEPS.md").write_text("# 下一步\n\n"+header+nxt+"\n")
    summary=header+assessment+"\n\n"+"\n\n".join(observations)+"\n\n"+nxt+"\n"
    (final/"EXECUTIVE_SUMMARY.md").write_text("# 实验摘要\n\n"+summary)
    (final/"FINAL_RESEARCH_REPORT.md").write_text("# CCS-SLiM 研究报告\n\n"+summary+"\n"+comparison+"\n"+
        "\n".join(details)+"\n## 相同预算下相对 P1 的开发差值\n\n"+json.dumps(improvements,ensure_ascii=False,indent=2)+
        "\n\n## 正式测试相对 P1 的变化\n\n"+table(formal_deltas,["model","dataset","metric","reference","delta_points"])+
        "\n\n## 模块取舍\n\n"+table(retention,["module","baseline","lt_scene_bootstrap_95","decision"])+
        "\n\n## 运行命令与阶段记录\n\n"+table(stage_rows,["stage","status","exit_code","started","ended","command"])+
        "\n\n## 工程问题\n\n"+issue_text+
        "\n\n## 同卡效率\n\n"+table(eff,["model","side","params","partial_flops","latency_ms","p95_ms","allocated_gib","reserved_gib","process_gib"])+
        "\n\n## 解释范围\n\nTexturePoor 使用两视图协议。计时来自当前共享设备，FLOPs 仅含 profiler 支持的算子。"
        "每对记录位于各模型 evaluation 阶段输出的 *_pairs.jsonl，所有缺失项保留为空。\n")
    figures(fig,scaling)
    return final


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--mode",required=True)
    p.add_argument("--run-id",required=True)
    a=p.parse_args()
    print(summarize(a.mode,a.run_id))
