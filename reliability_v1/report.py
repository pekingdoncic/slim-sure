"""Concise evidence-linked reports; no unexecuted result is labelled complete."""
import json
from pathlib import Path
import subprocess
from .runtime import atomic_json,now

def read(path,default=None):
    path=Path(path)
    return json.loads(path.read_text()) if path.exists() else default

def commands(config):
    base="cd /home/bupt/projects/SLiM-master\nsource /home/bupt/miniconda3/etc/profile.d/conda.sh\nconda activate slim\n"
    entry="python -u reliability_stage.py --config configs/reliability_v1.json"
    return dict(engineering=base+entry+" --run-stage engineering",
                full_train=base+entry+" --run-stage full_train",
                resume=base+entry+" --run-stage full_train --resume",
                evaluate=base+entry+" --run-stage evaluate",
                all=base+entry+" --run-stage all")

def write_engineering(config,error=None):
    run=Path(config["run_dir"]);a=read(run/"engineering_acceptance.json",{})
    data=read(run/"data_audit_summary.json",{});resources=read(run/"engineering/resources.json",{})
    budget=read(run/"formal_budget.json",{})
    status="工程验收通过，正式训练尚未启动" if a.get("status")=="pass" else "工程验收未通过；正式训练未启动"
    branch=subprocess.check_output(["git","branch","--show-current"],cwd=config["repo_root"],text=True).strip()
    lines=["# Engineering Reliability V1", "",status,"",
        f"- 分支：`{branch}`；基线 commit：`{(run/'baseline_commit.txt').read_text().strip()}`。",
        f"- 仓库：`{config['repo_root']}`；数据根：`{config['data_root']}`。",
        f"- 训练/验证：{data.get('train_count','待审计')} / {data.get('val_count','待审计')} pair；"
        f"{len(data.get('train_scenes',[]))} / {len(data.get('val_scenes',[]))} scene。范围为 full_available，不声称官方完整训练集。",
        "- 原验证索引属于 MegaDepth-1500，已排除；合法场景按 seed 2026 划分独立验证集。",
        "- 原 SLiM 参数及 BN 固定；新增共享投影与可靠性头为 FP32，默认开关关闭，score_only 不改变点集。",
        f"- smoke 逻辑更新 {resources.get('logical_optimizer_updates','未运行')}；实际 optimizer 调用 {resources.get('physical_optimizer_updates','未运行')}；上限 50。",
        f"- 原权重严格加载；恢复包含新头、AdamW、RNG、epoch 排列和位置。验收结果以 [engineering_acceptance.json](engineering_acceptance.json) 为准。",
        "- 标签是实际粗锚点的每轴半 stride 容差；无法确认深度/可见性者 ignore。详见 [feature_label_contract.json](feature_label_contract.json)。",
        f"- 新增参数量：{read(run/'parameter_audit.json',{}).get('new_parameter_count','未测')}。",
        f"- 冷加载 {resources.get('load_seconds','未测')} 秒；训练 pair 中位耗时 {resources.get('median_pair_seconds','未测')} 秒；"
        f"GPU allocated 峰值 {resources.get('peak_allocated','未测')} 字节；主存峰值 {resources.get('rss_peak','未测')} 字节。",
        "- 50 步以内的稳定性和独立验证入口检查不构成泛化或 10 个 epoch 稳定性结论。",
        "- sparse_direct 来自已有 wrapper 的等价裁剪；其收益不归因于可靠性头。","","## 证据","",
        "- [完整数据审计](data_audit_summary.json)、[逐候选纳入/排除](candidate_audit.jsonl.gz)、[文件/相机审计数据库](data_audit.sqlite)。",
        "- [路径与链接](data_paths.json)、[测试场景排除](test_scene_exclusions.json)、[配置](resolved_config.json)、[身份](run_identity.json)。",
        "- [真实训练日志](engineering/physical_visits.jsonl)、[恢复对照](engineering/resume_evidence.json)、[smoke checkpoint 指针](engineering/smoke.json)。",
        "- [资源测量](engineering/resources.json)、[全量预算](formal_budget.json)、[阶段状态](stage_tasks.json)。"]
    if error:lines+=["","## 当前缺口","",str(error)]
    if budget:lines+=["","## 正式训练预算（估计）","",json.dumps(budget,ensure_ascii=False,indent=2)]
    lines+=["","## 实际启动命令",""]
    for name,cmd in commands(config).items():
        lines += [name,"",chr(96)*3+"bash",cmd,chr(96)*3,""]
    (run/"Engineering_Reliability_V1_Report.md").write_text("\n".join(lines)+"\n")

def write_training(config):
    run=Path(config["run_dir"]);state=read(run/"full_train/status.json",{})
    text="# Training Reliability V1\n\n"+json.dumps(state,ensure_ascii=False,indent=2)+"\n\n"
    text+="仅 status=committed 且完成 10 个完整 epoch 才表示正式训练完成。每 epoch 覆盖证据见 full_train/epoch_*_coverage.json。\n"
    (run/"Training_Reliability_V1_Report.md").write_text(text)

def write_evaluation(config,summary):
    run=Path(config["run_dir"])
    text="# Evaluation Reliability V1\n\n"+json.dumps(summary,ensure_ascii=False,indent=2)+"\n\n"
    text+="本轮为开发验证；best 由该验证集选择。一个训练 seed；几何五 seed 不是训练重复。阶段1 smoke 不作泛化证据。\n"
    (run/"Evaluation_Reliability_V1_Report.md").write_text(text)
