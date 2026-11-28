# Phase 02：原始 SLiM 公平微调基线和训练闭环

前置：Phase 01 冻结的双验证协议与可读取的专项数据。先核对 hash。

## 实现
- `models/slim_wrapper.py` 复用原模型，关闭所有新模块必须与原版在相同输入/权重/配置下粗索引、匹配坐标、置信度、数量等价（预先确定容差）。检查训练 loss/梯度与原版一致，避免改变推理路径。若存在冗余全局相关，记录但不默默优化基线；另设公平效率对照。
- `training/train.py`、trainer/optimizer 复用原训练、Dataset、GT、Loss、sampler，支持 head_only/partial_unfreeze/end_to_end、AMP、梯度累积、checkpoint 保存恢复、日志、显存。官方 checkpoint 推理 E0 与继续微调 Exp-0 分开；后续 Exp-1A/B 使用等预算 Exp-0。
- 双验证接入：训练前、固定间隔（建议 500 optimizer steps，smoke 至少前后）、结束；MegaDepth 验证和 TexturePoor val。固定 checkpoint 选择（低纹理主要指标+通用退化约束）并保存 last/best_lowtex/best_general；正式模型选择必须用完整验证集。
- 实现 `configs/base.yaml`、`r4080.yaml`、`large_vram.yaml`、`exp0_baseline.yaml`，按公共规则实测。统一 Shell `scripts/run_all.sh`：默认 preflight；stage: preflight, prepare_texturepoor, baseline, smoke, exp0, exp1, exp2, evaluate；支持 --profile --mode --data-root --texturepoor-root --weights --checkpoint --gpu --seed --resume --output-dir --exp --dataset --split --resolution；未实现阶段必须明确报错。默认 TexturePoor 路径从仓库根解析。set -euo pipefail，自动定位根目录、检查环境/文件、支持 CUDA_VISIBLE_DEVICES、--help、独立 run_id、日志、失败退出、实际生效配置，不硬编码个人路径。输出目录限制在 EXP_ROOT 内。正式 test 由首次授权覆盖，但必须限制在 Phase 08，校验冻结模型记录和状态。run_all 默认 preflight 仍保留，工作流明确传 --mode full。

## 实际执行
1. `python -m compileall ...`、`bash -n`、pytest/实际测试入口、run_all --help/preflight/prepare_texturepoor。
2. 原始 SLiM 在 TexturePoor smoke val 真实推理→RANSAC→AUC→逐对/场景/可视化；环境可用时完成完整 val 基线，保存 `outputs/baseline_texturepoor/<run_id>/`。同时运行 MegaDepth 开发验证；不得执行正式 test。
3. Exp-0 R640 batch1 真实 20 optimizer steps（验证前后、梯度、参数更新、显存、save/reload/resume）；可行再测试 R832。Smoke 通过后自动执行 Exp-0 全量 full 微调；在训练前冻结共同预算、完整池覆盖计划及选模规则。必须保存 full checkpoint 和完整验证成绩，不能仅交付 20 步。

验收：wrapper 等价、真实 GPU 双验证和 train smoke、checkpoint 恢复、脚本真实调用通过；否则如实 BLOCKED/FAIL。交接 `reports/phase_02.md`，下一阶段 `03_texture_loss/prompt.md`。

## 4080 强制配置与 CLI 验收（Batch 1）
`r4080.yaml` 必须含：
```yaml
freeze_backbone_bn_stats: true
gradient_checkpointing: true
max_fine_train_matches: 1024
```
按公共规则实现真实 BN stats 冻结（不等于冻结 affine）、安全 checkpointing、训练候选 cap 和公平基线；测试 cap 关闭时的原版等价及开启时的 GT/索引一致，验证/推理不截断。旧版配置与受控微调配置分别记录。训练恢复时检查 BN buffers、优化器/调度器/RNG。
`run_all.sh` 在启动 Python 前（保留用户设置）执行：
```bash
if [[ -z "${PYTORCH_CUDA_ALLOC_CONF:-}" && -z "${PYTORCH_ALLOC_CONF:-}" ]]; then
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi
```
preflight 实测 allocator 支持并记录环境，给出显式关闭办法；不保证消除 OOM。实现适用于新增细头的两阶段热身及切换/续训单测（前 15% head_only、后 85% end_to_end 可配置）；当前 Exp-0 可按公平对照协议选择，不无条件套用。后续 stage 仅作 CLI 声明，未实现应报错退出，不提前创建 Exp-1C/1D 或 Exp-2 代码。

实现 full 数据覆盖计数与断点恢复；预注册匹配初始化/热身的公平对照。根据 STATE_SCHEMA.md 验证可选本地工作流监督入口；保存单阶段 full、全工作流启动和 resume 命令。EVALUATION_PROTOCOL.md 的计时器从此阶段实际接入。
