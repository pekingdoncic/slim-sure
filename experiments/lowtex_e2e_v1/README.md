# 当前主线（fast screening v1）

RTX 4080 在 R640 使用完整官方训练池进行单周期筛选。仓库根目录执行 `bash experiments/lowtex_e2e_v1/scripts/run_workflow.sh` 启动或恢复；加 `--status` 查询进度。所有新工作均在本目录，原 R832 组的正常断点与审计保存在 [切换记录](reports/fast_screening/legacy_r832/verified.json)。

[工程验收](reports/fast_screening/ENGINEERING.md) · [实时状态](reports/fast_screening/state.json) · [结果与分析](reports/fast_screening/FAST_SCREENING_REPORT.md) · [吞吐量](reports/fast_screening/TRAINING_THROUGHPUT.csv) · [比较表](reports/fast_screening/FINAL_COMPARISON.csv) · [训练覆盖](reports/fast_screening/TRAINING_COVERAGE.csv) · [版本化预算](protocols/fast_screening_v1.md) · [判据](protocols/fast_screening_v1_selection.md)。

以下为保留的 v3 历史说明。

# SLiM 两视图低纹理匹配实验

在 SLiM 仓库根目录使用 WSL Ubuntu。以下入口读取已登记的本地 GPU、现有 slim 环境、冻结协议和上次写入的 checkpoint。

```bash
bash experiments/lowtex_e2e_v1/scripts/run_workflow.sh
```

同一条命令用于首次执行和中断恢复，重复启动会报告现有监督进程。只查看状态可执行 `bash experiments/lowtex_e2e_v1/scripts/run_workflow.sh --status`。主机或 WSL 重启会终止进程，重新运行上述入口可恢复。跨阶段的本地 CLI 依赖现有登录和 WSL interop。

| 记录 | 用途 |
|---|---|
| [实验状态](reports/state.json) · [交接](reports/handoff.md) | 当前阶段、已提交的优化器步数、覆盖率与恢复位置 |
| [监督状态](reports/host_status.json) · [训练进程状态](reports/workflow_status.json) | 最近心跳与实际 run |
| [训练预算](manifests/full_budget.json) · [验证协议](protocol_frozen.md) | 所有正式对照共用的预算、开发集与选模规则 |
| [审计](source_audit.md) · [Phase 00](reports/phase_00.md) · [Phase 01](reports/phase_01.md) · [Phase 02](reports/phase_02.md) | 数据来源、历史结论、专项评测、原始权重开发结果与正式训练 |
| [测试记录](test_report.md) · [实现记录](implementation_report.md) | 实测命令、失败与修复 |
| [当前 full run](reports/full_run.json) | 训练目录与身份哈希 |

单项 CLI 由 `scripts/run_all.sh` 提供；不带参数运行 preflight。已启动的单卡控制组参数为 `exp0 --mode full --resolution 832 --seed 66`。单阶段手工恢复时，保留相同参数，并添加 `--resume --checkpoint experiments/lowtex_e2e_v1/outputs/exp0_full/20261127T173502053843Z/checkpoint_last.pt`。日常用带锁的工作流入口，避免创建新的正式 run。

协议入口为 [START_HERE](prompts/START_HERE.md)。只有通过当前阶段硬依赖才会进入下一阶段。最终测试由 Phase 08 校验冻结记录后执行。
