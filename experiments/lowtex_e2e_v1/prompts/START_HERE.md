# Codex 总控入口：SLiM 低纹理匹配（协议 v3：全量训练、一次授权）

在 SLiM 仓库根目录实际执行，不只给方案。使用标准 MegaDepth 完整官方训练池，建立低纹理专项与通用评测，并验证准确率及轻量化。

## 唯一启动指令及授权
用户发出本包 README 中的启动指令，即明确授权 Batch 1–4、Phase 00–08、各主实验 full 预算、真实 CLI smoke、模型冻结后的最终测试，以及本协议限定的条件性 Anchor / 几何后端扩展。不需每阶段询问用户。授权不包括下载大型新数据、安装/升级系统依赖、修改原仓库、租用付费算力、改变原始训练索引或 TA-MegaDepth/纹理增广专项重训；这些默认不做并报告。不要自动 git commit/push。若用户另行限定范围，以其明确限制为准。

## 连续执行规则
1. 定位本文件所在目录 `experiments/lowtex_e2e_v1/prompts/`；读 COMMON_RULES.md、STATE_SCHEMA.md、EVALUATION_PROTOCOL.md。所有新产物在 EXP_ROOT 内。读取 reports/state.json、handoff.md；首次按模板创建。旧 v1/v2 状态先备份、迁移；旧 smoke/short 不等于 full，保留证据并补足正式训练，禁止重置历史。
2. 首次启动写入授权记录（all four batches, full, final test after freeze）；依次读取当前阶段 prompt.md。每阶段重读公共规则及交接，只有当前阶段需要的文件才能创建。不要预读/实现后续阶段，不允许空壳。批次是任务分组而非授权停点。
3. 每阶段先单测、真实 smoke，再对实际主实验执行完整数据正式训练、完整开发验证；保存 checkpoint、日志、阶段报告及状态，完成硬依赖后自动读取下一阶段。不要只完成工程 smoke 就认为正式训练结束。科学负结果不阻塞独立实验；依赖性组合按冻结判据自动回退/跳过，并记录理由。所有主实验必须尝试，只有硬阻塞或不满足预注册依赖的分支可不运行。不得根据 test 调参。
4. Phase 00–07 不运行最终测试；Phase 08 在模型、超参数、选择规则、协议和 checkpoint hash 冻结后自动执行最终测试及轻量化全套实验，不另请授权。缺数据/权重/算力如实 BLOCKED；继续不依赖该阻塞项的任务。真正无法推进才停止，提供恢复命令。没有收益也要完整交付负结果，不强行宣称 SOTA。
5. 默认 full：完整官方合法训练图像对池，正式训练每组统一训练预算和覆盖协议；Phase 00/02 核实官方配置并冻结。默认参考 SLiM 论文 19 epochs 的训练安排，若实际官方配置不同采用核实值；不得静默缩短成固定 1000/5000 steps。详见 COMMON_RULES.md。Smoke/短诊断独立记录，不替代 full。
6. 长训练使用真实可恢复的命令、持久日志和 checkpoint；可在当前终端运行受控本地任务并监控真实退出码，不假定 CLI 会话结束后 Codex 仍能推理或执行下一阶段。支持同一启动指令幂等恢复；若当前 Codex CLI 具备非交互执行能力，可在 Phase 02 提供可选本地监督脚本（见状态规范），无需每阶段粘贴提示词。不要声称未实现的跨会话自动续跑已经可用。

## 阶段顺序
| Batch | Phase | 文件 | 任务 |
|---|---|---|---|
| 1 | 00 | `00_audit/prompt.md` | 仓库、数据、历史实验、环境审计 |
| 1 | 01 | `01_texturepoor/prompt.md` | 专项 benchmark、双验证协议 |
| 1 | 02 | `02_baseline/prompt.md` | 公平微调基线、完整训练和统一入口 |
| 1 | 03 | `03_texture_loss/prompt.md` | Exp-1A 纹理解耦监督 |
| 1 | 04 | `04_crop_fuse/prompt.md` | Exp-1B 多尺度精修、独立筛选对照 |
| 2 | 05 | `05_covariance/prompt.md` | Exp-1C/1D 联合训练及协方差 |
| 3 | 06 | `06_coarse/prompt.md` | Exp-2A/2B CIM、块级路由 |
| 3 | 07 | `07_sparse_integration/prompt.md` | Exp-2C/2E 稀疏相关与完整网络，可选 2D |
| 4 | 08 | `08_final_eval/prompt.md` | 正式测试、效率、论文实验交付 |

开始：核对状态并运行第一个未完成阶段。每阶段更新进度，无需人工授权；全部完成或真实不可恢复阻塞才结束。
