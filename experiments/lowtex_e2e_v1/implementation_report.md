Phase 00 已实现逐行官方索引与本地路径核对工具，详见 reports/phase_00.md。

Phase 01 完成专项两视图适配器、双开发清单与冻结协议。

## Phase 02

`models/slim_wrapper.py` 复用原生 SLiM 的推理、监督和损失，为受控训练加入骨干 BN running statistics 冻结、非重入 activation checkpointing、仅训练期的父窗与细候选上限。Exp-0 端到端学习，R832、BF16、四次累积。新增头两阶段策略在同一优化器及调度器中切换。冻结的 13 个模型、数据、训练和评价 Python 源文件在当前 full run 的 run_manifest.json 中逐文件存 SHA-256。

`training/sampling.py` 按官方合法行以及各 bin 预算生成可复现顺序，`training/pair_identity.py` 建立无向物理身份索引。checkpoint 保存 optimizer、scheduler、Python/NumPy/CPU/CUDA RNG、BN buffers、分箱访问数、无向位图和提交游标。BF16 使用无 scaler 配置。完整性验收同时校验更新次数、有向记录和无向唯一对。

`evaluation/development.py` 为同一模型执行两路开发验证。开发期计时含插桩。原生和统一匹配预算的最终精度、效率与高分辨率数据留给冻结后的 Phase 08。监督入口 `scripts/run_workflow.sh` 按 `reports/state.json` 推进。当前 full 及每个完整开发验证节点的模型权重分别保留，供依据冻结的通用退化约束离线核验。初始节点归档已与官方 checkpoint 逐 Tensor 核对。

当前正式 run 目录内的 frozen_sources.tar.gz 保存 20 个已经使用的模型、训练、验证源文件、配置、预算和协议，已回读比对。归档 SHA-256 为 d864784c29538400df2262fec9bda1c21837b6fa260c6d98187722ee50e7c3ee。

2026-11-28 快速筛选 v1 复用原有评测和几何，在 screening/ 内实现有向行无放回采样、真实双对骨干与逐对监督、纹理分母解耦、匹配热身的 RRU 及单/多尺度一步头。预算与判断见 protocols/fast_screening_v1.md 和其选型附录；旧 R832 断点保留。当前主线从 scripts/run_workflow.sh 进入。
