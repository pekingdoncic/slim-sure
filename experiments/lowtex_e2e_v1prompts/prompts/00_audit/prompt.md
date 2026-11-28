# Phase 00：仓库和历史实验审计

先读 COMMON_RULES.md、STATE_SCHEMA.md；本阶段不得启动正式训练。

1. 执行 `pwd`、`git status --short`、`git rev-parse HEAD`、`python --version`、`which python`（Windows 用等价命令）、`nvidia-smi`；检查 torch/CUDA/Mamba 导入、版本、GPU 实际显存，确认 Conda、WSL、MegaDepth 数据索引和 SLiM checkpoint。记录 stdout/exit code。禁止自动安装大依赖。
2. 查找原始 train.py、test.py、src/slim.py（以实际路径为准）、Conv-Mamba/SS2D/FPN/深层聚合、粗细相关、RRU、GT、Loss、Dataset、场景均衡 sampler、texture_budget.py、Phase T、Stage 2、Phase 3–5、summary.json、decisions.json、冻结协议。记录真实调用链、张量形状、粗/细/1/32 特征接口、坐标约定、训练和推理路径差异、细阶段阈值、全局相关是否冗余。核实 Phase 5 路线 B 的输入、训练与 NIG 退化证据，不猜测。
3. 历史参考（不是新结果，须核原报告）：100 对 R832，Stage 2 A1 53.8336%、B0 61.4543%、+7.621pp，非低纹理成功源格净减 4082；双侧入池可评 7728，严格粗匹配未成功 4152；仍漏失的正确目标约 76.66% 位于 P 行前五；B 排名无稳定增益、C 邻域支持小且 CI 跨零。Phase 5 一步精修有速度优势但定位/位姿增益未稳定。仅在核实后使用，记录分母及统计口径，不将不同阶段数据混合。若历史报告不可访问标记未确认。
4. 检查本地 `data/Texture-PoorSfMDataset` 是否存在及层级；不得重新下载或改变数据。列出所有开发/正式评测数据可用性、路径与潜在泄漏。
5. 仅创建当前阶段实际需要的 EXP_ROOT 目录骨架（不提前创建后续模块/配置/测试）、source_audit.md、requirements_notes.md、初始协议草案（数据对名单在 Phase 01 冻结前补齐），明确公平基线、待验证假设、两路验证、预算及阶段继续判据。不能把候选入池不足当作唯一原因。

验收：审计报告给出代码路径/接口/历史证据和真实环境检查；环境缺失明确列出。交接 `reports/phase_00.md`、状态及恢复命令；能进行数据适配则进入 `01_texturepoor/prompt.md`，缺关键数据则 BLOCKED。

补充环境审计：检查原主干 BN/SyncBN、训练候选父窗口与子匹配数量、Mamba checkpointing 兼容性、PyTorch allocator 版本/平台支持；记录差分 LR、热身切换与 resume 可复用接口。仅审计，不提前实现后续批次。

附加全量审计：按公共规则核实官方完整训练池、原采样覆盖行为、epoch 定义、全部 pair 及场景计数、完整覆盖所需训练预算，输出 train_pool manifest/hash 与覆盖计划草案。确认本地算力和数据路径，不自行租 GPU。
