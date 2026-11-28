# 正式准确率与轻量化评测协议（v3）

Phase 01 冻结名单、指标、失败处理、模型选择规则；Phase 02 建立真实基线与计时器；Phase 03–07 开发验证；Phase 08 模型及权重冻结后自动正式测试。统一配置记录 GPU、精度、预处理、分辨率、匹配预算、RANSAC 参数及种子，测试名单不用于调参。

## 准确率主表
- Texture-Poor SfM 两视图适配：3 个开发验证场景，14 个最终测试场景；MegaDepth-1500 和 ScanNet-1500：标准相对位姿 AUC@5°/10°/20°（角度，百分数）。使用旋转误差和平移方向误差的最大值，按官方/冻结的 LoFTR 风格 AUC 实现；确认平移符号约定，失败计入分母（不得丢弃）；同一几何估计器与阈值对照，记录 per-pair、per-scene、全局和场景等权。TexturePoor 是自适配的两视图协议，不直接和原多视图 SfM 表格比较。
- HPatches：单应性估计平均四角重投影误差 AUC@3/5/10 px，按固定官方图像对及 resize 坐标协议执行；不是角度 AUC。InLoc 可选下游定位，缺数据 BLOCKED。
- 低纹理区域：有效 GT 固定分母 LT Recall、Precision、EPE、覆盖率、入池条件召回、GT Rank/Recall@K，低纹理与 other 分组；仅在可靠点级 GT 上算 EPE/Recall，TexturePoor 若无该 GT 仅报告位姿/极线等指标。不同纹理难度分组、EPE 联合覆盖率、匹配数、几何失败率。统一坐标尺度和纹理标签协议。
- 对比官方 SLiM、同预算微调 Exp-0、可用的 LoFTR/EfficientLoFTR、JamMa、SURE、SP+LightGlue、XFeat、LoMa、DKM、RoMa/RoMa v2。缺模型/权重如实 BLOCKED；同表不混用异协议的论文数字。对比训练数据/分辨率/匹配预算清楚披露。官方推荐设置与统一控制设置可各列一张表，轻量半稠密与大型稠密分别比较，不能声称全领域 SOTA 而没有相应证据。

## 推理效率（强制）
- 同一实际 GPU（优先 RTX 4080，所有对照同卡；如另有大显存卡单独表）、batch=1、eval()+inference_mode、同 AMP/输入尺寸、固定代表性至少100对（不足报告全部）、每模型/分辨率 warmup 30 次、固定重复次数（建议3轮）、CUDA 同步，避免将首次编译/权重加载混入稳态。记录设备、驱动、PyTorch/CUDA、功耗/温度或外部负载如可用。重复测量 Mean/Median/P95 ms/pair，报告计时样本数与方差；不能比较不同硬件的实测延迟。
- matcher 时间：图像在设备上到最终匹配输出，包含 backbone/检测描述、CIM、路由、粗细关联、精修、置信度/协方差等；不包含文件 I/O、图像预处理、H2D、RANSAC。另报 preprocess、H2D、RANSAC/pose、完整 pipeline wall-clock（明确 I/O 是否包含）；端到端总时间直接测，不用不同测量边界的数字相加冒充。模块时间另用同步事件测量，避免计时插桩改变总延迟；模块间重叠注明。
- 显存：隔离进程或统一清理策略，warmup 后重置峰值，记录 max_memory_allocated 和 max_memory_reserved（GiB），说明模型权重、输入、workspace 是否计入及空载基线；训练峰值另表。参数量区分总/可训练；FLOPs 明确乘加口径、输入形状、动态候选规模，Mamba/自定义算子不支持统计时列 unknown 并报告覆盖范围，不用不完整数字做强结论。
- 报告原生输出数与统一匹配预算（事前固定上限及各方法排序规则）的两套准确率/效率结果；不强行用 GT 选择匹配。统一预算对后处理的影响与网络计算分别记录。报告实际边数、路由候选数、输出匹配数。

## 分辨率扩展（强制）
- 固定相同输入图像对/协议，测试最长边/方形 padding 480、640、832、1024、1216；模型 stride 不兼容时用事先规定的最近合法尺寸并报告实际 H×W；1600 可选。不同分辨率内参/掩码/尺度正确，避免低纹理阈值变化改变评价含义。固定同一 checkpoint 不重新训练，逐分辨率记录 Pose AUC@5/10/20、LT Recall（有 GT）、Latency、Peak VRAM、实际候选量。OOM 明确记录，不删除失败点或用低分辨率冒充；另一张 GPU 的结果单列。训练分辨率扩展是独立实验。
- 输出准确率—分辨率、延迟—分辨率、显存—分辨率、LT Recall—分辨率曲线；输出同设备/协议下准确率—延迟与准确率—显存 Pareto 图。真实结果绘图，不预填理想趋势。

## 模块消融与统计
- 原版 RRU-4/1、Exp-0、纹理 Loss、单尺度一步头、Crop-Fuse、联合、协方差、CIM、Router、Halo、条件性 Anchor、最终模型。分别报 backbone/coarse/fine/refine/total；如模块不适用标 N/A，未运行 NOT_RUN，环境问题 BLOCKED。训练专属 Loss 不应增加推理网络参数，但仍实测延迟。普通几何后端与协方差加权后端单列。
- 固定开发验证的选模规则、主指标和容许退化/时间预算；关键对比多种子、场景级配对 bootstrap CI、效应大小；同对照预算。输出逐对 CSV/JSONL、逐场景统计、完整模型训练池覆盖、论文表 CSV/Markdown 与 PNG/SVG 图。不得将 smoke 成绩列为正式结果。无收益也输出真实结果。
