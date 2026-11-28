# Phase 08：论文评测、可选扩展与最终交付

前置：首次统一授权覆盖 Batch 4；存在实际可训练的基线及至少一个已验证工程实现的候选；若科学结果不足，交付真实负结果/阶段计划，不强行宣称论文完成。

## 正式实验与防泄漏
- 首先汇总完整开发验证结果、预注册选模规则、匹配预算、训练预算、数据/协议/代码/权重 hash，冻结模型与选择记录。模型及协议冻结后自动运行最终 test，首次授权已覆盖。不得根据 14 个 TexturePoor 测试场景或通用最终测试调参。
- 正式低纹理：TexturePoor 14 test scenes 两视图固定 manifest；通用：MegaDepth-1500、ScanNet-1500、HPatches（实际任务分别位姿/单应性）；有可靠 GT 的开发或独立区域级数据单列 LT Recall/EPE 等。TexturePoor 场景为物体中心，不足以单独证明室外荒漠泛化。可选 InLoc 下游定位、HR-LoTex（若尚未定义来源和 GT 则 NOT_RUN）。
- 同评测协议重新跑本地可用的 SLiM、JamMa、SURE、LoFTR/EfficientLoFTR、RoMa 等；后续可扩展 LoMa、XFeat、DKM、RoMa v2 等。缺模型/权重明确 BLOCKED，不把他处论文数字和本 benchmark 混在同一主表；不同训练数据和分辨率清楚标注。报告场景等权与全局、失败样本、固定可视化、匹配预算和效率。通用评测的官方协议与统一协议若不同应分别标注。
- 正式消融：Exp-0、1A、1B、1C、1D、2A、2B、2C、可选2D、最终2E；只报告实际完成。建议多种子/场景级配对 bootstrap CI（图像对并非完全独立），报告效应大小与不确定性。研究目标为低纹理提升、通用不过度退化、轻量优势，不承诺 SOTA。

## 条件性扩展（几何后端可按证据自动执行；数据重构默认不做）
- 协方差加权几何后端：先普通 LO-RANSAC + covariance-weighted pose refinement，再考虑加权 RANSAC scoring；明确 A/B 两侧测量误差及坐标系变换，不能从单侧 Sigma 假定双侧真值。前端普通后端与加权后端分别报告。仅在协方差校准可靠且预注册条件满足时自动接入，不以最终测试成绩决定。
- 若数据审计/失效证据支持，预注册 TA-MegaDepth：场景×共视×有效低纹理 GT，保持训练/验证/测试隔离，比较原数据/新数据 × 原模型/新模型的 2×2，全部相同预算重新训练。在线纹理抹平增广亦独立对照、保留深度及结构边界，不能当作无需重训。无证据不修改 MegaDepth。

## 工程最终验收
- 运行全量已实现单测、compileall、所有 shell bash -n、--help 和实际可用 stages；检查原始仓库未被本任务修改、所有新增文件和日志在 EXP_ROOT。核对协议 hash、manifest、checkpoint resume 和真实 CLI 记录。
- `README.md` 给出准确的 WSL/Conda 运行命令、数据及权重配置、preflight、prepare_texturepoor、baseline、smoke、单个 exp、resume、evaluate val 和冻结记录通过后的 evaluate test；未实现模块不能暴露为成功。`test_report.md` 列命令、exit code、PASS/FAIL/BLOCKED 和实际日志；`implementation_report.md` 列完成/未完成、原代码复用关系、模型张量接口；保留历史审计与冻结协议。
- 输出实际对比表（TexturePoor val/test AUC@5/10/20、通用 AUC、LT Recall、EPE、覆盖、耗时、显存）；缺项 NOT_RUN/BLOCKED。汇报读取场景/图像数、val/test pairs 数、基线、训练 step、真实 GPU 峰值、哪些实验确实通过、哪些仍阻塞、后续运行命令。不得把工程 PASS 说成效果 PASS。

验收：所有报告/脚本/配置与实际证据齐全。最终测试按首次授权自动执行；若因硬阻塞未完成，记录 final_test BLOCKED/NOT_RUN 及恢复命令，不伪造。交接 `reports/phase_08.md`，总控停止，不自动执行额外长训。

最终工程验收额外复核：阶段顺序及首次统一授权、BN buffers、checkpoint 梯度、训练 cap/推理无 cap、两阶段切换/续训、allocator 实际配置、协方差掩码与数值约束、Dual-Softmax 乘法回归测试。未实现/未运行项目不得标 PASS。

强制执行 `prompts/EVALUATION_PROTOCOL.md` 中主表、HPatches、同硬件计时、显存、分辨率扩展、模块级消融及 Pareto 曲线。最终交付每组 full 训练池覆盖率与实际预算；未达全量者单独标记。
