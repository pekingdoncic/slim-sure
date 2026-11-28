# Phase 01：本地 Texture-Poor SfM 两视图 Benchmark

前置：Phase 00 审计完成。必须实际适配本地数据，不是预留接口。参考 https://github.com/zju3dv/DetectorFreeSfM 、https://arxiv.org/html/2306.15669v1 （§3.3、Appendix D），尤其官方数据评测脚本与相机读取。若外部源码不可访问，说明并使用已核实的本地资料，不猜测格式。

## 数据与划分
- 检查 `data/Texture-PoorSfMDataset` 的实际解压层级、场景、图像数/尺寸、内外参、相机模型、位姿约定、图像组、可见性、深度和点级 GT 是否存在。输出 `manifests/texturepoor_dataset_audit.json`。原数据只读。
- 核实官方验证场景 1012、1026、1029；测试场景 1000、1002、1003、1005、1006、1007、1008、1010、1014、1021、1024、1025、1027、1030。如果本地不同，记录缺失/冲突，不能擅自混用。
- 以原始多视图组生成两视图对，去除自配对与 A-B/B-A 重复；按场景平衡、固定随机种子，排除极小基线/近重复和不合理共视。若有真实可见性/overlap 则分层，否则只用原始组和几何代理并记录局限，不伪造 overlap。不能根据匹配器结果选择样本。记录每对 scene/pair id、image0/1、内参、相对位姿、来源组、几何难度。输出 `texturepoor_val_pairs.json`、`texturepoor_test_pairs.json` 和固定 smoke val 子集，全部位于 manifests/，冻结 hash。生成测试 manifest 不等于测试运行。

## 评测实现
在 `evaluation/texturepoor_sfm/` 实现 dataset、build_pairs、geometry、metrics、evaluate、visualize；优先复用 SLiM 的几何估计/指标。确认 c2w 或 w2c、crop/resize/padding、内参转换、原始像素与模型坐标、畸变（如有）、旋转误差、平移方向误差及极小平移问题。合成相机投影验证相对位姿；测试不同 resize/padding、去重、场景隔离、失败计入 AUC 分母、固定随机种子。

报告 Pose AUC@5/10/20、旋转/平移误差、成功率、GT 极线精确率、RANSAC 内点数/率、匹配数、覆盖率、延迟、显存；逐对、逐场景与场景等权/全局统计。无点级 GT 不输出真实 EPE/Recall/PCK。固定可视化样本并包含失败案例；协方差椭圆后续启用。

同时核实 MegaDepth 开发验证和正式测试隔离，冻结两路验证子集、正式清单及模型选择约束。依据 EVALUATION_PROTOCOL.md 写评测协议及 SHA256（成绩出现前）；Phase 02 在正式训练前补充冻结训练预算和覆盖计划的版本化附录。不要运行正式 test。脚本 `scripts/prepare_texturepoor.sh` 和 `scripts/eval_texturepoor.sh` 至少支持数据审计、构建、验证入口。

验收：实际 CLI 执行审计、生成清单、真实验证图像读取、几何及指标单元测试，记录数量和 hash；GPU 原版模型推理在下一阶段执行。若数据/几何不可用则 BLOCKED，不能虚构。交接 `reports/phase_01.md`，下一阶段 `02_baseline/prompt.md`。
