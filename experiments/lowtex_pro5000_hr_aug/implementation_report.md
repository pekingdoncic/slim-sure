# 实现与交付清单

本次新增 62 个文件。状态为 **代码已写入，数值与运行验收 PENDING_ON_SERVER**。以下路径相对于本实验目录。

## 核心实现

| 模型 | 入口与数据流 | 可学习路径 |
|---|---|---|
| P0 | CCSMatcher.core 严格加载官方 state_dict，原生 Norm-Clue → Fine → RRU×4 | 零样本 |
| P1 | CCSMatcher.features 一次处理真正的微批；逐对生成监督、流式计算原 coarse focal，调用原细匹配/回归 | 同一原始网络微调 |
| P2 | CCFI.forward 从 F8 构造 F32，Q/K/V、熵和 margin，经门控将两侧上下文回写 F8 | 特征投影、门控、Backbone；原粗细目标 |
| P3 | BlockRouter → _EdgeDot → dual_logp → select_mnn，预测时只读图像特征和有效掩码 | 稀疏 coarse focal 和可微多正块路由损失。离散 Top-K 只产生索引 |
| P4 | HypothesisGraph 在完整 BBR 边集中选 Top-M，逐格有限邻域消息传递、零初始化 residual，用全部稀疏边重新竞争 | LHG、CCFI、原主干由匹配损失与可靠难负排序监督更新 |

单图特征为 [B,192,R/8,R/8]，R1024 与 R1216 的 1/32 块网格分别为 32² 和 38²。块分数字段为 [B,N32,N32]，每对稀疏相关/概率为 [E]。LHG 只为每源格最多 M 条边构造节点。不同 batch 无共享归一化。真正的 1/8 密集参考仅用于原生对照语义与小尺寸服务器测试。局部候选图的节点特征、块路由预算和稀疏点积的反向重计算见下列文件。

## 复用的仓库代码

- configs/cfg_outdoor_test.py 提供冻结原模型形状、原级联匹配阈值和 Loss 权重。src/slim.py 负责官方权重转换、原范数筛选、fine correlation、训练/推理细坐标以及 refinement。src/recurrent_refinement.py 的 RecurrentRefinementUnit 保留四步原逻辑。
- src/backbone/convmamba.py 及其 Mamba/SS2D/FPN 通过 SLiM.feature_backbone 执行；src/utils/misc.py 的原生 fine_upsample 继续复用。
- src/utils/geometry.py 的 warp_kpts 用于复现原生训练粗父窗；src/utils/supervision.py 的 spvs_intermediate、compute_supervision_fine 用于原细监督。src/loss.py 的 SLiM_Loss.compute_fine_loss 和 compute_refine_loss 直接使用，原 compute_coarse_loss 作为数学等价参考。
- src/texture_budget.py 的 local_texture 和 TEXTURE_SCHEMA 用于分辨率专属训练池标定；utils/metrics.py 的 estimate_pose、relative_pose_error 和 symmetric_epipolar_distance 用于两视图几何评测。
- experiments/lowtex_e2e_v1/manifests 的固定训练池、开发/测试对及通用测试索引已制成独立快照。原始绝对路径只取可迁移标识，原始与新哈希保存在 hashes.json。训练枚举和评测均只读摄像机、图像、深度及索引，新缓存位于本实验。

## 文件清单

| 目录 | 已交付文件 |
|---|---|
| 根目录 | .gitignore、README.md、source_audit.md、protocol_pro5000.md、implementation_report.md、test_report.md |
| configs/ | base.yaml、pro5000_shared.yaml、p0_official.yaml、p1_baseline.yaml、p2_ccfi.yaml、p3_bbr.yaml、p4_lhg.yaml |
| models/ | slim_wrapper.py、ccfi.py、block_router.py、sparse_correlation.py、sparse_dual_softmax.py、hypothesis_graph.py |
| losses/ | routing_loss.py、sparse_matching_loss.py、hard_negative_ranking.py |
| data/ | pad32_adapter.py、dataset_adapter.py、supervision_adapter.py、texture_adapter.py、texturepoor_protocol.py |
| training/ | options.py、optimizer.py、runtime.py、trainer.py、vram_probe.py、preflight.py、worker.py、pipeline.py、monitor.py |
| evaluation/ | datasets.py、pose.py、lowtex_metrics.py、eval_hr.py、efficiency_benchmark.py、summarize_results.py |
| utils/ | run_io.py |
| scripts/ | run_full_pipeline.sh、run_tests.sh、preflight.sh、monitor.sh |
| tests/ | pytest.ini、test_geometry.py、test_sparse.py、test_features.py、test_runtime.py、test_cuda_network.py |
| manifests/ | hashes.json、train_pool.json、megadepth_dev_pairs.json、megadepth_smoke_val.json、texturepoor_val_pairs.json、texturepoor_smoke_val.json、texturepoor_test_pairs.json、general_test_indices.json、official_megadepth_indices.json |

实验内 .gitignore 对本目录的 data/ Python 源码作局部放行。服务器生成的运行结果与缓存单独忽略，便于只同步代码及冻结清单。

## 已知限制与服务器验收

1. 未运行 PyTorch、CUDA 工具、编译、单元测试、推理或训练。Mamba/Triton ABI、BF16 支持、实际峰值显存与高分辨率吞吐需通过服务器脚本验收。
2. 单进程单 GPU。原生 RRU 和范数候选池仍有随分辨率和真实图像内容变化的显存成本。PyTorch allocator 限额、driver 进程监控、设备空闲预留及隔离探测同时使用。共享 GPU 的外部分配或硬终止可能只允许从最近一次已落盘的完整更新恢复。
3. 所有比较使用同一高分辨率坐标、深度与纹理协议。深度按有效分辨率最近邻采样，原像素中心/网络边界坐标有显式半像素转换，因此需在服务器验证真实标定与点级 GT。跨旧 R832 成绩的直接差值没有同协议保证。
4. 图路由和排序依赖几何可见性；训练时记录 GT 补入，科学结论以无补入的开发及封存测试结果为依据。Top-M=5 和 loss 权重是预先冻结的初值。
5. FLOPs 为 profiler 支持的算子小计，未覆盖算子单列。驱动占用为实际采样峰值，极短的外部显存峰可能错过。任何跨硬件加速、AUC 涨幅或低纹理改善均待实测。
6. 一个全池 epoch 需要 553,918 个有效更新。服务器需自行准备完整数据、已有依赖、可持久化磁盘和可读取本进程的驱动监控。默认不会下载数据。
