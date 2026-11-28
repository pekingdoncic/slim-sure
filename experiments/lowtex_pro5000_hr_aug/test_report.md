# 测试状态

**NOT_RUN / PENDING_ON_SERVER**。本地仅阅读源码、历史报告及 Git 状态并写入独立实验。真实数值、语法编译、Shell 语法、CPU/CUDA 测试、显存探测、训练与数据集评测均待服务器执行。

| 服务器入口 | 编写的覆盖项 | 当前状态 |
|---|---|---|
| scripts/run_tests.sh | compileall、全部 Shell 的 bash -n、无可见 CUDA 的 pytest、真实 CUDA pytest | NOT_RUN |
| tests/test_geometry.py | R1024 和 1200→1216、半像素、K、深度遮挡、投影、失败 pose 分母 | NOT_RUN |
| tests/test_sparse.py | 原粗 focal 前向/梯度，稀疏点积与稠密 masked 参考、跨 chunk 反传、等分 0.25、去重、空行、MNN、双向块/halo/多正样本、GT 注入禁用及截断 | NOT_RUN |
| tests/test_features.py | CCFI 门控、空掩码与梯度，Top-M、64 维投影、图邻域、分块、零残差及背景边、可靠难负 | NOT_RUN |
| tests/test_runtime.py | 冻结划分、数据重放、Full 预算、热身切换、模型/优化器恢复、Shell 参数、跳过与重跑、路径与缺失指标 | NOT_RUN |
| tests/test_cuda_network.py | 严格加载官方权重、原版与关闭新模块等价、同权 P3/P4 零残差、真实 Conv-Mamba/CCFI/LHG 多步反传、BN、重算、权重恢复 | PENDING_ON_SERVER |
| training/vram_probe.py | 独立进程、真实 MegaDepth batch、热身/解冻、完整累积与更新、验证切换及吞吐选择 | PENDING_ON_SERVER |
| scripts/run_full_pipeline.sh --mode smoke | 官方数据、两种高分辨率 P0、每个微调组每段 20 个真实 optimizer updates、开发验证、状态/报告落盘 | PENDING_ON_SERVER |

服务器按 run_id 保存真实回执。Smoke 通过的条件不涉及路由召回或 AUC 提升。CPU 数学参考不替代真实 SLiM。返回非零时，可从 latest_error.json、debug_bundle 和原始阶段日志定位首个失败。
