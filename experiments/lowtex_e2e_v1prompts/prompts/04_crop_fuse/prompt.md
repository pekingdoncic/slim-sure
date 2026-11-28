# Phase 04：Exp-1B 多尺度 Crop-Fuse 一步残差精修

前置：Phase 02 基线和训练管线。Exp-1A 无收益不阻塞本独立实验。先核实 Phase 5 路线 B 真实实现及历史结论。只改变精修结构，原 loss 与粗匹配、细粒度相关和候选生成保持不变，默认纹理解耦关闭。

## 模块
- `models/crop_fuse.py` / `residual_refiner.py`：从真实 1/2、1/4、1/8 特征提取两视图候选位置附近局部窗口，轻量通道投影、空间对齐、局部相关/融合（concat/轻卷积/门控），预测一步二维残差。以仓库实际通道/坐标为准，不重跑 backbone、不全图高分辨率 attention、不默认整图上采样；chunk 控制候选窗口显存。
- 明确原图/特征像素中心、粗网格中心 vs 左上角、grid_sample align_corners、边界/掩码、残差单位与范围、固定参考点还是双侧回归（原 RRU 行为优先保留）。测试整数/分数坐标和 padding 及梯度。
- 至少支持原 RRU 4 次、RRU 1 次、单尺度一步回归、多尺度一步回归；同候选、同预算、同精度计时。冻结主干验证新增细头可以 head_only，但需要同冻结范围对照；正式对比使用相同解冻/训练预算的模型。
- 细阶段全图相对阈值作为独立实验开关（先核源码）：original、local_topk、absolute_threshold、local_topk_with_threshold。默认保持 original；禁止直接放开全部候选。比较时匹配预算相近，报告实际输出数/精确率。不得把筛选收益归到 Crop-Fuse。

## CLI 验证
实现 config `exp1b_crop_fuse.yaml`；单测形状/尺度/边界/反向传播/禁用等价，真实 MegaDepth 20 optimizer updates、save/reload、双验证、峰值显存。可先 640 再 832。Smoke 通过后自动开展 full 训练对照（原生 RRU-4/1 若同权重可直接推理，单尺度和多尺度新增细头使用完整训练池并匹配热身对照），对比四种精修配置，报告固定分母低纹理 Recall、EPE 与覆盖率、细阶段/全流程延迟、通用与专项 AUC。以完整训练及完整开发验证决定保留 RRU 或 Crop-Fuse。

验收：有效梯度和坐标正确性、真实训练与双验证，记录有效与无效分支。交接 `reports/phase_04.md`。Batch 1 在此完成验收与交接，自动读取 `05_covariance/prompt.md`，无需额外授权；未验收前不提前创建相关代码或配置。

新增细头训练：full（及 short/medium）默认前 15% optimizer updates head_only、剩余 end_to_end，差分 LR；执行公共规则中的 BN stats 冻结、训练 cap（1024/图像对）、checkpointing 和 allocator。验证冻结期/解冻期参数更新、切换前后 resume、scheduler 不重置；对照匹配热身和预算，并独立报告原权重微调基线。smoke 可用短热身覆盖切换。
