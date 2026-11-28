# Phase 03：Exp-1A 高低纹理分母解耦 Loss

前置：Exp-0 同预算/同解冻范围基线可用或已按协议登记。保持网络、标准 MegaDepth 和采样不变，仅改变训练监督。

- 复用 texture_budget.py；如需适配 `data/texture_adapter.py`，保留历史标签语义，审计 Sobel/窗口/阈值来源、R832 到 R640 的尺度一致性。可在统一参考分辨率打标签后映射；禁止用验证/测试数据估计阈值。正确处理 padding、resize、网格、双侧纹理定义、遮挡、有效深度和共视 GT；纹理低不等于可匹配。
- `losses/texture_balanced_loss.py`：对原 loss 的逐有效正样本项按 LT/other 分组，`L=alpha*mean(L_LT)+(1-alpha)*mean(L_other)`；不随意改变原有其他 loss 项和负样本语义。空组回退到非空组/全局原规则，两组为空用可微零或明确跳过；小组计数保护、单点权重上限、FP32 reduce、NaN/Inf 检查。关闭后严格还原原 loss。
- 支持 coarse/fine/coordinate 独立开关，第一轮优先 coarse-only，坐标或 fine 后续独立消融，不能同时改全部损失。alpha 候选 .25/.4/.5，仅在验证集且按实际有效样本占比预注册；记录真实组权重和梯度累积归一化约定。确保相应主干可训练，不能完全冻结粗特征却宣称粗特征学习。
- 测试数值、空组、mask 坐标、权重上限、梯度、禁用等价。真实 MegaDepth CLI smoke 20 updates，双验证及恢复。记录各组有效 GT 数、loss、权重、空组率、梯度范数、L2 范数、固定分母 LT/other 粗召回、入池条件正确率、TexturePoor val AUC 与通用性能。
- Smoke 通过后自动使用完整训练池执行 full，与相同预算和解冻策略的 Exp-0 比较；不要由一次短训或 smoke 宣称有效。不提升则记录失败，后续 Crop-Fuse 独立开展。纹理 loss 不是数据重划分。

验收：原 loss 禁用等价、真实训练梯度和双验证通过，输出 `configs/exp1a_texture_loss.yaml`、实验报告和实际 shell 命令。交接 `reports/phase_03.md`，下一阶段 `04_crop_fuse/prompt.md`。

Exp-1A 没有新增随机初始化精修头：不要套用 head_only 热身。粗监督从第一个 optimizer step 起能更新目标主干；使用公共 BN/checkpoint/cap/allocator 规则，保持与 Exp-0 对照一致，记录 cap 前后低纹理有效 GT。
