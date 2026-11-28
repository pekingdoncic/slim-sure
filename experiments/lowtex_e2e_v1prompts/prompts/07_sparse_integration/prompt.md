# Phase 07：Exp-2C Halo 稀疏 Dual-Softmax 与完整网络

前置：首次统一授权覆盖 Batch 3；Phase 06 路由实现与监督通过。科学准入门槛未验证时仅做算子工程测试/真实 smoke，不开展正式联合训练；未达门槛则跳过该模型，保留已验证 CIM/原版粗匹配路线。

## 稀疏关联
- `models/sparse_correlation.py`：将路由块与可配置 Halo 映射到 1/8 候选边集合 E，去重；仅计算实际边分数。严禁先构造完整 1/8 相关矩阵再 mask 冒充稀疏。记录实际边数、候选预算、峰值工作区；优先 PyTorch 原生 gather/scatter，避免过大 E×C 临时张量，chunk 或有界 block 实现；检查当前版本接口。
- 在全稀疏边集做行/列 logsumexp：`p_ij = exp(s_ij-row_lse_i) * exp(s_ij-col_lse_j)`（乘法，绝非减法）；等价稳定形式 `exp(2*s_ij-row_lse_i-col_lse_j)`；不在各块独立归一化后拼接。FP32 reduction，正确处理空行空列、重复边、padding、MNN、tie、候选规模导致的置信度变化。保留无 GT 的推理路径；训练 GT 并集/候选预算透明。
- 小 Tensor 稠密 masked reference 对比分数、行列 softmax、概率、MNN、去重、前向及梯度；测试空边/边界与 BF16 稳定性。增加确定性回归测试：完整 2×2 等 logits 时每条边的 Dual-Softmax 必为 0.25（错误减法会得 0）；单边时为 1；非均匀稀疏边与稠密 masked reference 对比，概率必须有限且在 [0,1]，不要求乘积的行列和等于 1。1/32 全局块相关仍平方复杂度，禁止宣称整体严格线性。若基线存在冗余全局相关，单独对照移除，不把其收益归给路由。
- Exp-2C 对照 Router+Halo 与 Router+CIM+Halo（CIM 有收益才保留），同粗/细监督及预算，比较正确 GT 覆盖、竞争失败、LT 与非 LT recall、Pose AUC、时间/显存、不同分辨率。配置 `exp2c_sparse_matching.yaml`。

## 可选分支和整合
- Exp-2D Anchor-to-LowTex：仅当存在经验证的剩余消歧瓶颈且预注册的实施条件满足时实现。按可靠匹配/可匹配性选 Anchor，局部/块级结构条件化特征或匹配先验传播，门控/残差，避免跨深度边界，禁止直接复制位移或全局高分辨率 attention。独立消融，默认关闭。
- Exp-2E：组合已经有合理验证依据的粗/细模块和监督，保留相同配置的原版 SLiM 基线。完整端到端可从官方 checkpoint 初始化，不等于从零训练；离散路由仍有独立可微监督。R640→832 先实测，1024/1216、大显存与 checkpointing 按实际资源和冻结计划使用，不自行租卡；升分辨率是独立实验，不混入结构消融。
- 在线保深度纹理抹平增广、TA-MegaDepth 默认不实施；有证据再独立预注册，且必须重新训练；不得将数据变化与模块收益混淆。

## CLI
单元/数值/梯度等价测试真实执行；对实际启用方案做真实 20 optimizer updates、双验证、checkpoint 和训练/推理显存；在显存不足时记录 BLOCKED/OOM。门槛通过后自动执行 Exp-2C 及最终 Exp-2E 的完整训练池 full 正式训练；若路由未通过，则以选定的非路由有效结构完成 full 对照；短训不能声称 SOTA。

验收：稀疏等价与真实运行通过，注明科学准入和最终保留模块，未通过时给出可用回退网络。交接 `reports/phase_07.md`，Batch 3 完成后自动进入 `08_final_eval/prompt.md`，无需额外授权。
