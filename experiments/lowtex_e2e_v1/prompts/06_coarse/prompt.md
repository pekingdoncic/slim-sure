# Phase 06：Exp-2A 深层 CIM / Exp-2B 块级路由

前置：首次统一授权覆盖 Batch 3；Phase 02 训练闭环稳定。精修分支无收益不阻塞本独立研究。基于历史入池后匹配失败现象，优先改善特征消歧而不是盲目扩大候选池。先对原始相关/范数筛选的真实训练与推理实现复核。

## Exp-2A CIM
- 复用真实 1/32 深层聚合特征（如未暴露则最小适配），低成本跨图相关信息注入 1/8 粗特征，使用接近恒等的残差初始化；避免重复 SLiM SS2D/FPN 的功能。第一版保持原范数筛选、全局粗相关、原细头/RRU、原 loss。独立 config `exp2a_deep_cim.yaml`。
- 比较同固定候选集合或明确分母的 GT rank、Recall@1/5/10、相似度 margin、入池条件正确率、最终粗召回、范数分布、非 LT 损失、Pose AUC 和延迟。不能只提高入池率就宣称消歧成功。

## Exp-2B Router
- 1/32 全局块相关，源块 Top-K 目标块，映射到 1/8 网格；从可靠几何 GT 聚合多正目标块，处理深度/遮挡/padding/边界。Top-K 索引不可微，需要独立 route loss（多正分类/排序监督）；不得假装细 loss 通过离散选择传梯度。
- 训练可使用 predicted_topk ∪ GT blocks 冷启动，记录 GT 补入率和候选预算，预注册逐渐减少补入的策略。验证及推理只能 predicted-only。路由分支先单独训练/诊断（可使用原全局匹配作参考），不要声称已经实现稀疏关联。独立 config `exp2b_block_routing.yaml`。
- 测试多正 GT、坐标映射、边界、空块、候选预算、非零 route 梯度和推理 GT 禁用；报告低纹理及其他区域 predicted-only GT edge/block Recall@K，明确块级指标和点级指标区别。绘制 Recall@K vs 实际候选预算。路由覆盖门槛在成绩之前以基线可评 GT 覆盖与预算设定，不可事后追分。

## CLI 与决策
真实 20 optimizer updates、checkpoint、双验证及显存。Smoke 通过后独立执行 CIM、Router 的 full 正式训练，使用完整训练池。两分支独立归因；如 router 覆盖未达预注册门槛，不启用完整稀疏训练；允许开发并验证算子，但标明未达到科学准入条件。CIM 亦不得未经证据默认组合。

验收：CIM 和 router 真实接口/梯度/评测可运行或如实 BLOCKED；冻结路由配置与通过/回退理由。交接 `reports/phase_06.md`，下一阶段 `07_sparse_integration/prompt.md`。
