# SLiM 源码和实验审计

本地静态审计的 HEAD 为 e9f5a691563d913b2c2b6b4db4a233f94f94da85，分支 exp/slim_reliability_v1_20261114T172319。开始时 Git 已报告未跟踪的 experiments/，目标实验目录尚不存在。权重入口为 ckpt/megadepth_19epochs.ckpt，src/slim.py 的 load_state_dict 会去掉 slim. 前缀。训练入口 train.py 将 PL_SLiM、SLiM_Dataset 交给 Lightning。原始 RandomConcatSampler 按子集抽样，一个 epoch 不保证全池覆盖。

- src/backbone/convmamba.py 的 ConvVMamba 先按 batch 拼图，加入位置编码后沿宽度拼接。Easy_SS2D 使用四向扫描，Aggregated_InceptionMamba 在深层分别处理两图并交互。实际 FPN 返回 stride 2/4/8、48/96/192 通道，最后一级为拆分后的深层特征。新分支从这一级掩码池化构造 stride 32。
- src/slim.py 训练用整个 1/8 网格打分，按 GT 父窗作局部细相关和四次坐标更新。推理在两侧各过 0.65×最大 L2 范数筛选，子矩阵上执行双向 softmax、全局 Top-K 和相对阈值。原版此处没有要求双向第一。sparse_direct 只省去推理时未被范数分支使用的全图矩阵。
- src/utils/supervision.py 的粗中心位于 8i+4，细像素中心从 0.5 起。原粗监督分配 HW×HW 的 GT，并忽略 warp_kpts 的可见性返回值。src/loss.py 的粗项对抽出的 90% GT 正例取双向概率的 focal loss，细项使用窗内双向概率，RRU 项按迭代衰减。原 batched 索引和 fine 监督默认的局部父窗索引对多对样本有交叉风险。本实验按对构造监督、计算原损失公式和采样细窗，共享真正成批的 backbone。
- src/reliability.py 在粗父点已经选出后评分，原生 _coarse_match_norm_clue 为 no_grad。历史冻结 SLiM 时，分数监督更新 reliability head，无法改变候选或粗分数竞争。runs/phase3_diag_20261116T100427/Phase3_Diagnostics_Report.md 将排序优势第一次反转定位在选出细子点、RRU 前的 t=0。Phase4_Analysis_Report.md 的同窗 oracle−fine 为 8.1946 个百分点。
- experiment_results/coarse_rank_20261123T224428/Stage1_Report.md 及 stage1_rankings.json 使用同一批 100 对、15 场景的 A1 缓存。固定可评价 LT 源格 7,954，双侧入池 7,728，严格命中 3,576，差额 4,152，其中 272 由邻格细点救回。严格漏失集合 P 行前五 pooled 比例 0.766618，场景等权 0.773980。分母与最终仍无好子点的 3,880 个不同。
- 同目录 Stage2A_Report.md 中 A1/B0 的成功源格为 3,486/4,178，分母均为 7,954。场景等权 LT 召回 0.538336/0.614543，非低纹理成功源格净减 4,082。排名 B−B0 为 −0.000980，95% 区间 [−0.003889, 0.002669]。Stage2B_Report.md 的邻域 C4−B 为 +0.002678，[−0.000424, 0.005665]。阶段 3 的 MegaDepth 历史复验计 1,488 个批准对/2 场景，ScanNet 计 1,500 对/100 场景。新协议保留 MegaDepth 原索引全部 1,500 行。历史复验单独报告，见 stage3_official_20261124/Research_Conclusion.md。
- experiment_results/phaseT_20261117T115519/PhaseT_Report.md 的 S0/S1 为同一 100 对扫描，场景等权 LT 召回 0.538336/0.568936。src/texture_budget.py 在归一化灰度上计算 Sobel/8 的平方和，stride 8 的零偏移锚点使用 16×16 有效邻域。R832 阈值 0.0001920461654663086 来自历史 100 个训练对的 1,531,088 个合法网格。
- experiment_results/phase5_20261117T115519/Phase5_Report.md 的 B 对真实原生一步 EPE<1px 差 −0.004954，G8 满足预设容差，G10 的准确率条件失败。差值判据为 99 对/15 场景。计时为 100 对的细阶段，B/native1/native4 分别为 8.833565/16.970247/54.590655 ms。此实验继续用原版 RRU-4。
- 4080 冻结协议 experiments/lowtex_e2e_v1/protocol_frozen.md 的 MegaDepth 开发集为 8×32，TexturePoor 开发集为 3×100，正式专项为 14×100。历史数据目录实际拼写是 data/TexturePoorSfM_dataset。官方 train_list 的正 overlap 行由 train_pool.json 给出，物理无向唯一数 6,105,188。正式服务器须重新核验索引、场景和数据可读性。

交付检查发现仓库根 .gitignore 的 data/ 规则也匹配实验内的数据适配源码。实验内局部规则已放行这些 Python 文件；原 .gitignore 无改动。
