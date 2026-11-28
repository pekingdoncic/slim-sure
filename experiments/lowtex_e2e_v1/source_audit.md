# Phase 00 源码及证据审计

仓库 HEAD 为 e9f5a691563d913b2c2b6b4db4a233f94f94da85。启动时已有未跟踪的 experiments/。原入口文本实际在 experiments/lowtex_e2e_v1prompts/prompts，逐字节比对后复制到本实验 prompts。已有 v2 ZIP 未作为协议。原始命令输出位于 logs/audit，源文件和权重身份位于 manifests/source_identity.json。

## 训练、推理及几何

- train.py → SLiM_Dataset.setup('fit') → ConcatDataset(MegaDepthDataset) → RandomConcatSampler。Lightning 的 _trainval_inference 先做深度双向投影粗监督和窗内监督，再以显式 training=True 调用模型，随后做细监督及 Loss。model.train() 本身不足以进入训练分支。仅读配置不能确定实际推理分支。
- src/backbone/convmamba.py 的 ConvVMamba 在 batch 拼接两图做 stem，在宽度维拼接做 ConvNeXt、SS2D 和深层双图聚合。两个视图各返回三级特征 B×48×H/2×W/2、B×96×H/4×W/4、B×192×H/8×W/8。FPN 的两级深度卷积之后有 BatchNorm2d。src/backbone/mamba/MambaEncoder.py 的深层聚合步幅 4，作用于 1/8 特征。1/32 是内部聚合位置，返回的金字塔没有原生 1/32 输出。
- src/slim.py 的 _forward_train 将首级细特征上采样 2 倍，实际细步幅为 1。粗相似度 B×L0×L1 支持粗监督的行列部分 softmax。GT 父窗口的两侧 8×8 关联产生 M×64×64；_get_fine_coord_train 在有效 GT 子点上采样，加噪后交给 RRU。父索引 m、网格 i/j、监督坐标及 mask 必须共同限制。RRU 按 max_fine_matches 补齐后取真实部分，故仅在输出端截断不能降低展开或 padding 的峰值。src/utils/supervision.py 的 spvs_intermediate 也会先按父点展开，需要在此之前处理父窗口预算，同时保留用于粗损失的原始监督。
- _forward_test 的 _coarse_match_norm_clue 在范数候选池上计算相关。默认 sparse_direct=False 时预先计算的全图粗关联在此分支冗余，训练的粗监督仍使用它。细测按父窗 top-k，阈值是全批细置信度最大值乘 fine_match_thres；配置的 outdoor_test=0.03、outdoor_train=0.1。控制组和原始推理必须记录实际配置。RRU 以两侧原始像素坐标和各自 scale0/1 恢复到特征格，再做双线性取窗、迭代更新目标点。K、最终匹配点与 RANSAC 必须在同一图像坐标系。
- src/loss.py 对粗、细、逐次精修损失加权。Lightning 的 optimizer_step 暖身会将所有 param_group 写成相同 LR，无法直接用于新头和主干差分 LR；需保留同一 optimizer、全局调度计数及采样器身份。冻结 BN 统计不应切断梯度。CUDA kernel 与字典原地更新使整条 SLiM.forward 不适于盲目 checkpoint，先核验纯 Tensor 主干。

## 历史实验的可复核结论

- experiment_results/coarse_rank_20261123T224428 的 Stage2A、Stage2B 和 Final_Report。100 对 R832，固定有效低纹理源格 7954，13 对分母为零。A1、B0 pooled 分别 3486/7954 和 4178/7954；15 场景等权分别 53.8336% 和 61.4543%，差 7.6207 个百分点。非低纹理成功源格净减 4082。7728 个双侧入池的可评对应中，4152 个严格粗格未命中，272 个获邻格精修救回。余下 3880 个中，76.66% 正确目标在 P 行前五。B 改排严格命中 762，对照 B0 为 771；C4−B 等权差 0.2678 个百分点，95% 区间约 [−0.0424, 0.5665]。候选入池、预算竞争、GT 排名与精修均需区分。
- experiment_results/phaseT_20261117T115519/decisions.json。GT1 的 S2−S0 场景等权差 0.03060，95% 区间 [−0.00029, 0.08161]。3108 个新增匹配中仅 171 个可评，134 个正确。等预算 S5 收益判据未通过。
- runs/phase3_diag_20261116T100427/C/summary.json。100 对冻结诊断确认 fine_scale=1、lookup radius=3、4 次精修。99 对可比的 fine_confidence−mconf 对 EPE 的 Spearman 差约 −0.1525。Phase 4 从缓存进行同窗比较，fine−随机的 <2px 差 0.1629 个百分点，95% 区间跨零；oracle−fine 差 8.1946 个百分点。证据在 experiment_results/phase4_20261116T193724/Phase4_Analysis_Report.md。
- evidential_fine/native.py 与 heads.py，experiment_results/phase5_20261117T115519/Phase5_Report.md 和 decisions.json。路线 B 从原生细匹配子点两侧双线性抽取 48×6×6 特征，拼成 96×6×6；GN 卷积后两轴各预测 13 个偏移分箱及 NIG 参数。主干冻结，B 两轮各 13800 次访问，其中因无监督跳过 59、63 次。54427767 个监督样本对应的 α−1 下限比例 84.9357%，不能将该参数视为已校准方差。B 的 <1px 相对 RRU 一步满足 −1pp 容差，整体差仍为负；B 的细阶段 8.83 ms，native1 为 16.97 ms。历史数据支持速度和部分排序收益，无法确立一致的位姿或定位提升。

## 训练池与通用评测

configs/cfg_outdoor_train.py 指向 scene_info_0.1_0.7。保留官方 train_list 中所有行及场景。原索引引用 Undistorted_SfM，当前本地训练照片放在 phoenix/.../dense0/imgs；已有 *_no_sfm 索引改写路径。scripts/audit_pool.py 比对每一条 pair、顺序、共视值、中心参考、引用的内外参、帧身份和深度路径，逐一检查引用帧的本地 RGB/深度文件。即便身份一致，也需区分文件存在、图像解码与几何一致。原文件哈希分别保留，图像字节与原发行版逐张一致性没有外部校验和可验证。

官方 sampler 每个子池每轮有放回抽 100 对，shuffle，repeat=1。单卡逻辑轮 36800 次访问，27 轮 993600 次；原实现不保证遍历完整合法池。论文报告19轮、1024输入、6卡；仓库与上游 train.py 设置 27 轮。最终公平组预算至少覆盖所有合法行，具体计划须在 Phase 02 前按全池统计冻结。[训练代码](https://raw.githubusercontent.com/Band-127/SLiM/master/train.py) · [论文](https://openaccess.thecvf.com/content/CVPR2026/papers/Choo_Scalable_Feature_Matching_via_State_Space_Modeling_and_Sparse_Correlation_CVPR_2026_paper.pdf)

原 outdoor 配置的开发与正式测试都指向 scene_info_val_1500 的 0015/0022。现有 24 个额外索引分桶涉及 8 个其他场景，须和训练名单逐场景求交后方可作为开发候选。历史 runs/coarse_rank_20261123T224428/inventory 有可复核清单。ScanNet 有 100 个测试序列，官方对名单可在相邻 SURE 工程 assets/scannet_test_1500 只读核验。HPatches 有 116 个序列。专项实际目录 data/TexturePoorSfM_dataset 有 17 个场景、每场景 60 个 bag，提供 RGB、内参和位姿。作者给出的开发场景是 1012、1026、1029。[作者划分](https://raw.githubusercontent.com/zju3dv/DetectorFreeSfM/main/scripts/eval_texturepoorsfm_dataset.sh)
