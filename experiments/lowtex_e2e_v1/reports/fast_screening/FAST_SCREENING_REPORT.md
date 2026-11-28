# 4080 单周期筛选报告

R640 训练使用全部官方合法有向索引，R640/R832 开发评测单列。百分号只用于 AUC；召回、精度、覆盖率均用 0–1。最终测试场景封存。

## 当前结果

- exp0，官方权重原始 RRU4，端到端。RUNNING，已提交更新 13000，训练访问 52000。
- exp1a，原始 RRU4，仅粗正样本 LT/other 分母独立加权。NOT_RUN，已提交更新 —，训练访问 —。
- rru_matched，重新初始化原生 RRU4，10% 头部热身。NOT_RUN，已提交更新 —，训练访问 —。
- single，单尺度一步头，10% 热身。NOT_RUN，已提交更新 —，训练访问 —。
- crop，多尺度 Crop-Fuse 一步头，10% 热身。NOT_RUN，已提交更新 —，训练访问 —。
- joint，有条件组合纹理损失与多尺度一步头，匹配热身。NOT_RUN，已提交更新 —，训练访问 —。

## 保留的 R832 历史基线

正常 checkpoint 在 32700 步写入，学习率 [2e-05]，访问 130800 对，有向覆盖 1.4759%，无向覆盖 2.1325%。历史训练平均 3.212 对/秒，峰值 allocated/reserved 分别为 4.014/4.709 GiB。
各节点两域开发指标、checkpoint SHA-256、正常退出验证在 legacy_r832/verified.json 和 transition.json。历史结果使用不同分辨率和预算。

## 配置、实测与证据

当前选型 confirm_loader_w4_pin1_persist1_pf2。训练 R640，effective batch 4，BF16，冻结 BN，clip 1。
DATA 与预算见 protocols/fast_screening_v1.md、frozen_inputs.json。每组身份、初始化、源文件哈希、训练日志、梯度/参数变化、恢复记录、两域逐对结果均在 outputs/fast_screening_v1 对应子目录。
TRAINING_THROUGHPUT.csv 包含独立 GPU 实测。TRAINING_COVERAGE.csv 使用真实 checkpoint。FINAL_COMPARISON.csv 的训练耗时以秒表示，正式选型计时在独立进程上使用 100 个固定 TexturePoor 开发对、30 次预热、3 轮重复；推理为 matcher 中位毫秒，VRAM 为字节、推理和训练分别给出。空值表示尚无该次测量。

## 初步趋势与不确定性

- exp1a 对 exp0：PENDING。
- crop 对 single：PENDING。
- crop 对 rru_matched：PENDING。
单轮结论用于筛选。尚未验证的假设包括多轮后的排序、跨随机种子泛化和封存场景上的性能。
