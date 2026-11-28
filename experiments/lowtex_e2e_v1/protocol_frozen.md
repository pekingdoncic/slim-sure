# v3.1 开发评测冻结

TexturePoor 使用作者 1012/1026/1029 开发场景和其余 14 个测试场景，原始图像与每帧非中心主点 K。姿态文件为 world-to-camera，T01=T1·inv(T0)。不重裁剪，右下补零；匹配结果还原原始像素后配原 K。无额外畸变系数，采用发布标定的针孔投影。仅使用原 bag 内同现且两侧帧不同的组合，无向帧对跨 bag 去重，以光轴角 [2°,85°]、基线 ≥0.005（发布位姿单位）过滤。以 30°、60° 划难度，每个场景在非空层间轮转，按 seed 66 的哈希选 100 对。该几何代理没有逐对真实 overlap。从最终清单首个记录固定示例，另展示每场景首个失败对。

MegaDepth 开发采用未被官方训练和 1500 测试使用的 8 个场景，每场景固定哈希选 32 个唯一对。使用原深度和相机计算 GT，保留无法估姿的记录。开发验证全量为 256+300 对。smoke 两路每场景 1 对。正式测试的名单与哈希在 manifests，专项 1400 对，MegaDepth 按全部原 1500 条索引保留，ScanNet 1500，HPatches 全部 116 序列共 580 个 1→j 对。HPatches 在原始像素系用估计单应与真值的四角均值误差做 AUC@3/5/10px；失败记无穷。

姿态使用 SLiM utils.metrics.estimate_pose，OpenCV 原图 0.5px、置信度 0.99999、seed 66。取旋转与符号对称平移夹角的较大者，失败进入 AUC 分母，输出 AUC@5/10/20（百分数）、求解率与 <10° 成功率。双向归一化极线距离平方和阈值 1e−4；空匹配保留、其条件精度未定义。两侧 32px 网格占用率。专项缺少稠密深度、可见性和点对应，不计算点级指标。源网格纹理按 source_audit.md，双侧入池、严格粗召回、定位召回、GT rank 分开统计。全局与场景等权分别提供，空分母显式记录。

开发选模以有 GT 的 MegaDepth 场景等权低纹理成功源格召回（原始像素 EPE <2，固定有效 GT 源格分母）为主，另存开发通用 AUC@10 最佳。接受叠加要求相对同预算对照低纹理差 ≥1 个百分点且场景配对 bootstrap 95% 下限 >0（2000 次，seed 66），两路开发位姿 AUC@10 各下降 ≤1 个百分点，同卡同尺寸中位推理延迟 ≤1.10×。并列取更早的更新。训练中按冻结验证间隔执行完整双验证，所有实验共用频率；最终测试不参与选模。首种子 66，关键入选和基线再用 67、68 复验。新路由附录须在查看相应成绩前冻结。

原生输出和统一 1000 匹配上限两套，按原置信度降序、相同分数按原始输出顺序。效率沿用 prompts/EVALUATION_PROTOCOL.md，以相同至少 100 对数据、30 次预热、3 轮、CUDA 同步计时；独立 matcher/pipeline、各模块、allocated/reserved、动态规模、参数与支持算子 FLOPs。480/640/832/1024/1216 同一冻结模型，OOM 单列。最终测量与开发结果分开。

数据来源见 [作者划分](https://raw.githubusercontent.com/zju3dv/DetectorFreeSfM/main/scripts/eval_texturepoorsfm_dataset.sh)、[位姿读取](https://raw.githubusercontent.com/zju3dv/DetectorFreeSfM/main/src/evaluator/evaluator.py) 及 [采集与裁剪](https://arxiv.org/html/2306.15669v1)。

开发清单索引 SHA-256 4e602669518588b53a761eb142224e5bfa7dac6ed0168a5aafc2f92233c3fec5；通用正式测试清单 SHA-256 1827030d7103cbbd64050476bf9fa4a65e4cc1e2b7b6c5648b527f1245f87f92。
