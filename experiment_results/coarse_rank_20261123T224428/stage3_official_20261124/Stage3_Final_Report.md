# 阶段 3 完整官方基准复验交付

完成时间：2026-11-24T19:18:08.106334+08:00。MegaDepth 1,488 对、ScanNet 1,500 对；两份获批完整清单均已完成条件允许的全部方法、五种子位姿和实测计时。未训练，未运行其他室外留出集、oracle、探索半径或仿射。

两份基准都存在历史选参/基准使用，分别作为室外和室内描述性复验。室内场景不并入室外场景数；ScanNet 按官方 scene 整数归并 scan 后缀，缺少更高层级建筑地点身份。

## megadepth1500

报告：megadepth1500_Report.md；完整指标：megadepth1500_summary.json。

- B0-A：低纹理召回差 7.020279 pp；在本完整已用基准上的描述性正差值，不构成未见独立场景证据。
- B-B0：低纹理召回差 -0.246532 pp；当前固定配置在此指标上退化。

主 C：skipped，原因 ['V scenes<3']。

## scannet1500

报告：scannet1500_Report.md；完整指标：scannet1500_summary.json。

- B0-A：低纹理召回差 0.122789 pp；在本完整已用基准上的描述性正差值，不构成未见独立场景证据。
- B-B0：低纹理召回差 -0.000157 pp；现有样本证据不足以确定增益方向。
- C-B：低纹理召回差 0.011102 pp；在本完整已用基准上的描述性正差值，不构成未见独立场景证据。
- C-A：低纹理召回差 0.133734 pp；在本完整已用基准上的描述性正差值，不构成未见独立场景证据。

主 C：pass，原因 []。

## 对机制和训练投入的解释

诊断集已确认：原生全局 Top-K 与相对阈值是主要筛选环节，配额 B0 的收益须与目标重排的 B−B0、邻域 C−B 分开归因。同池 oracle 是诊断集严格粗格上界，不能作为真实部署方法或细点召回保证；其历史结果见父目录 Final_Report.md。
megadepth1500：配额差 7.020279 pp，排名增量 -0.246532 pp。排名判断：当前固定配置在此指标上退化。
该域未运行 C：覆盖门槛不满足，不能推断邻域零收益。
scannet1500：配额差 0.122789 pp，排名增量 -0.000157 pp。排名判断：现有样本证据不足以确定增益方向。
邻域判断：在本完整已用基准上的描述性正差值，不构成未见独立场景证据。

当前证据可支持固定配置的机制取舍，尚缺未曾用于选择的独立场景复验来支持训练泛化收益。不得把配额增益归给未证实的可学习重排，也不把区间含零解释为排除了该机制。

实际 run 占用 3.881 GiB；结束时宿主 C: 58.046 GiB、Linux 584.886 GiB 可用。环境监控保留每次资源采样及内核游标，检查边界见 initial_environment.json 和 environment_probe/；不宣称历史底层存储故障已根治。

协议、批准记录和派生清单在 run 根目录；逐对预测及原始选择证据在 dataset/method/pairs；几何结果在 pose；进程与真实命令在 launch.json、logs/*_launch.json；代码冻结在 code_identity.json 与 code_snapshot。

运行目录：/home/bupt/projects/SLiM-master/runs/coarse_rank_20261123T224428/stage3_official_20261124
报告目录：/home/bupt/projects/SLiM-master/experiment_results/coarse_rank_20261123T224428/stage3_official_20261124

恢复：cd /home/bupt/projects/SLiM-master && /home/bupt/miniconda3/envs/slim/bin/python -B /home/bupt/projects/SLiM-master/diagnostics_coarse_rank_stage3/run_stage3.py --background。完成后执行会校验并复用已提交结果，不重复模型前向。
