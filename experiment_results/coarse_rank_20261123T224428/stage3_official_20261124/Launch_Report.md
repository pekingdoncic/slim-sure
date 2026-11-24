# 阶段 3 启动实测记录

此文件是启动时快照，不是完成报告。当前状态请读 run/stage_status.json 与 supervisor_heartbeat.json；最终报告名称为 Stage3_Final_Report.md。

用户已明确批准两份官方完整清单，按描述性复验执行。MegaDepth 无序去重后 1,488 对、2 场景；ScanNet 1,500 对、100 个官方场景 ID。两者都有历史选参或基准使用，不能命名为未见独立验证集。未批准的其他室外清单不执行。

- MegaDepth SHA-256：21f702c9f6dcdf97c76322ba72f60a5deee090f3d8bbc5bef10fc6a71e693a21
- ScanNet SHA-256：248c9a744b3efba107ff85fbb6063e50f6ab9f20d82b37592a9a6a5959c6f8c4
- HEAD：d033f6dbb47c2e650390ba6053a22ac921fd5f3e
- 新执行代码清单 SHA-256：69c4878a36264a7b1d62d00f27ba69fc8b30cd68bb5e8ecbf107715d82e4a4aa
- 当前 boot：817dfd85-9798-40e4-bc96-bb39996b2538
- 启动北京时间：2026-11-24 09:00:07
- supervisor PID：3173957
- 首个 GPU worker PID：3173973

当前任务不训练。R832、batch=1、AMP、候选及配额、低纹理阈值和 GT 定义全部沿用冻结合同。完整前向保留原生细匹配与 RRU；每方法新进程，每 pair 预热一次再计时三次。计时窗口排除 GT/审计/哈希/落盘/几何求解。完整原生 P/logits 分块重建并哈希核验，不保存全矩阵。

## 前置检查

三个既有诊断 pair 的新加载器与原路径逐元素一致，完整 P/logits 哈希与已提交 A1 相同；A/B0/B/C 包装器在同一既有诊断 pair 的真实输出逐元素一致。2,988 对正式记录的位姿有限且相对平移非零。CPU PoseLib 2.0.5 与冻结 canonical solver 合同一致。

ScanNet 使用 uint16 PNG 毫米深度 /1000；将以 640×480 为标定基准的 K 按原始 RGB 宽高逐轴映射；原始 C2W 取逆得到 W2C。已核对图像、深度哈希及原始 pose 文件；正式清单的方向和顺序未改变。

启动前有界真实数据读取、1 MiB 写入/fsync/回读/清理通过。内核日志沿此前确认的同 boot 游标连续读取，未见新增存储故障。历史故障记录保留，不宣称根因已修复。

09:00:06 实测：宿主 C: 可用 50,856,165,376 bytes（约 47.36 GiB）；Linux 输出文件系统可用 632,167,096,320 bytes（约 588.75 GiB）。WSL VHD 与 swap 同在 C:，宿主卷只计一次。预算 12 GiB 新存储峰值 + 2 GiB 余量，启动容量通过。运行中始终保留 2 GiB 宿主及 guest 余量、512 MiB 主存门禁；监控新内核存储异常，心跳过期上限 900 秒。

## 首批实测与 ETA

09:01:42：A 已提交 79 对，总墙钟约 84.35 秒，即 1.068 秒/对（包括数据读取、审计、落盘和环境检查）。前向计时约 0.094 秒/对。按当前速度且 ScanNet C 可执行时，剩余完整前向粗估 3.1 小时；含五种子位姿与统计汇总暂估 4–6 小时。不同方法和两域的数据加载、相关矩阵规模、几何难度不同，后续以各阶段实际速度更新。

09:04:03：A 已提交 227/1,488，所有环境状态 pass；C: 47.17 GiB，主存约 12.2 GiB，swap 已用约 0.645 GiB。首批峰值 allocated 约 2.13 GiB；reserved 约 2.66 GiB，详见逐对记录。

## 执行路径与恢复

运行目录：/home/bupt/projects/SLiM-master/runs/coarse_rank_20261123T224428/stage3_official_20261124

报告目录：/home/bupt/projects/SLiM-master/experiment_results/coarse_rank_20261123T224428/stage3_official_20261124

实际后台命令：

    cd /home/bupt/projects/SLiM-master
    /home/bupt/miniconda3/envs/slim/bin/python -B -u diagnostics_coarse_rank_stage3/run_stage3.py --background

恢复使用相同命令。先核查 launch.json 中 PID 的实际命令行；已运行则接管监控，已有效提交则校验身份和哈希后复用。逐对原子 receipt 是完成边界，不以半写入 npz/json 视为完成。确认身份不一致时停止整个运行，不覆盖原证据。

进度查看：

    /usr/bin/python3 -B runs/coarse_rank_20261123T224428/stage3_official_20261124/operator_status.py

精确子命令、PID：logs/*_launch.json。监督日志：supervisor.log；方法日志：logs/<dataset>_<method>.log；心跳：heartbeats/ 与 supervisor_heartbeat.json；环境：environment_probe/。GPU 前向和 CPU 位姿串行；CPU 位姿最多三个持久进程。

完成前所有结论均为未完成状态。MegaDepth C 因最多只有两个场景，无法满足至少三个有效场景的原覆盖门槛，仍会完整计算 S/U/H/V 后记录跳过；A/B0/B 全量执行。ScanNet C 根据自己的无 GT 覆盖门槛决定。oracle、R8/R12、仿射不进入此次官方复验。