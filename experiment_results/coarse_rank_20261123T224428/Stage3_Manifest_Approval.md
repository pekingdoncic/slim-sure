# 阶段 3 清单确认资料



当前状态：等待用户明确确认，独立集模型运行数为 0。以下是阶段 0 在模型结果揭盲前冻结的完整候选；本文件不授权启动。



拟运行 A/B0/B，主 C 在获批清单上重新过同一无 GT 覆盖闸门后才运行。三份清单按来源/域分别汇总；oracle、R8/R12、仿射不进入阶段 3。



## megadepth1500



完整 pair 清单：/home/bupt/projects/SLiM-master/runs/coarse_rank_20261123T224428/inventory/megadepth1500.jsonl

SHA-256：21f702c9f6dcdf97c76322ba72f60a5deee090f3d8bbc5bef10fc6a71e693a21

帧与标定清单：/home/bupt/projects/SLiM-master/runs/coarse_rank_20261123T224428/inventory/megadepth1500_frames.jsonl

帧清单 SHA-256：67cdeba32b5aaf7a425954981f8469dd3262343e6e8974bc64a0e65e1c0632ce

原始 1500 对；无序/反向去重后 1488 对，全部结构可读；唯一图像 806；场景 ID 2。最终 GT 可评场景数尚未实测。

逐场景完整对数：0015=596；0022=892

与本地已知使用清单无交集的场景：[]。

- phase1_2_train：场景/图像/无序 pair 交集 = 0/0/0。

- phase1_2_val：场景/图像/无序 pair 交集 = 0/0/0。

- phase5_training：场景/图像/无序 pair 交集 = 0/0/0。

- phase5_validation：场景/图像/无序 pair 交集 = 0/0/0。

- phaseT_and_phase3_4：场景/图像/无序 pair 交集 = 0/0/0。

- prior_hybrid_benchmark_selection：场景/图像/无序 pair 交集 = 2/806/1488。



初步成本：四方法逐 pair 预热+三次前向约 0.71 GPU 小时（当前诊断实测外推）；五种子位姿若沿历史逐单元启动方式，约 0.14 单工作进程天，理想三工作进程约 0.05 天。该值含旧进程启动/import 开销，不是纯求解时间；未对独立数据实测，误差可能很大。

照搬本轮完整诊断缓存将约需 2.53 GiB。阶段 3 若批准大清单，必须先完成有界流式指标/位姿提交与峰值预算，不能照搬完整矩阵或逐点缓存。



## megadepth_other_holdout



完整 pair 清单：/home/bupt/projects/SLiM-master/runs/coarse_rank_20261123T224428/inventory/megadepth_other_holdout.jsonl

SHA-256：c47abd6b55b2e6b5518cb9548920fe3d1a34d6470f9ebf7dbb22c1d0f018241e

帧与标定清单：/home/bupt/projects/SLiM-master/runs/coarse_rank_20261123T224428/inventory/megadepth_other_holdout_frames.jsonl

帧清单 SHA-256：884392e4bdf3767d5206e30e63bb98be2bd2370cd34a7850a69f06b729b689d6

原始 657080 对；无序/反向去重后 456845 对，全部结构可读；唯一图像 7036；场景 ID 8。最终 GT 可评场景数尚未实测。

逐场景完整对数：0008=121587；0019=70909；0021=41507；0024=8925；0025=9669；0032=109986；0063=51974；1589=42288

与本地已知使用清单无交集的场景：['0008', '0019', '0021', '0024', '0025', '0032', '0063', '1589']。

- phase1_2_train：场景/图像/无序 pair 交集 = 0/0/0。

- phase1_2_val：场景/图像/无序 pair 交集 = 0/0/0。

- phase5_training：场景/图像/无序 pair 交集 = 0/0/0。

- phase5_validation：场景/图像/无序 pair 交集 = 0/0/0。

- phaseT_and_phase3_4：场景/图像/无序 pair 交集 = 0/0/0。

- prior_hybrid_benchmark_selection：场景/图像/无序 pair 交集 = 0/0/0。



初步成本：四方法逐 pair 预热+三次前向约 216.63 GPU 小时（当前诊断实测外推）；五种子位姿若沿历史逐单元启动方式，约 42.12 单工作进程天，理想三工作进程约 14.04 天。该值含旧进程启动/import 开销，不是纯求解时间；未对独立数据实测，误差可能很大。

照搬本轮完整诊断缓存将约需 776.69 GiB。阶段 3 若批准大清单，必须先完成有界流式指标/位姿提交与峰值预算，不能照搬完整矩阵或逐点缓存。



## scannet1500



完整 pair 清单：/home/bupt/projects/SLiM-master/runs/coarse_rank_20261123T224428/inventory/scannet1500.jsonl

SHA-256：248c9a744b3efba107ff85fbb6063e50f6ab9f20d82b37592a9a6a5959c6f8c4

帧与标定清单：/home/bupt/projects/SLiM-master/runs/coarse_rank_20261123T224428/inventory/scannet1500_frames.jsonl

帧清单 SHA-256：7f4e5e9a4637c29894e020d6d76ada789f3f03070ee1df64548d43cb69e0a0a3

原始 1500 对；无序/反向去重后 1500 对，全部结构可读；唯一图像 2596；场景 ID 100。最终 GT 可评场景数尚未实测。

逐场景完整对数：scene0707=15；scene0708=15；scene0709=15；scene0710=15；scene0711=15；scene0712=15；scene0713=15；scene0714=15；scene0715=15；scene0716=15；scene0717=15；scene0718=15；scene0719=15；scene0720=15；scene0721=15；scene0722=15；scene0723=15；scene0724=15；scene0725=15；scene0726=15；scene0727=15；scene0728=15；scene0729=15；scene0730=15；scene0731=15；scene0732=15；scene0733=15；scene0734=15；scene0735=15；scene0736=15；scene0737=15；scene0738=15；scene0739=15；scene0740=15；scene0741=15；scene0742=15；scene0743=15；scene0744=15；scene0745=15；scene0746=15；scene0747=15；scene0748=15；scene0749=15；scene0750=15；scene0751=15；scene0752=15；scene0753=15；scene0754=15；scene0755=15；scene0756=15；scene0757=15；scene0758=15；scene0759=15；scene0760=15；scene0761=15；scene0762=15；scene0763=15；scene0764=15；scene0765=15；scene0766=15；scene0767=15；scene0768=15；scene0769=15；scene0770=15；scene0771=15；scene0772=15；scene0773=15；scene0774=15；scene0775=15；scene0776=15；scene0777=15；scene0778=15；scene0779=15；scene0780=15；scene0781=15；scene0782=15；scene0783=15；scene0784=15；scene0785=15；scene0786=15；scene0787=15；scene0788=15；scene0789=15；scene0790=15；scene0791=15；scene0792=15；scene0793=15；scene0794=15；scene0795=15；scene0796=15；scene0797=15；scene0798=15；scene0799=15；scene0800=15；scene0801=15；scene0802=15；scene0803=15；scene0804=15；scene0805=15；scene0806=15

与本地已知使用清单无交集的场景：[]。

- phase1_2_train：场景/图像/无序 pair 交集 = 0/0/0。

- phase1_2_val：场景/图像/无序 pair 交集 = 0/0/0。

- phase5_training：场景/图像/无序 pair 交集 = 0/0/0。

- phase5_validation：场景/图像/无序 pair 交集 = 0/0/0。

- phaseT_and_phase3_4：场景/图像/无序 pair 交集 = 0/0/0。

- prior_hybrid_benchmark_selection：场景/图像/无序 pair 交集 = 100/2596/1500。



初步成本：四方法逐 pair 预热+三次前向约 0.71 GPU 小时（当前诊断实测外推）；五种子位姿若沿历史逐单元启动方式，约 0.14 单工作进程天，理想三工作进程约 0.05 天。该值含旧进程启动/import 开销，不是纯求解时间；未对独立数据实测，误差可能很大。

照搬本轮完整诊断缓存将约需 2.55 GiB。阶段 3 若批准大清单，必须先完成有界流式指标/位姿提交与峰值预算，不能照搬完整矩阵或逐点缓存。



官方 MegaDepth 与 ScanNet 均和以前 hybrid_match 基准/选择分析清单重叠；它们可作描述性复验或室内迁移描述，不能声称未用独立集。其他 8 场景未发现已知场景、规范路径图像或无序 pair 重叠；缺少跨 scene ID 物理地点别名表及完整离机选参历史。ScanNet 按索引第一场景整数合并重复扫描后缀；100 个序列恰为 100 个 sceneXXXX，均 15 对。



不缩小清单换取短任务。请明确确认所选完整清单及上述 SHA-256 后才能进入阶段 3；未确认时停下。
