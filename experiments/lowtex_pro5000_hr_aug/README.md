# CCS-SLiM 高分辨率实验

从 SLiM 仓库根目录，在已有 CUDA/Mamba 环境的服务器运行。所有模型、缓存、日志、权重快照及报告写入 experiments/lowtex_pro5000_hr_aug。当前交付状态为 **PENDING_ON_SERVER**。

## 准备数据

脚本默认从 ckpt/megadepth_19epochs.ckpt 严格加载权重；从 data/megadepth 下读取 train、test 和 index。index 需含官方 trainvaltest_list、scene_info_0.1_0.7 和 scene_info_val_1500；已冻结的等价 _no_sfm 布局也受支持。TexturePoor 从 data/Texture-PoorSfMDataset 或 data/TexturePoorSfM_dataset 读取。其场景目录下应含 bag 以及每帧 images、intrins、poses。ScanNet 需要测试场景的 color/pose 和官方 test.npz、intrinsics.npz。若索引位于另一个项目，只需把 --scannet-index-root 指向已有文件所在目录。HPatches 按原序列和 H_1_j 读取。

可通过 MEGADEPTH_ROOT、TEXTUREPOOR_ROOT、SCANNET_ROOT、SCANNET_INDEX_ROOT、HPATCHES_ROOT、SLIM_WEIGHTS 配置路径，也可使用同名功能的 CLI 参数。--python 接受已安装依赖的解释器，--gpu 接受单个设备 ID 或 UUID。需要 PyTorch 支持 scatter_reduce_、非重入 checkpoint、torch.amp.GradScaler("cuda")，以及原仓库的 Mamba/选择性扫描扩展。服务器的 nvidia-smi 需要能读取本进程的显存。代码读取本目录已冻结的可迁移 manifest，不需要同步 4080 的历史输出或重新生成旧实验。

## 服务器命令

以下示例使用服务器环境中已激活的 Python 和可见设备 0。在不同 shell 中重设前两行变量即可复用。

```bash
EXP=experiments/lowtex_pro5000_hr_aug
RUN="$EXP/scripts/run_full_pipeline.sh"

# 查看选项及单独环境检查
bash "$RUN" --help
bash "$EXP/scripts/preflight.sh" --gpu 0 --mode smoke --run-id env_check

# 核心矩阵工程验收，两种高分辨率，每段每模型 20 个 optimizer updates
bash "$RUN" --gpu 0 --mode smoke --run-id smoke01 --vram-cap-gib 30

# 只做 R1024 架构筛选，同预算 P1-P4，开发集评价
bash "$RUN" --gpu 0 --mode short --run-id r1024a --resolution-plan 1024 --experiments P0,P1,P2,P3,P4 --dev-only

# 固定 5000+3000 更新，双开发验证和效率测量
bash "$RUN" --gpu 0 --mode medium --run-id medium01 --vram-cap-gib 30 --dev-only

# 覆盖完整官方训练池，封存检查点后运行 TexturePoor、MegaDepth-1500、ScanNet-1500
# 示例索引目录内放置服务器已有的 test.npz 和 intrinsics.npz
bash "$RUN" --gpu 0 --mode full --run-id full01 --full-epochs 1 --scannet-index-root data/scannet/index

# 断点恢复保留原命令的 mode、run_id、预算、数据路径、模型和分辨率
bash "$RUN" --gpu 0 --mode full --run-id full01 --full-epochs 1 --scannet-index-root data/scannet/index --resume-pipeline

# 新终端实时查看同一个 run，可随时 Ctrl-C 结束监视
bash "$RUN" --mode full --run-id full01 --monitor
```

单独启用 P0 的双分辨率开发验证可用 `bash "$RUN" --mode smoke --run-id p0dev --experiments P0`。明确指定 Full 的两段步数时，用 --steps-1024 与 --steps-1216 替代 --full-epochs。两个预算之和至少覆盖完整训练池的一次访问。1 个全池 epoch 是 **553,918 次**有效 batch=16 的更新，时间与磁盘成本由服务器实测。Medium 只有 8,000 次更新，覆盖率按实际访问统计。

## 状态、结果和恢复

| 内容 | 本实验目录下的位置 |
|---|---|
| 每段训练、更新日志、固定末步、选择出的权重 | outputs/MODE/P1/RUN_ID/R1024/，其他模型及分辨率同理 |
| 显存试探、每个候选及实际吞吐选择 | 同一分辨率目录的 probe/trials.json、probe/selected.json |
| 各域逐对结果、场景等权和总体统计 | 各段 before/、quick_STEP/、after/、selected/，正式结果在模型的 RUN_ID/test/ |
| 共享无写回原始数据的索引缓存、真实 Smoke 回执 | outputs/shared/indices/、outputs/shared/smoke/ |
| 环境、状态、校准、任务命令、模型冻结 | reports/MODE/RUN_ID/ 下的 server_preflight.json、progress.json、live_status.json、calibration/、jobs/、model_freeze.json |
| 故障与恢复命令 | reports/MODE/RUN_ID/latest_error.md、latest_error.json、debug_bundle/ |
| 终端输出 | logs/MODE/RUN_ID/ 及 reports/MODE/RUN_ID/pipeline.log |
| 总报告、CSV 和曲线 | reports/MODE/RUN_ID/final/FINAL_RESEARCH_REPORT.md 及同目录文件 |

工程失败会保留子进程退出码和最近完整更新的检查点，独立阶段可继续。缺数据的实验域标为 BLOCKED；未运行的正式测试标为 NOT_RUN。流水线退出码 0 为所请求阶段完成，1 为工程失败，2 为部分阻塞。恢复时比较源码、配置、权重、清单和预算身份；若版本变化，使用新的 run_id 保存新的运行记录。Smoke 验收回执与正式训练状态分别保存。所有报告均带 run_id，可追溯到模型各阶段 JSON/JSONL。

可独立重建已有运行的报告：

```bash
python -m experiments.lowtex_pro5000_hr_aug.evaluation.summarize_results --mode full --run-id full01
```

源码审计见 [source_audit.md](source_audit.md)，实验与选模约定见 [protocol_pro5000.md](protocol_pro5000.md)，文件及模块清单见 [implementation_report.md](implementation_report.md)，待执行的验收项见 [test_report.md](test_report.md)。
