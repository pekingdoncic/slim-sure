# 全阶段公共约束（每阶段必读）

## 1. 路径与保护
- 仓库根目录以下统一使用 `experiments/lowtex_e2e_v1/`（简称 EXP_ROOT）；Prompt 放在 `EXP_ROOT/prompts/`。新增源码、配置、测试、脚本、缓存（尽量定向）、报告、日志、运行输出及生成索引全部留在 EXP_ROOT。禁止覆盖原始 SLiM、train.py/test.py、历史实验及权重、原始数据/划分、已有冻结协议；禁止清理用户工作区或重置 Git。审计工作区初始状态并与结束状态比较。
- 优先 import、wrapper、subclass、composition，必要时在 EXP_ROOT 内建立最小适配副本并记录差异。不要复制整库、数据或大型权重。实际路径通过 `git ls-files`、`rg`、`find` 确认，禁止假定文件存在。默认不安装/升级 CUDA、PyTorch、Mamba，不下载大型数据；需要环境变更时报告原因。
- 目录建议：`configs/ models/ losses/ data/ evaluation/ training/ utils/ tests/ scripts/ manifests/ outputs/ logs/ reports/`，根目录 `README.md`、`source_audit.md`、`protocol_frozen.md`、`implementation_report.md`、`test_report.md`、`requirements_notes.md`。不创建空壳伪装实现。

### 阶段范围（硬约束）
首次统一授权 Phase 00–08、full 与冻结后最终测试。Batch 1–4 仅作为任务分组，不停等授权。只创建当前阶段实际需要的文件，不提前实现后续模块；通过硬依赖即自动继续。未实现 stage 明确非零退出。可选 Anchor/几何后端在预注册条件满足时自动实施；数据重构和额外增广不在默认范围。

## 2. 数据和公平性
- 训练只用标准 MegaDepth 的完整官方合法训练图像对池；保留官方场景/图像对/共视筛选与场景均衡采样原则，不默认修改索引、增加数据或纹理抹平增广。保留完整图像输入。具体全量覆盖规则见本节附录。
- 专项数据默认 `data/Texture-PoorSfMDataset`，只用于评测，原始数据只读。官方 3 个验证场景用于开发，14 个测试场景只用于冻结后的最终评测。具体场景编号在 Phase 01 核实。禁止以测试成绩调参。
- MegaDepth 独立验证集与训练、最终 MegaDepth-1500 场景隔离；如果官方命名 val 与正式 test 实际重合，记录并从训练来源预先按场景留出开发验证集，所有对照一致，重新冻结协议。不得把同一个集合同时用于调参和独立测试。ScanNet-1500、HPatches 为最终通用测试，缺数据报告 BLOCKED。
- 所有公平对照使用相同训练图像对预算、初始化条件（新增模块的初始化另记录）、解冻策略、分辨率、优化器、调度器、随机种子和几何评估规则。官方 checkpoint 推理结果与公平继续微调基线分别报告。不要把额外训练收益归因于模块。每次只改变目标变量；筛选规则、损失、精修和协方差分别开关。
- 先冻结协议再查看相应实验成绩：数据名单、哈希、采样规则、纹理阈值来源、有效 GT 分母、精度、步数、指标、RANSAC、匹配预算、成功/停止判据、验证频率与选模规则。协议变更建立新版本，保留旧文件和原因；不得覆盖或事后改阈值追成绩。新增阶段的预注册附录同样在该阶段成绩之前冻结。记录 `sha256sum`（或等价工具）、Git commit 和 dirty diff/相关源文件哈希。

## 3. 训练配置与真实执行
- 4080/4080 SUPER/Laptop 以实际 GPU 检查为准；目标 16GB。建议 R640、micro-batch=1、BF16 mixed、accumulate=4、clip norm=1.0、新模块 LR=1e-4、主干 LR=2e-5；只是起点。640 成功后可测试 832；1024/1216 优先大显存设备并实测，不能假定 PRO 5000 显存。Mamba 不支持 BF16 时显式记录并选 FP16/FP32 配置，不静默改变。
- 4080 配置默认 `freeze_backbone_bn_stats: true`、`gradient_checkpointing: true`、`max_fine_train_matches: 1024`。主干所有 BatchNorm（含 SyncBatchNorm 如存在）保持 eval 的 running stats 行为，在每次 model.train()、验证返回、阶段解冻和 resume 后重新应用；检查 running_mean、running_var、num_batches_tracked 不变。BN affine 是否训练由解冻策略决定；新增模块使用 LayerNorm/GroupNorm（不用新 BN）。不对整个可训练主干调用 no_grad。
- 激活 checkpoint 优先 `use_reentrant=False`（以当前 PyTorch 支持为准），只用于可安全重算的可训练纯 Tensor 路径；审计 Mamba 自定义算子、随机性及原地/字典状态副作用；比较开启/关闭时输出、目标参数梯度和峰值显存。若不兼容，不得静默回退，记录原因并显式使用兼容配置。冻结且 no_grad 的层不需要 checkpoint。
- `max_fine_train_matches: 1024` 定义为每个图像对进入受控细匹配/精修训练路径的候选数上限（审计实际父窗口与子匹配语义，另外记录窗口数及子匹配数）；在昂贵 crop/展开前尽量限制其工作量，保持索引、GT 与 mask 对齐，确保原生正/负监督语义不被破坏。采用可复现、默认不依赖纹理标签的采样，适用于所有公平训练对照；不得仅给 Ours 额外低纹理配额。记录 LT/other 采样前后数量、有效 GT 和空组；推理/验证不使用训练上限。上限是初始预算，不保证显存；必要时预注册调整并同步所有对照。官方无改动基线与受控训练基线分别报告，关闭 cap 时恢复原行为。
- 模式 head_only / partial_unfreeze / end_to_end；检查 requires_grad、冻结层 BN/dropout 状态及优化器参数组。冻结特征可 no_grad，但不能切断需学习的梯度。坐标/协方差/softmax reduction 等数值敏感操作 FP32。
- full（也兼容 short/medium）对新增随机初始化精修头支持两阶段热身：默认前 15% optimizer updates（可预注册 10%–20%，按冻结的 full optimizer updates 计算，short=150/1000、medium=750/5000）head_only，其余 end_to_end，差分 LR（新头 1e-4、主干 2e-5 为起点）。此比例包含在总步数内，不额外增加训练样本。Exp-1A 无新头，保持粗特征梯度有效，从第 1 步按预注册解冻训练；无新头的 Exp-0 对照按对应实验的匹配策略训练。Crop-Fuse 的对照可采用重新初始化的原生精修头并匹配两阶段策略，同时保留原权重继续微调基线；注明初始化差异。独立协方差校准/路由训练不机械套用该热身。切换必须保持 optimizer 已有状态、scheduler 的全局 step/差分 LR、AMP scaler（若有）、RNG 和 sampler 状态；禁止重复/遗漏参数，resume 能在切换前后正确恢复。smoke 可显式设短热身以覆盖切换，不把它当正式训练。
- Shell 在任何 torch 导入之前配置 CUDA allocator。仅当 `PYTORCH_CUDA_ALLOC_CONF` 与 `PYTORCH_ALLOC_CONF` 均未由用户设置时，默认导出 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；尊重用户已有配置，记录生效值、PyTorch 版本及 allocator backend。旧版/WSL/驱动不支持时记录警告或错误，提供显式关闭开关/环境变量覆盖（例如 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`），不静默改变；此实验性选项可能减少碎片，不能保证消除 OOM。
- 预算 `smoke`=20 optimizer updates，`short`=1000，`medium`=5000，`full`=默认正式预算（完整训练池及冻结的完整 epoch/step 计划）。max_steps 是 optimizer 更新次数，记录实际 micro-batch/有效图像对数量。梯度累积不降低单样本显存，分组 loss 的归一化协议须明确。支持 checkpoint 及 optimizer/scheduler/RNG 恢复，OOM 明确报错。
- 每个可训练阶段必须验证 BN buffers、checkpointing 梯度、训练 cap 与推理无 cap、热身切换（适用时）及实际 allocator 配置；必须实际运行 CLI：import/compileall、`bash -n`、单元测试、真实 MegaDepth 数据 forward、GT、loss、backward、optimizer update、checkpoint save/reload、两路 validation。记录有限 loss、非零目标梯度、参数更新、冻结参数不变、峰值 allocated/reserved、完整实际命令及日志。可额外小子集 overfit 检查梯度，但不能当作泛化证据。
- 若 Windows CLI 可调用 WSL，则优先检查 `wsl.exe` 和现有 Conda/GPU/数据；环境不可达时完成可运行检查并标记真实 GPU/数据测试 BLOCKED，提供准确恢复命令。合成 Tensor 不替代真实训练，不允许编造 PASS。默认正式长训，Phase 08 冻结后自动最终测试。短训只用于诊断。

## 4. 评价与日志
- 两路开发验证：MegaDepth 独立验证（Pose AUC@5/10/20、匹配质量、低纹理区域指标）；Texture-Poor SfM 官方验证场景两视图适配（Pose AUC、旋转/平移方向误差、成功率、GT 极线精确率、RANSAC 内点率/数、匹配数、空间覆盖）。没有可靠点级 GT 时不计算真实 EPE、Recall、PCK；极线一致不能等同真实像素对应。
- 低纹理区域诊断复用历史协议，固定有效 GT 源网格分母；双侧入池率、GT 候选覆盖、GT rank/Recall@K、严格粗召回、精召回、EPE、精确率、覆盖率、非低纹理损失。区分未入池、路由漏失、竞争失败、精修失败、过滤损失。报告场景等权和全局统计，位姿失败计入分母。模型仍输入全图。EPE 需同时报告匹配覆盖，防止过滤困难点后虚高。
- 效率：同硬件/精度/分辨率/匹配预算，CUDA 同步、warmup、固定重复次数。训练 forward/backward/update、数据加载、allocated/reserved 峰值；推理 backbone/coarse/fine/refine/total、参数/FLOPs（无法可靠计算则标记）。区分推理与训练峰值。
- 每次运行独立 run_id，不覆盖。`outputs/<exp>/<run_id>/` 保存 resolved_config.yaml、run_manifest.json（commit、源/协议/数据/权重 hash、种子、GPU、软件、分辨率、预算、启用模块、解冻范围）、train_metrics.jsonl、双验证/逐场景/逐对结果、memory_stats.json、checkpoint_last/best_lowtex/best_general、summary.json、stdout.log。无可用成绩填 NOT_RUN / BLOCKED，禁止虚构。
- 训练前、固定间隔（建议 500 optimizer steps，可预先调整）、训练后做两路开发验证；smoke 使用预先冻结的小验证子集，明确区分完整成绩。正式选模使用冻结规则与完整验证。最终测试前冻结 checkpoint/配置，不依赖最终测试选择模型。

## 5. 交接与判据
- 每阶段读取 `STATE_SCHEMA.md`，写 `reports/phase_XX.md`、更新 `reports/state.json` 和 `reports/handoff.md`，汇总 `implementation_report.md`、`test_report.md`。工程状态与科学状态分别记录；真实命令、exit code、日志、产物路径、hash、失败原因必须可查。
- 工程 PASS 不等于效果 PASS。未达到预注册收益门槛不强行叠加模块，可继续独立分支；涉及上一阶段模型的分支只能使用已明确选择且符合门槛的版本。所有未验证创新均标为研究假设。发现历史结论不符，以原始报告为准。

## 6. Full 训练数据和预算（覆盖所有阶段中旧的短训/授权表述）
- Phase 00 统计官方训练索引中所有合法 pair、scene/subscene 数、实际 sampler 的每 epoch 抽取数、有/无放回、去重身份、有效过滤、训练/验证隔离。生成只读训练池 manifest/哈希，禁止只取 100 对、几个 scene、`Subset`、`fast_dev_run`、`limit_train_batches<1` 等作为正式训练。仅 smoke 可用小数据；所有完整主实验（Exp-0/1A/1B，及满足依赖的 1C/1D/2A/2B/2C/2E）必须使用完整训练池。训练期每对 fine cap 1024 是点级显存控制，不是图像对数据截断；同对照一致。
- 区分「全量训练池」与「每 epoch 遍历全部 pair」：若官方 sampler 每 epoch 只抽取部分 pair，优先保持其场景/共视配额，在各子池采用可复现的无放回循环覆盖（保留原索引，另存 sampler 计划；不等同原版逐轮随机抽样）。正式报告逐 epoch 及累计 unique pair coverage，确保在完整训练结束时所有合法训练 pair 至少访问一次；如原官方 sampler 本身不能保证覆盖，使用上述 coverage-aware sampler，并对所有公平对照一致启用，另保留官方 sampler/权重基线用于区分影响。不允许把修改后的采样称为完全等同官方采样。若官方训练池大于原计划预算，事前提高所有公平组共同预算到足以遍历全池，禁止静默漏掉场景或难例。若用户需要逐 epoch 全量遍历，提供 `full_epoch` 选项；默认完整训练周期累计全量覆盖，不承诺每 epoch 全遍历。
- 训练总预算参考核实的 SLiM 官方配置（论文报告 19 epochs / 1024；本机分辨率可不同），在 Phase 02 正式训练前冻结共同的 optimizer updates、micro-batch 数、有效样本预算、epoch 定义、完整覆盖计划、LR/scheduler、随机种子和分辨率。优先 R640 真实显存测试后固定 R832 若可行；若 R832 不可行统一 R640，不为单个模型偷偷降分辨率。高分辨率训练/评测独立配置。主实验同一初始 checkpoint 和同一可复现采样序列（能共享时）；不同冻结策略/初始化需匹配额外对照。正式训练从预训练权重微调，不自动等同从零复现。
- 全量运行日志至少记录训练池大小、实际已见 unique pairs、累计覆盖率、各场景覆盖、过滤/读取失败清单、训练步数/epoch、每组实际有效 GT、sampler 游标与 RNG。完整覆盖未达到不得标记 full PASS，恢复未完成的 run。坏文件不能静默删除；如必须排除，在正式训练前冻结规则及共同合法池并报告。
- Smoke 20 optimizer updates 只用于工程验收；short/medium 可选诊断，不自动替代 full 或用其成绩淘汰主实验。需要科学判断时使用完整训练及完整开发验证的预注册判据。固定同一主实验种子，正式入选模型与关键公平基线按预注册多种子（建议 3）复验；如果算力客观不够，记录未完成，不编造成果。
- 阶段收益判据必须在成绩之前冻结：低纹理主指标、容许通用退化、推理预算、配对场景 bootstrap 及路由覆盖门槛；不得以测试集选择网络。科学失败保留负结果并继续独立分支，依赖失败按可解释规则回退，不能用无限调参阻塞流程。
- 若独立 MegaDepth 开发验证只能从官方训练场景中预留，必须事先场景级划出、所有组一致，并明确全量是预留后合法池，不再称为完整未修改官方训练 split；优先使用现有独立验证。最终测试隔离优先于覆盖率。
- 完整评测要求见 `prompts/EVALUATION_PROTOCOL.md`，从 Phase 01 冻结规则，Phase 02 起接入计时，Phase 08 自动运行。
