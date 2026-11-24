# 本次实验入口与冻结边界

运行目录：/home/bupt/projects/SLiM-master/runs/coarse_rank_20261123T224428  
报告目录：/home/bupt/projects/SLiM-master/experiment_results/coarse_rank_20261123T224428

本目录为独立、免训练入口；生产 SLiM、已有配置、权重和历史证据不变。未创建 optimizer。protocol.json 在看本轮输出前冻结，A1 是唯一配对基准，A2 只核验跨进程波动。

- inventory.py：只读索引/图像/深度/标定审计；仅向新 run 写数据库和清单。
- policies.py：纯模型输出/图像网格选择器；不接收 GT。oracle 是单独函数。
- capture.py：AST 克隆原生粗匹配，只在原生 P 和 Top-K/阈值之后插入新选择。P 全矩阵分块 SHA 校验，不持久化。
- worker.py、chain.py：逐方法新进程、逐 pair 原子提交、hash 恢复、互斥写入与资源监控。
- report.py、mechanisms.py：阶段 0/1/2A/2B 检查点。
- timing.py：固定诊断集、每对预热+3次同步完整前向；测量中不读 GT、缓存选择或审计数据。
- empty_guard.py：后续真实空输出测试发现原生零父点细窗口的 JIT 异常，故新增显式保护。仅对零父点/零子点返回空张量，非空分支调用原生方法。已在 timing.py 启用；不篡改早先900对已提交推理的代码身份，也不回写其 receipt。全部100对非空结果通过400组、每组3次计时的逐元素一致性验证后，才交付最终报告。未来新数据入口须显式安装此保护并建立自己的输入身份；现有 worker 的诊断清单固定，不可直接替换成新独立清单。
- finalize.py：选做/最终报告、源码快照、交付校验清单。
- proposal.py：仅导出阶段3确认资料；没有自动启动独立集的实现或命令。

全量阶段0–2续跑（只复用有效提交，先查当前进程/状态）：
    cd /home/bupt/projects/SLiM-master
    /home/bupt/miniconda3/envs/slim/bin/python -B diagnostics_coarse_rank/chain.py --methods A1 A2 --background
    /home/bupt/miniconda3/envs/slim/bin/python -B diagnostics_coarse_rank/report.py --stage 0
    /home/bupt/miniconda3/envs/slim/bin/python -B diagnostics_coarse_rank/report.py --stage 1
    /home/bupt/miniconda3/envs/slim/bin/python -B diagnostics_coarse_rank/chain.py --methods B0 B oracle --background
    /home/bupt/miniconda3/envs/slim/bin/python -B diagnostics_coarse_rank/mechanisms.py --stage 2a

C4/C8/C12/Caffine 由各自 coverage_METHOD.json 的 gate_pass 授权，C8/C12/仿射必须在2A/2B交付之后；独立集必须另获明确清单哈希确认。本 README 不是阶段3授权。

只读重新汇总最终报告：
    /home/bupt/miniconda3/envs/slim/bin/python -B diagnostics_coarse_rank/finalize.py --stage final

阶段3位姿协议仍为 canonical LO-RANSAC、原图坐标/K、0.5px、confidence=0.99999、seeds=0..4；此处没有执行该阶段，不把历史 AUC 挪作本次指标。

成本核查补充：最终 B0/B 部署计时使用 timing_deploy.py。它对 B0 不计算行列评分或锚点，对 B 只计算选定池的列排名并执行同一 b 排序；GT/审计/邻域统计不进入计时。600次被替代的旧计时证据保留在 recovery/shared_selection_timing，最终四方法有效部署计时仍为1200次。原效果实验及其900个receipt不变。复跑更正计时前须核对现有进程，入口：python -B diagnostics_coarse_rank/timing_deploy.py --method B0（或B）。