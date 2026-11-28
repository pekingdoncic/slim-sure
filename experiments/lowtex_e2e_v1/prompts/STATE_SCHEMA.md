# v3 阶段交接与恢复规范

运行时文件位于 EXP_ROOT/reports/。首次由模板创建，原子写入；既有 v1/v2 状态备份迁移，保留日志、协议及 run_id。初次启动记录用户授权 `authorized_through_batch=4`, `budget=full`, `final_test_authorized=true`（仅在 Phase 08 模型冻结后生效）。批次边界不是暂停点，阶段硬依赖才是。

每阶段工程状态 NOT_RUN/RUNNING/PASS/FAIL/BLOCKED/SKIPPED；科学状态 NOT_EVALUATED/INCONCLUSIVE/SUPPORTED/NOT_SUPPORTED；正式训练状态 NOT_RUN/RUNNING/PASS/FAIL/BLOCKED/SKIPPED/NOT_APPLICABLE。工程 PASS 不能替代 full PASS；覆盖率不足或预算未完成不可标 full PASS。每个实验分别记录状态、覆盖率、计划/实测 optimizer updates、checkpoint、日志、数据与协议 hash。可选分支不满足预注册条件可 SKIPPED，主实验硬阻塞须 BLOCKED，不隐瞒。

每阶段写 reports/phase_XX.md、state.json、handoff.md：当前/下一阶段、真实命令/exit code、产物/日志、hash、科学判断、依赖/回退理由、恢复命令。状态 READY/RUNNING/BLOCKED/COMPLETE；不再使用 AWAITING_AUTHORIZATION 作为批次停点。已完成阶段自动推进；真实硬阻塞仅阻止依赖任务，继续独立任务。Phase 08 自动 test（模型、选模、配置、权重、协议 hash 已冻结），缺条件则 BLOCKED/NOT_RUN 并说明。

长训练的 checkpoint 记录 optimizer/scheduler/AMP/RNG、sampler 游标、已覆盖图像对身份和训练预算；恢复时核对 run 身份，不能丢掉已见样本计数或静默重训。若 CLI 中断，重新使用同一启动指令恢复。建议 Phase 02 生成 `scripts/run_workflow.sh` 或等效本地监督入口：先实际检查 `codex --help`/非交互接口是否存在及是否允许在当前环境启动（不得硬编码未核实的 CLI 选项）；可用时按 state 启动单阶段子会话，锁防并发、日志持久化、检查真实状态、有限重试、无进度/硬阻塞停止、禁止越权；不支持时使用当前 Codex 会话内循环并明确限制。不能将脚本存在等同其已验证运行，不承诺 CLI 关闭后自动继续。所有自动执行仅限本地授权范围，不改系统环境、不自动 commit/push、不租用 GPU。
