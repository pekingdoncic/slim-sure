# DataLoader 工程修复

第一批 smoke 在初始化 CUDA、运行双域验证之后 fork CPU 解码进程。worker 报 Segmentation fault，异常没有转为成功访问。错误在 first_attempt_failures.json 和 logs/fast_screening 对应命令日志中。测速程序先启动解码进程，四项真实计算配置均通过。

训练器现与已测路径一致，在 CUDA、验证库和计算模型初始化之前建立有限队列。仅对处于第 0 步、采样游标 0、优化器状态为空的受影响 smoke 迁移运行标识，保留下属目录中的原始 checkpoint、identity、progress、前后 SHA-256 及旧代码快照。原模型初始化、完整数据序列、候选、数学目标、优化器和待比较的参数均保留，已完成的起点开发验证沿用其原回执。

Exp-0 从第 1 步、4 对的真实断点恢复，完成 20 步、80 次访问及两分辨率验证和独立推理测速。Exp-1A 也完成同样的闭环，单独对真实训练对的粗监督反传至骨干和 coarse temperature。正式训练的源文件身份在各自的 identity.json 与 frozen_sources.tar.gz 中。
