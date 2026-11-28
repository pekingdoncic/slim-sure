# 实际运行环境

WSL Ubuntu，现有解释器 /home/bupt/miniconda3/envs/slim/bin/python。Python 3.10.22、torch 2.1.1+cu118、CUDA runtime 11.8、mamba_ssm 2.2.2、Lightning 2.3.3、OpenCV 4.11.0。RTX 4080 SUPER，设备报告 16375 MB。首次 nvidia-smi 采样时全卡已用 7169 MiB，实际可用内存会随其他负载变化。allocator 探针在 torch 导入前指定 expandable_segments:True，backend 报告 native。导入和单次 CUDA 分配成功。requests 发出依赖版本警告，保留于 logs/audit/8.json。

原始数据保持只读。实验运行设置 PYTHONDONTWRITEBYTECODE，并将新编译和绘图缓存指向 EXP_ROOT/cache。使用现有环境，无系统安装步骤。

纯主干 GPU 96 像素探针（logs/audit/backbone_probe.json）：BF16、FP16 可反传；非重入重算输出最大差 0，首层梯度最大差分别 0.00415039、0.00044250。FP32 Triton 内核共享内存需求 114944 B，硬件限额 101376 B。仅代表本探针；正式训练需另验。

## 正式训练与本地监督

R832、batch 1、累积 4 的实际 smoke 峰值 allocated / reserved 为 3.752 / 4.336 GiB。训练用 BF16，数值敏感损失输入转 FP32；开发推理用原生阈值、FP16 autocast。用户已设置 allocator 时保留其值，需要显式关闭时可设 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False。相同 full run 恢复会核对已冻结的 allocator 配置。

当前基线第 1–562 步实测中位数为 1.2783 秒/optimizer update，按此速度单组 2,267,529 步约需 33.5 天连续训练，另加周期性验证、停机与其他负载耗时。该数字是早期速度外推。训练日志较大，WSL 所在卷本次查得空闲约 575 GiB。

Ubuntu 内通过已有 Windows CLI 使用管理环境的自动审批，实际 WSL 读写验证凭据在 logs/phase02，身份和已验证参数在 reports/host_config.json。阶段监督与正式训练分别持锁，独立 WSL 进程在主机开机、WSL/GPU/登录仍可用时持续执行。系统重启后用 README 的同一入口接续。初次 UNC 沙箱初始化与未签名 UNC 脚本启动的实际错误保留于日志；采用原生 Linux 入口与已验证本机 CLI，无需更改系统策略。
