# 阶段 1：正确目标排名与筛选流失

100/100 对真实 A1 缓存，15 场景；固定可评价低纹理源格 7,954。双方入池 7728；严格正确目标未被选中 4152；其中邻格细匹配已救回 272，仍无好子点 3880。

严格漏失集合中，P 行前五可达比例：pooled 0.766618；场景等权 0.773980。这只衡量池内可达性，不能视为端到端增益。

最大的筛选交叉单元：global Top-K=False、严格相对阈值=False，共 2643 个低纹理正确对应。两条线均未通过的对应只计一次。

真正未救回集合拆分：无子点 3543；有子点但 GT 全无效 62；有有效 GT 但全部 EPE≥2 px 275。

原始温度缩放相似度与 P 分别报告行/列排名、1/2–5/6–10/>10 分档、前 1/3/5 命中率和非低纹理对照；见 stage1_rankings.json、rank_loss_per_pair.csv。矩阵外正确目标单列，零分母保留并标 undefined。每对详细排名、P margin、阈值标记和并列边界成员数在 A1/pairs/*.npz。

新增排名并列按原始格 ID 破平；A 的原生 Top-K 顺序保持不变。统计用 pair→scene→场景等权，区间为 2,000 次整场景 bootstrap。参数保持预注册值，阶段 1 结果不用于调参。

清单 SHA：113c82ca8b0fba6a45648cbf69a88440be4246bf6a531e960a977dc0253446ee；协议 SHA：2b915fadd960082f327bff5cfc5a53fdb2763d40215a42ea2b36fbce9e1988c2。代码哈希见 JSON。

续跑：cd /home/bupt/projects/SLiM-master && /home/bupt/miniconda3/envs/slim/bin/python -B diagnostics_coarse_rank/chain.py --methods B0 B oracle --background。独立集运行数仍为 0。
