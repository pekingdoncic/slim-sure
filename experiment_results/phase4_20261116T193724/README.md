# SLiM Phase 4

本目录保存全部结论性产物。入口：[Phase4_Analysis_Report.md](Phase4_Analysis_Report.md)。

执行HEAD：d9213801860d8d1cef1f3692b3aac5c43b8a4c0e。新增diagnostics_phase4脚本未提交，精确SHA见[final_integrity.json](final_integrity.json)。

冻结身份：1dc74dff202052885202e401ecf8fe44c40714b5877f0f547320aad86661808a。

中间数据与日志：/home/bupt/projects/SLiM-master/runs/phase4_diag_20261116T193724。新前向、训练更新、checkpoint写入均为0。

恢复命令（显式复用同一run_id和结果目录，跳过已完成阶段）：

~~~bash
cd /home/bupt/projects/SLiM-master && /home/bupt/miniconda3/envs/slim/bin/python -B diagnostics_phase4/run_phase4.py --run-id 20261116T193724 --stage remaining --background
~~~
