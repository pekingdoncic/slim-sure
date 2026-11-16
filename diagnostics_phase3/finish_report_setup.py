
from pathlib import Path
import json
root=Path('/home/bupt/projects/SLiM-master/diagnostics_phase3')
p=root/'report_phase3.py'
s=p.read_text()
s=s.replace("    change=changes['common_BC/parent/t0_minus_coarse_linf']", "    csvout(run/'Phase3_Diagnostics_Report.csv',predictive)\n    change=changes['common_BC/parent/t0_minus_coarse_linf']")
s=s.replace("        '## A：粗标签与最终结果','']", "        '核心表CSV：'+link(run,'Phase3_Diagnostics_Report.csv')+'。附图：'+link(run,'figures/stage_advantage.svg')+'；'+link(run,'figures/stage_advantage.png')+'。','',\n        '## A：粗标签与最终结果','']")
assert s!=p.read_text()
p.write_text(s)
m=root/'report_manifest.json'
rows=json.loads(m.read_text())
rows.extend([
 ['逐阶段预测力核心CSV与区间图','Phase3_Diagnostics_Report.csv;figures/stage_advantage.svg;figures/stage_advantage.png','相同共同父点；负值代表完整头相对mconf更好'],
 ['独立统计核查','independent_statistical_checks.json','7项核查通过，包括固定共同父点与子点、计数守恒、首20几何计划']
])
m.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
