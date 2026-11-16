from pathlib import Path
p=Path('/home/bupt/projects/SLiM-master/diagnostics_phase4/audit4.py')
s=p.read_text()
s=s.replace("    p3by={r['pair_id']:r for r in p3prep}", "    p3by={r['pair_id']:r for r in p3prep}\n    p4by={r['pair_id']:r for r in read(run/'P4/prediction_preparation.json')['rows']}")
s=s.replace("                P3_counts[combo]+=len(ix)", """                ref=p3by[r['pair_id']]['predictions'][combo];predpath=run/'P3/predictions'/pairname(r['pair_id'])/ref['payload']
                if sha(predpath)!=ref['sha256']:raise GateFailure('P3 prediction checksum')
                pred=npzload(predpath)
                if not np.array_equal(pred['points0_orig'],a['source'][ix]*a['scale0']) or not np.array_equal(pred['points1_orig'],a['target'][ix]*a['scale1']):raise GateFailure('P3 selected coordinate/order mismatch')
                P3_counts[combo]+=len(ix)""")
s=s.replace("            P4_counts['t'+str(t)]+=len(target)", """            ref=p4by[r['pair_id']]['predictions']['t'+str(t)];predpath=run/'P4/predictions'/pairname(r['pair_id'])/ref['payload']
            if sha(predpath)!=ref['sha256']:raise GateFailure('P4 prediction checksum')
            pred=npzload(predpath)
            if not np.array_equal(pred['points0_orig'],a['source']*a['scale0']) or not np.array_equal(pred['points1_orig'],target*a['scale1']):raise GateFailure('P4 replayed coordinate/order mismatch')
            P4_counts['t'+str(t)]+=len(target)""")
p.write_text(s)
p=Path('/home/bupt/projects/SLiM-master/diagnostics_phase4/phase4_report.py');s=p.read_text()
s=s.replace("    child=p2['summaries']['all/child']['scene_equal']", "    child=p2['summaries']['all/child']['scene_equal']\n    qmap={r['combo']:r for r in primary}")
needle="      '比例为场景等权；总计数为汇总数，不应直接相除解释加权比例。汇总比例、内点数、50%结果：[P3_filter_quality.csv](P3_filter_quality.csv)。',"
replacement=needle+"""
      '80%预算下，S_m的对极精度/EPE<1比例为 '+fmt(qmap['S_m_80']['epipolar_precision']*100,3)+'% / '+fmt(qmap['S_m_80']['precision_lt1']*100,3)+'%；fine为 '+fmt(qmap['fine_80']['epipolar_precision']*100,3)+'% / '+fmt(qmap['fine_80']['precision_lt1']*100,3)+'%；mconf为 '+fmt(qmap['mconf_80']['epipolar_precision']*100,3)+'% / '+fmt(qmap['mconf_80']['precision_lt1']*100,3)+'%。这体现了对极精度与精细EPE之间的实际折中，不能称为全面超过fine。',"""
assert needle in s;s=s.replace(needle,replacement);p.write_text(s)
