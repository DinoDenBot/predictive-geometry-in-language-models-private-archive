"""Audit and export all declared gap-attribution phases without changing outcomes."""
from pathlib import Path
import argparse
import csv
import hashlib
import json


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(root, write=False):
    spec_path = root / 'specification.json'
    spec = json.loads(spec_path.read_text())
    inventory, rows, sensitivity = [], [], []
    for phase in spec['scope']:
        path = root / phase / 'results.json'
        if not path.exists():
            inventory.append({'phase': phase, 'status': 'incomplete' if path.parent.exists() else 'pending',
                              'results': str(path)})
            continue
        result = json.loads(path.read_text())
        assert result['status'] == 'complete' and result['cells'] == 3600
        assert result['specification_sha256'] == sha(spec_path)
        assert result['input_manifest_sha256'] == sha(path.parent / 'inputs.json')
        assert result['per_document_sha256'] == sha(path.parent / 'per_document.parquet')
        assert len(result['sensitivities']) == 12
        inventory.append({'phase': phase, 'status': 'complete', 'results': str(path),
                          'sha256': sha(path), 'input_manifest_sha256': result['input_manifest_sha256'],
                          'analysis_code_sha256': result['analysis_code_sha256']})
        p = result['primary']; s = p['slopes']
        row = dict(phase=phase, **s, unmasked_G=result['unmasked_cached_gap_slope'],
                   G_R_percent=100*s['delta_G']/s['delta_R'],
                   H_G_percent=100*s['H']/s['delta_G'],
                   P_G_percent=100*s['P']/s['delta_G'],
                   H_first_G_percent=100*s['H_first']/s['delta_G'],
                   H_last_G_percent=100*s['H_last']/s['delta_G'],
                   retention_percent=100*p['retained']/p['total'],
                   retained=p['retained'],total=p['total'],
                   max_checkpoint_residual=p['max_absolute_checkpoint_residual'],
                   residual_slope_fraction=p['residual_slope_fraction'],
                   numerical_gate_passed=p['numerical_gate_passed'])
        for metric in ('H','P','H_first','H_last','delta_G'):
            values=list(p['target_slopes'][metric].values())
            row[metric+'_positive_targets']=sum(v>0 for v in values)
            row[metric+'_target_min']=min(values)
            row[metric+'_target_max']=max(values)
        rows.append(row)
        for v in result['sensitivities']:
            assert abs(v['slopes']['delta_G']-v['slopes']['H']-v['slopes']['P']-v['slopes']['delta_e']) < 1e-10
            sensitivity.append(dict(phase=phase,tau=v['tau'],mode=v['mode'],**v['slopes'],
                                    H_G_percent=100*v['slopes']['H']/v['slopes']['delta_G'],
                                    retention_percent=100*v['retained']/v['total'],
                                    clipped=v['clipped_transition_instances'],
                                    residual_slope_fraction=v['residual_slope_fraction'],
                                    **{m+'_positive_targets':sum(x>0 for x in v['target_slopes'][m].values())
                                       for m in ('H','P','H_first','H_last','delta_G')}))
    historical=[]
    for filename,key in [('binary_fisher_comparator_results.json','analyses'),
                         ('smollm2_g_external_results/smollm2_g_external_results.json','phases')]:
        old_path=root.parent/filename
        for old in json.loads(old_path.read_text())[key]:
            phase=(('initial_' if old['setting']=='original70' else 'fresh_')+old['phase']
                   if key=='analyses' else 'smollm2_'+old['role'])
            current=next((r for r in rows if r['phase']==phase),None)
            if current is None:
                continue
            difference=current['unmasked_G']-old['observed_projections']['M_G']
            assert abs(difference)<1e-12, (phase,difference)
            historical.append({'phase':phase,'source':str(old_path),'source_sha256':sha(old_path),
                               'unmasked_G_slope_difference':difference})
    report={'specification_sha256':sha(spec_path),'inclusion':inventory,'primary':rows,
            'sensitivities':sensitivity,'summary_code_sha256':sha(Path(__file__))}
    report['historical_unmasked_reproduction']=historical
    if write:
        assert len(rows)==len(spec['scope']), 'All declared phases must be accounted for before final export'
        amendment_path=root/'input_precedence_amendment.json'
        amendment=json.loads(amendment_path.read_text())
        assert {i['analysis_code_sha256'] for i in inventory} <= {amendment['old_code_sha256'],amendment['new_code_sha256']}
        report['input_precedence_amendment_sha256']=sha(amendment_path)
        with (root/'all_phases_summary.json').open('x') as f:
            json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
        for name,data in [('primary_summary.csv',rows),('sensitivity_summary.csv',sensitivity)]:
            with (root/name).open('x',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    print(json.dumps({'inclusion':inventory,'primary':rows},indent=2))
    return report


if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('--write',action='store_true')
    args=ap.parse_args();summarize(args.root,args.write)
