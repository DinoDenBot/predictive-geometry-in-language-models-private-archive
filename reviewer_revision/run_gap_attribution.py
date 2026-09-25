"""Retrospective, descriptive two-group attribution of paired Fisher gaps.

Every analysis uses a frozen specification, a content-hashed input inventory,
and a new output directory. Historical study outputs are read-only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exposure_observability import phase_statistic, target_slopes

VOLUME = Path('/Volumes/My Passport/data_inference')
PHASES = ('initial_validation', 'initial_confirmation', 'fresh_validation',
          'fresh_confirmation', 'pythia160_validation', 'pythia160_confirmation',
          'smollm2_validation', 'smollm2_confirmation')
REVISIONS = {'70m': 'e93a9faa9c77e5d09219f6c868bfc7a1bd65593c',
             '160m': '582159a2dfe3e712a8d47ae83dec95ae3bde8e7e',
             'smollm2': '93efa2f097d58c2a74874c7e644dbc9b0cee75a2'}
THRESHOLDS = (0., 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
METRICS = ('delta_G', 'H', 'P', 'H_first', 'H_last', 'interaction',
           'delta_e', 'delta_F', 'delta_R', 'delta_B')


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def binary(a, b):
    return 2 * (np.arcsin(np.sqrt(b)) - np.arcsin(np.sqrt(a)))


def gap(a, b, h):
    """F((a,b),h), with the continuous angular limit and exact h=1 branch."""
    a, b, h = np.broadcast_arrays(*[np.asarray(v, dtype=np.float64) for v in (a, b, h)])
    if not (np.all(np.isfinite(a+b+h)) and np.all((a > 0) & (a < 1))
            and np.all((b > 0) & (b < 1)) and np.all((h >= 0) & (h <= 1))):
        raise ValueError('gap requires interior probabilities and affinity in [0,1]')
    c = np.clip(np.sqrt(a*b) + np.sqrt((1-a)*(1-b))*h, 0., 1.)
    sine = np.sqrt((1-c)*(1+c))
    theta = np.arctan2(sine, c)
    ratio = np.divide(theta, sine, out=np.ones_like(c), where=sine > 0)
    r = 2*ratio*(np.sqrt(b)-c*np.sqrt(a))/np.sqrt(1-a)
    return np.where(h == 1, 0., r-binary(a, b))


def decompose(a0, b0, h0, a1, b1, h1):
    f00, f01 = gap(a0,b0,h0), gap(a0,b0,h1)
    f10, f11 = gap(a1,b1,h0), gap(a1,b1,h1)
    first, last = f01-f00, f11-f10
    return {'H': .5*(first+last), 'P': .5*((f10-f00)+(f11-f01)),
            'H_first': first, 'H_last': last, 'interaction': last-first,
            'delta_F': f11-f00, 'F0': f00, 'F1': f11}


def phase_inputs(phase: str, overlays: list[Path]):
    role = phase.rsplit('_',1)[1]
    initial = phase.startswith('initial_')
    smol = phase.startswith('smollm2_')
    arch = 'smollm2' if smol else ('160m' if phase.startswith('pythia160_') else '70m')
    study = ('exposure_observability_v1' if initial else
             'exposure_geometry_study3_smollm2' if smol else 'exposure_geometry_extension_v2')
    root = VOLUME/study
    manifest_path = root/('candidate_manifest.parquet' if initial else 'design/candidate_manifest.parquet')
    manifest = pd.read_parquet(manifest_path)
    frame_path = root/('analysis/'+role+'/per_document.parquet' if initial or smol else
                      'analysis/'+arch+'/'+role+'/per_document.parquet')
    sources = [manifest_path]
    if frame_path.exists():
        frame = pd.read_parquet(frame_path); sources.append(frame_path)
    else:
        paths = sorted((root/f'outcomes/{arch}/primary/{role}').glob('*.parquet'))
        if len(paths) != 3:
            raise FileNotFoundError(f'no complete phase frame: {phase}')
        frame = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True); sources += paths
    if 'text_hash' not in frame:
        frame = frame.merge(manifest[['doc_id','text_hash']],on='doc_id',validate='many_to_one')
    if 'd' not in frame:
        frame['d'] = np.log2(frame.K + 1)
    if len(frame) != 3600 or frame.block_id.nunique()!=200 or frame.target_id.nunique()!=3:
        raise ValueError(f'unexpected panel {phase}')
    if frame.duplicated(['target_id','doc_id']).any():
        raise ValueError('duplicate phase cell')
    for _, g in frame.groupby(['target_id','block_id']):
        if sorted(g.K.tolist()) != [0,1,2,4,8,16]:
            raise ValueError('incomplete dose block')
    frame = frame.sort_values(['target_id','block_id','doc_id']).reset_index(drop=True)
    records = []
    metas = {}
    for target in sorted(frame.target_id.unique()):
        meta = root/(f'outcomes/{role}/{target}.json' if initial else f'targets/{target}/completion.json')
        metas[target] = json.loads(meta.read_text()); sources.append(meta)
    candidates = [o/study for o in overlays]+[root]
    def locate(relative: Path, base: bool, target: str):
        if base and not initial:
            for tree in candidates:
                paired = tree/relative.parent.parent/f'base-{role}-{target}'/relative.name
                if paired.is_file():
                    return paired
        for tree in candidates:
            p = tree/relative
            if p.is_file():
                return p
            if base and not initial:
                # Base namespace may differ across acquisition workers.
                for q in sorted(p.parent.parent.glob('*/'+p.name)):
                    if q.is_file():
                        return q
        return candidates[0]/relative
    for row in frame.itertuples(index=False):
        m = metas[row.target_id]
        if initial:
            br = Path(f'outcomes/base_cache/{role}/{row.text_hash}.npz')
            fr = Path(f'outcomes/{role}/{row.target_id}/cache/{row.text_hash}.npz')
            bc, fc = m['base_cache_config_hash'], m['target_cache_config_hash']
        else:
            prefix = 'outcomes/cache' + ('' if smol else '/'+arch)
            br = Path(f'{prefix}/base/base/{row.text_hash}.npz')
            fr = Path(f'{prefix}/final/{row.target_id}/{row.text_hash}.npz')
            rev = REVISIONS[arch]
            bc = hashlib.sha256(f'{arch}:{rev}:base:base:geometry-v2'.encode()).hexdigest()
            fc = hashlib.sha256(f'{arch}:{rev}:final:{m["checkpoint_sha256"]}:geometry-v2'.encode()).hexdigest()
        records.append({'doc_id':str(row.doc_id), 'text_hash':row.text_hash,
                        'target_id':row.target_id,'base':str(locate(br,True,row.target_id)),
                        'final':str(locate(fr,False,row.target_id)),'base_config':bc,'final_config':fc})
    return frame, records, sorted(set(sources))


def freeze(output: Path):
    spec = {'status':'retrospective_descriptive_specification_before_new_outcome_computation',
            'created_utc':datetime.now(timezone.utc).isoformat(), 'scope':list(PHASES),
            'primary':{'complement_mass_min':.01,'affinity_mode':'project'},
            'thresholds':list(THRESHOLDS), 'affinity_modes':['project','exclude_out_of_range'],
            'attribution':'average both replacement orders for endpoint pair x and complement affinity h',
            'closure':'delta_G = H + P + delta_e; all use one aligned common mask',
            'numerical_gate':'abs(phase beta_delta_e) <= .01 abs(phase beta_delta_G)',
            'closure_tolerance':1e-10,'residual_quantiles':'64 evenly spaced primary retained transitions per document-target cell; descriptive sample',
            'inference':'descriptive phase/target slopes and dose means only; no new p-values or causal CIs',
            'selection':'analyze every complete available phase; inventory unavailable phases; no partial-target or outcome-based phase selection',
            'all_outcomes_reported':True,'source_plan_sha256':digest(ROOT/'paper/STRENGTHENING_PLAN_20260907.md')}
    write_json(output/'specification.json',spec)


def load_cache(path: str, expected: str):
    with np.load(path,allow_pickle=False) as z:
        if str(z['config_hash'].item()) != expected:
            raise ValueError(f'cache config mismatch: {path}')
        logp = z['likelihood'][:,:,0].astype(np.float64)
        alr = z['alr'].astype(np.float64)
        mask = z['distinct'].astype(bool)
    if logp.shape[1]!=24 or alr.shape != (*mask.shape,3) or mask.shape!=(logp.shape[0],23):
        raise ValueError('invalid context grid shape')
    return logp, alr, mask


def paired_values(base, final):
    if base[0].shape != final[0].shape or not np.array_equal(base[2], final[2]):
        raise ValueError('base/final context masks do not match')
    mask = base[2]
    arrays = []
    for logp,alr,_ in (base,final):
        a,b = np.exp(logp[:,:-1])[mask],np.exp(logp[:,1:])[mask]
        arrays.append((a,b,alr[:,:,1][mask],alr[:,:,2][mask]))
    interior = np.ones(int(mask.sum()),dtype=bool)
    for a,b,l,r in arrays:
        if not np.all(np.isfinite(a+b+l+r)) or np.any(l<0) or np.any(l>np.pi+1e-8):
            raise ValueError('nonfinite or invalid retained geometry')
        interior &= (a>0)&(a<1)&(b>0)&(b<1)
    vals = [tuple(x[interior] for x in ar) for ar in arrays]
    (a0,b0,l0,r0),(a1,b1,l1,r1) = vals
    hs = [(np.cos(l/2)-np.sqrt(a*b))/np.sqrt((1-a)*(1-b)) for a,b,l,r in vals]
    h0,h1 = [np.clip(h,0,1) for h in hs]
    d = decompose(a0,b0,h0,a1,b1,h1)
    g0,g1 = r0-binary(a0,b0),r1-binary(a1,b1)
    e0,e1 = g0-d.pop('F0'),g1-d.pop('F1')
    d.update(delta_G=g1-g0,delta_e=e1-e0,delta_R=r1-r0,delta_B=binary(a1,b1)-binary(a0,b0))
    closure = np.abs(d['delta_G']-d['H']-d['P']-d['delta_e'])
    if closure.size and float(closure.max())>1e-10:
        raise ArithmeticError('transition closure failed')
    mass = np.minimum.reduce([1-a0,1-b0,1-a1,1-b1])
    domain = (hs[0]>=0)&(hs[0]<=1)&(hs[1]>=0)&(hs[1]<=1)
    rows = []
    for tau in THRESHOLDS:
        for mode in ('project','exclude_out_of_range'):
            keep = (mass>=tau)&(domain if mode=='exclude_out_of_range' else True)
            n = int(keep.sum())
            if n == 0:
                raise ValueError('empty document under declared sensitivity')
            rows.append({'tau':tau,'mode':mode,'total':int(mask.sum()),'retained':n,
                         'interior':int(interior.sum()),'h_clipped':int((~domain & keep).sum()),
                         'e_abs_max':float(np.maximum(np.abs(e0[keep]),np.abs(e1[keep])).max()),
                         'delta_e_abs_mean':float(np.abs((e1-e0)[keep]).mean()),
                         'closure_max':float(closure[keep].max()),
                         **{k:float(v[keep].mean()) for k,v in d.items()}})
    pk = np.flatnonzero(mass>=.01)
    chosen = pk[np.unique(np.linspace(0,len(pk)-1,min(64,len(pk)),dtype=int))]
    samples = np.column_stack([e0[chosen],e1[chosen],(e1-e0)[chosen]])
    # Unmasked cached means reproduce prior results, independently of saturation exclusion.
    gfull = []
    for a,b,l,r in arrays:
        gfull.append(float(np.mean(r-binary(np.clip(a,0,1),np.clip(b,0,1)))))
    return rows,samples,{'base_G':gfull[0],'final_G':gfull[1],'delta_G':gfull[1]-gfull[0]}


def analyze_phase(phase: str, output: Path, overlays: list[Path]):
    dest = output/phase
    if dest.exists():
        raise FileExistsError(f'phase output already exists: {dest}')
    frame,records,sources = phase_inputs(phase,overlays)
    paths = sorted({Path(r[k]) for r in records for k in ('base','final')})
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        print(json.dumps({'phase':phase,'status':'missing_inputs','missing':len(missing)}),flush=True)
        return {'phase':phase,'status':'missing_inputs','missing_paths':missing}
    dest.mkdir(parents=True)
    started = time.monotonic()
    inventory = {str(p):{'sha256':digest(p),'bytes':p.stat().st_size} for p in paths+sources}
    write_json(dest/'inputs.json',{'files':inventory,'pairs':records,
              'specification_sha256':digest(output/'specification.json'),
              'analysis_code_sha256':digest(Path(__file__))})
    frame[['block_id','doc_id','target_id','K','d','text_hash']].to_parquet(dest/'panel.parquet',index=False)
    base_cache={}; allrows=[]; samples=[]; fullrows=[]
    for idx,(record,row) in enumerate(zip(records,frame.itertuples(index=False)),1):
        bp = record['base']
        if bp not in base_cache:
            base_cache[bp]=load_cache(bp,record['base_config'])
        final=load_cache(record['final'],record['final_config'])
        # The published primary endpoint provides an independent pairing/reproduction check.
        for label,cache in [('base',base_cache[bp]),('final',final)]:
            col=label+'_S_R'
            if col in frame:
                observed=float(np.quantile(cache[1][:,:,2][cache[2]],.9,method='linear'))
                if abs(observed-float(getattr(row,col)))>1e-9:
                    raise ValueError(f'published Q90 mismatch: {phase}/{row.doc_id}/{label}')
        values,sample,full=paired_values(base_cache[bp],final)
        ident={'block_id':row.block_id,'doc_id':row.doc_id,'target_id':row.target_id,'K':int(row.K),'d':float(row.d)}
        allrows.extend([{**ident,**v} for v in values]);samples.append(sample);fullrows.append({**ident,**full})
        if idx%300==0:
            print(json.dumps({'phase':phase,'cells_done':idx,'elapsed_seconds':round(time.monotonic()-started,1)}),flush=True)
    data=pd.DataFrame(allrows);full=pd.DataFrame(fullrows)
    data.to_parquet(dest/'per_document.parquet',index=False);full.to_parquet(dest/'unmasked_cached_gap.parquet',index=False)
    sensitivity=[]
    for (tau,mode),group in data.groupby(['tau','mode'],sort=True):
        slopes={k:phase_statistic(group,k) for k in METRICS}
        targets={k:target_slopes(group,k) for k in METRICS}
        closure=abs(slopes['delta_G']-slopes['H']-slopes['P']-slopes['delta_e'])
        if closure>1e-10:
            raise ArithmeticError('slope closure failed')
        sensitivity.append({'tau':float(tau),'mode':mode,'slopes':slopes,'target_slopes':targets,
                            'dose_means':{str(int(k)):v for k,v in group.groupby('K')[list(METRICS)].mean().to_dict('index').items()},
                            'counts_by_target_dose':[{'target_id':t,'K':int(k),'total':int(g.total.sum()),'retained':int(g.retained.sum())}
                                                     for (t,k),g in group.groupby(['target_id','K'])],
                            'retained':int(group.retained.sum()),'total':int(group.total.sum()),
                            'clipped_transition_instances':int(group.h_clipped.sum()),
                            'max_absolute_checkpoint_residual':float(group.e_abs_max.max()),
                            'max_transition_closure_error':float(group.closure_max.max()),
                            'slope_closure_error':closure,
                            'residual_slope_fraction':abs(slopes['delta_e']/slopes['delta_G']) if slopes['delta_G'] else None,
                            'numerical_gate_passed':bool(abs(slopes['delta_e'])<=.01*abs(slopes['delta_G']))})
    primary=next(s for s in sensitivity if s['tau']==.01 and s['mode']=='project')
    sampled=np.concatenate(samples)
    result={'phase':phase,'status':'complete','cells':len(frame),'targets':sorted(frame.target_id.unique()),
            'primary':primary,'sensitivities':sensitivity,
            'unmasked_cached_gap_slope':phase_statistic(full,'delta_G'),
            'residual_sample':{'n_transitions':len(sampled),'quantile_levels':[0,.01,.5,.99,1],
                               'columns':['e0','e1','delta_e'],
                               'quantiles':np.quantile(sampled,[0,.01,.5,.99,1],axis=0).tolist()},
            'elapsed_seconds':time.monotonic()-started,'input_manifest_sha256':digest(dest/'inputs.json'),
            'per_document_sha256':digest(dest/'per_document.parquet'),
            'specification_sha256':digest(output/'specification.json'),'analysis_code_sha256':digest(Path(__file__))}
    write_json(dest/'results.json',result)
    print(json.dumps({'phase':phase,'status':'complete','primary_slopes':primary['slopes'],
                      'numerical_gate_passed':primary['numerical_gate_passed']}),flush=True)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--freeze',action='store_true');ap.add_argument('--phase',choices=PHASES,action='append')
    ap.add_argument('--overlay',type=Path,action='append',default=[])
    args=ap.parse_args()
    if args.freeze:
        freeze(args.output)
    spec=json.loads((args.output/'specification.json').read_text())
    if spec['thresholds']!=list(THRESHOLDS):
        raise ValueError('specification thresholds differ from implementation')
    for phase in args.phase or []:
        analyze_phase(phase,args.output,args.overlay)


if __name__=='__main__':
    main()
