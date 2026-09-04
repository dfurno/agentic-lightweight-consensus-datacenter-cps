#!/usr/bin/env python3
"""Audit historical impact before changing the production FPR tie-break."""
from __future__ import annotations
import argparse, hashlib, json, platform, sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.consensus.fpr import dominance_scores, fuzzy_preference_relation, reliability_scores
from src.consensus.owa import reliability_ordered_owa_weights
from src.simulation.thermal_pilot import _attack, exogenous_arrays

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--pilot',type=Path,required=True); ap.add_argument('--config',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    if a.output.exists(): raise SystemExit('output exists')
    cfg=yaml.safe_load(a.config.read_text()); records=[]; manifests=[]
    for p in sorted(a.pilot.glob('checkpoints/*fpr_crp.ticks.csv')):
        f=pd.read_csv(p); seed=int(f.seed.iloc[0]); si=int(f.scenario_index.iloc[0]); attack=f.attack.iloc[0]; label=f.intensity.iloc[0]
        arrays=exogenous_arrays(seed,si,cfg); history=[]; intensity=0 if attack=='nominal' else float(cfg['attacks']['definitions'][attack][label]); w=reliability_ordered_owa_weights(cfg['sensors']['count'],cfg['crp']['alpha'],cfg['crp']['beta'])
        for row in f.itertuples():
            clean=row.true_temperature_c+arrays['sensor_noise'][row.tick]; history.append(clean.copy()); x=_attack(clean,history,attack,intensity,row.tick,cfg,arrays['attacked'])
            d=dominance_scores(fuzzy_preference_relation(reliability_scores(x),10)); old=np.argsort(-d); new=np.lexsort((np.arange(len(d)),-d)); groups=[np.flatnonzero(d==v) for v in np.unique(d) if np.count_nonzero(d==v)>1]
            records.append({'file':p.name,'tick':row.tick,'has_exact_tie':bool(groups),'relevant_tie':any(np.unique(x[g]).size>1 for g in groups),'order_changed':not np.array_equal(old,new),'aggregate_delta_c':float(w@x[new]-w@x[old])})
        manifests.append({'path':str(p),'sha256':sha(p)})
    rng=np.random.default_rng(20260904); vectors=[np.array([.2,.8]),np.array([.5,.5,.1])]+[rng.normal(size=n) for n in (3,5,9) for _ in range(100)]; numeric=[]
    for rho in vectors:
        d=dominance_scores(fuzzy_preference_relation(rho,7)); numeric.append(np.array_equal(np.sign(rho[:,None]-rho[None,:]),np.sign(d[:,None]-d[None,:])))
    a.output.mkdir(parents=True); q=pd.DataFrame(records); q.to_csv(a.output/'historical_tie_scan.csv',index=False); q[q.order_changed].to_csv(a.output/'selective_impact.csv',index=False)
    result={'files':len(manifests),'ticks':len(q),'exact_tie_ticks':int(q.has_exact_tie.sum()),'relevant_tie_ticks':int(q.relevant_tie.sum()),'changed_order_ticks':int(q.order_changed.sum()),'changed_aggregate_ticks':int((q.aggregate_delta_c.abs()>1e-15).sum()),'maximum_absolute_aggregate_delta_c':float(q.aggregate_delta_c.abs().max()),'numeric_vectors':len(numeric),'numeric_ordering_pass':bool(all(numeric)),'gate':'STOP'}
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n'); (a.output/'provenance.json').write_text(json.dumps({'config_sha256':sha(a.config),'runner_sha256':sha(__file__),'input_manifest':manifests,'python':platform.python_version(),'interpreter':sys.executable,'numpy':np.__version__,'pandas':pd.__version__},indent=2)+'\n')
if __name__=='__main__': main()
