#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, platform, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.simulation.thermal_pilot import exogenous_arrays, run_condition

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def onset(values, threshold, sustain, start=0):
    for i in range(start, len(values)-sustain+1):
        if np.all(values[i:i+sustain] <= threshold): return i
    return None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--commit',required=True); a=ap.parse_args()
    actual=subprocess.run(['git','rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip()
    if actual!=a.commit: raise SystemExit('commit mismatch')
    if a.output.exists(): raise SystemExit('output exists')
    close=yaml.safe_load(a.config.read_text()); base_path=Path(close['base_config']); base=yaml.safe_load(base_path.read_text())
    if len(close['seeds'])*len(close['conditions'])*len(close['cases']) != 18 or close['maximum_condition_runs'] != 18: raise SystemExit('budget mismatch')
    a.output.mkdir(parents=True); (a.output/'checkpoints').mkdir(); summaries=[]; input_hashes={}
    for seed in close['seeds']:
        arrays=exogenous_arrays(seed,0,base)
        ih=hashlib.sha256(arrays['process_noise'].tobytes()+arrays['sensor_noise'].tobytes()+arrays['attacked'].tobytes()).hexdigest(); input_hashes[str(seed)]=ih
        for case,spec in close['cases'].items():
            for condition in close['conditions']:
                rows,s=run_condition(seed,0,'nominal',None,condition,base,arrays,initial_temperature_c=spec['initial_temperature_c'],common_bias_c=spec['common_bias_c'],common_bias_end_tick=spec['common_bias_end_tick'])
                f=pd.DataFrame(rows); temps=np.r_[f.true_temperature_c.to_numpy(),f.next_true_temperature_c.iloc[-1]]; dt=base['time']['dt_seconds']/60; sla=close['recovery']['sla_temperature_c']; sustain=close['recovery']['sustain_ticks']; excess=np.maximum(temps-sla,0); rec=onset(temps,sla,sustain); removal=spec['common_bias_end_tick']; rec_after=onset(temps,sla,sustain,removal) if removal else rec
                target=np.abs(temps-close['recovery']['target_temperature_c'])<=close['recovery']['target_band_c']; target_on=None
                for i in range(len(target)-sustain+1):
                    if target[i:i+sustain].all(): target_on=i; break
                s.update({'case':case,'paired_input_sha256':ih,'safety_recovered':rec is not None,'safety_recovery_onset_min':None if rec is None else rec*dt,'recovery_after_removal_min':None if rec_after is None else (rec_after-removal)*dt,'subsequent_violation_ticks':0 if rec is None else int((temps[rec+sustain:]>sla).sum()),'target_return_onset_min':None if target_on is None else target_on*dt,'duration_above_sla_including_terminal_min':float((excess>0).sum()*dt),'peak_excess_including_terminal_degC':float(excess.max()),'integrated_excess_including_terminal_degC_min':float(excess.sum()*dt),'terminal_temperature_c':float(temps[-1])})
                name=f'{case}_seed{seed}_{condition}'; f.to_csv(a.output/'checkpoints'/f'{name}.ticks.csv',index=False); (a.output/'checkpoints'/f'{name}.summary.json').write_text(json.dumps(s,indent=2,allow_nan=False)+'\n'); summaries.append(s)
    pd.DataFrame(summaries).to_csv(a.output/'summary.csv',index=False)
    raw_frames=[pd.read_csv(p) for p in sorted((a.output/'checkpoints').glob('*.ticks.csv'))]
    actuator_contract_violations=sum(int(((f.fan_after < base['control']['fan_min']) |
        (f.fan_after > base['control']['fan_max']) |
        ((f.fan_after-f.fan_before).abs() > base['control']['fan_max_slew_per_step']+1e-12)).sum()) for f in raw_frames)
    runtime_failures=sum(int((f.runtime_snapshot_id.isna() | f.runtime_proposal_id.isna() |
        f.executed_action.isna()).sum()) for f in raw_frames)
    audit={'runs':len(summaries),'ticks':sum(x['ticks'] for x in summaries),'paired_between_cases':all(pd.DataFrame(summaries).groupby('seed').paired_input_sha256.nunique()==1),'actuator_contract_violations':actuator_contract_violations,'runtime_failures':runtime_failures}
    (a.output/'validity.json').write_text(json.dumps(audit,indent=2)+'\n')
    prov={'run_id':close['run_id'],'source_code_commit':actual,'commit_verified':True,'closeout_config_sha256':sha(a.config),'base_config_sha256':sha(base_path),'runner_sha256':sha(__file__),'input_hashes':input_hashes,'python':platform.python_version(),'interpreter':sys.executable,'completed_runs':len(summaries),'llm_calls':0,'gpu_calls':0}
    (a.output/'provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
if __name__=='__main__': main()
