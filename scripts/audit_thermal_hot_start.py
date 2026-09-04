#!/usr/bin/env python3
"""Independent audit of all existing hot-start raw outputs; never reruns them."""
from __future__ import annotations
import argparse, hashlib, json, platform, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.simulation.thermal_pilot import exogenous_arrays
from scripts.run_thermal_closeout import onset
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--run',type=Path,required=True); ap.add_argument('--config',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
 if a.output.exists(): raise SystemExit('output exists')
 close=yaml.safe_load(a.config.read_text()); base_path=Path(close['base_config']); base=yaml.safe_load(base_path.read_text()); summaries=pd.read_csv(a.run/'summary.csv'); checks=[]; manifest=[]
 paths=sorted((a.run/'checkpoints').glob('*')); expected_files=18*2
 for p in paths: manifest.append({'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size})
 for p in sorted((a.run/'checkpoints').glob('*.ticks.csv')):
  f=pd.read_csv(p); first=f.iloc[0]; seed=int(first.seed); case=p.name.split('_seed')[0]; condition=str(first.condition); spec=close['cases'][case]; arrays=exogenous_arrays(seed,0,base)
  expected=f.true_temperature_c+base['plant']['thermal_exchange_per_step']*(f.ambient_c-f.true_temperature_c)+f.load_c_per_step-base['plant']['fan_gain_c_per_step']*f.fan_after+arrays['process_noise']
  summary_path=p.with_name(p.name.replace('.ticks.csv','.summary.json')); s=json.loads(summary_path.read_text()); temps=np.r_[f.true_temperature_c.to_numpy(),f.next_true_temperature_c.iloc[-1]]; dt=base['time']['dt_seconds']/60; sla=close['recovery']['sla_temperature_c']; excess=np.maximum(temps-sla,0); rec=onset(temps,sla,close['recovery']['sustain_ticks']); removal=int(spec['common_bias_end_tick']); rec_after=onset(temps,sla,close['recovery']['sustain_ticks'],removal) if removal else rec
  contract=((f.fan_after<base['control']['fan_min'])|(f.fan_after>base['control']['fan_max'])|((f.fan_after-f.fan_before).abs()>base['control']['fan_max_slew_per_step']+1e-12))
  runtime=(f.runtime_snapshot_id.isna()|f.runtime_proposal_id.isna()|f.executed_action.isna())
  metrics=(np.isclose((excess>0).sum()*dt,s['duration_above_sla_including_terminal_min']) and np.isclose(excess.max(),s['peak_excess_including_terminal_degC']) and np.isclose(excess.sum()*dt,s['integrated_excess_including_terminal_degC_min']) and np.isclose(temps[-1],s['terminal_temperature_c']))
  checks.append({'file':p.name,'case':case,'seed':seed,'condition':condition,'ticks':len(f),'temperature_continuity':bool(np.allclose(f.next_true_temperature_c.iloc[:-1],f.true_temperature_c.iloc[1:])),
   'plant_equation':bool(np.allclose(expected,f.next_true_temperature_c)),'fan_continuity':bool(np.allclose(f.fan_after.iloc[:-1],f.fan_before.iloc[1:])),
   'finite':bool(np.isfinite(f[['true_temperature_c','next_true_temperature_c','estimate_c','fan_before','fan_after','ambient_c','load_c_per_step']]).all().all()),
   'terminal_state_match':bool(np.isclose(temps[-1],s['terminal_temperature_c'])),'metrics_match':bool(metrics),'recovery_onset_match':(None if rec is None else rec*dt)==s['safety_recovery_onset_min'],
   'recovery_after_removal_match':(None if rec_after is None else (rec_after-removal)*dt)==s['recovery_after_removal_min'],
   'sustained_recovery_match':int((temps[(rec or 0)+close['recovery']['sustain_ticks']:] > sla).sum())==s['subsequent_violation_ticks'],
   'actuator_contract_violations':int(contract.sum()),'runtime_failures':int(runtime.sum()),'paired_input_sha256':s['paired_input_sha256']})
 a.output.mkdir(parents=True); c=pd.DataFrame(checks); c.to_csv(a.output/'audit_by_run.csv',index=False); pd.DataFrame(manifest).to_csv(a.output/'raw_manifest.csv',index=False)
 pairing=bool(c.groupby(['seed','condition']).paired_input_sha256.nunique().eq(1).all() and c.groupby('seed').paired_input_sha256.nunique().eq(1).all())
 audit={'files':len(c),'expected_files':18,'raw_artifacts':len(manifest),'ticks':int(c.ticks.sum()),'pairing':pairing,'all_continuity':bool(c.temperature_continuity.all() and c.fan_continuity.all()),'all_plant_equation':bool(c.plant_equation.all()),'all_finite':bool(c.finite.all()),'all_terminal_state_match':bool(c.terminal_state_match.all()),'all_metrics_match':bool(c.metrics_match.all()),'all_recovery_match':bool(c.recovery_onset_match.all() and c.recovery_after_removal_match.all() and c.sustained_recovery_match.all()),'actuator_contract_violations':int(c.actuator_contract_violations.sum()),'runtime_failures':int(c.runtime_failures.sum())}; audit['passed']=len(c)==18 and all(v for k,v in audit.items() if k.startswith('all_')) and pairing and audit['actuator_contract_violations']==0 and audit['runtime_failures']==0
 (a.output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n'); commit=subprocess.run(['git','rev-parse','HEAD'],check=True,text=True,capture_output=True).stdout.strip(); prov={'run_id':a.output.name,'source_code_commit':commit,'command':' '.join(sys.argv),'runner_sha256':sha(__file__),'closeout_config_sha256':sha(a.config),'base_config_sha256':sha(base_path),'raw_manifest_sha256':sha(a.output/'raw_manifest.csv'),'python':platform.python_version(),'interpreter':sys.executable,'numpy':np.__version__,'pandas':pd.__version__,'rerun_condition_runs':0,'llm_calls':0,'gpu_calls':0}; (a.output/'provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
if __name__=='__main__': main()
