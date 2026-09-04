#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.simulation.thermal_pilot import exogenous_arrays

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run',type=Path,required=True); ap.add_argument('--config',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    if a.output.exists(): raise SystemExit('output exists')
    a.output.mkdir(parents=True); (a.output/'tick_extracts').mkdir(); cfg=yaml.safe_load(a.config.read_text()); summary=pd.read_csv(a.run/'summary.csv'); checks=[]; manifest=[]; ranges=[]; details=[]; windows=[]
    selected={('nominal','none'),('drift','high'),('freeze','low'),('spike','high')}; extracted=set(); max_reject=summary.loc[summary.verifier_rejects.idxmax()]
    for p in sorted((a.run/'checkpoints').glob('*.ticks.csv')):
        f=pd.read_csv(p); first=f.iloc[0]; seed=int(first.seed); idx=int(first.scenario_index); arrays=exogenous_arrays(seed,idx,cfg); plant=cfg['plant']; expected=f.true_temperature_c+plant['thermal_exchange_per_step']*(f.ambient_c-f.true_temperature_c)+f.load_c_per_step-plant['fan_gain_c_per_step']*f.fan_after+arrays['process_noise']; key=(seed,idx,first.condition); s=summary[(summary.seed==seed)&(summary.scenario_index==idx)&(summary.condition==first.condition)].iloc[0]; temps=f.true_temperature_c.to_numpy(); fans=f.fan_after.to_numpy(); dt=cfg['time']['dt_seconds']/60; excess=np.maximum(temps-cfg['control']['sla_temperature_c'],0)
        metric_ok=np.isclose(np.abs(temps-cfg['control']['target_temperature_c']).sum()*dt,s.iae_degC_min) and np.isclose(excess.sum()*dt,s.integrated_excess_degC_min) and np.isclose(fans.sum()*dt,s.fan_effort_normalized_min)
        required=f[['true_temperature_c','next_true_temperature_c','estimate_c','fan_before','fan_after','ambient_c','load_c_per_step']]
        checks.append({'file':p.name,'ticks':len(f),'temperature_continuity':bool(np.allclose(f.next_true_temperature_c.iloc[:-1],f.true_temperature_c.iloc[1:])),'plant_equation':bool(np.allclose(expected,f.next_true_temperature_c)),'fan_continuity':bool(np.allclose(f.fan_after.iloc[:-1],f.fan_before.iloc[1:])),'finite':bool(np.isfinite(required).all().all()),'metrics_match':bool(metric_ok),'bounds_slew':bool(((f.fan_after>=0)&(f.fan_after<=1)&((f.fan_after-f.fan_before).abs()<=.2+1e-12)).all())})
        manifest.append({'file':p.name,'sha256':sha(p),'bytes':p.stat().st_size,'ticks':len(f)})
        effective_end=104 if first.attack=='freeze' and first.intensity=='low' else (144 if first.attack!='nominal' else None)
        ranges.append({'seed':seed,'scenario_index':idx,'attack':first.attack,'intensity':first.intensity,'condition':first.condition,'min_true_c':temps.min(),'max_true_c':temps.max(),'min_margin_to_sla_c':cfg['control']['sla_temperature_c']-temps.max(),'target_return_min':s.recovery_time_min,'effective_attack_end_exclusive':effective_end})
        details.append({'attack':first.attack,'intensity':first.intensity,'condition':first.condition,'ticks':len(f),'mean_confidence':f.confidence.mean(),'min_confidence':f.confidence.min(),'anomaly_ticks':int((f.anomaly_count>0).sum()),'crp_rounds':int(f.crp_rounds.sum()),'crp_nonconverged':int((f.crp_converged==False).sum()),'crp_excluded_sum':int(f.crp_excluded.sum()),'rejections':int((f.verifier_accepted==False).sum()),'rejection_reasons':';'.join(sorted(x for x in f.rejection_reasons.dropna().unique() if x))})
        estart=84 if first.attack!='nominal' else None; eend=effective_end
        active=f[(f.tick>=estart)&(f.tick<eend)] if estart is not None else f.iloc[0:0]
        windows.append({'seed':seed,'scenario_index':idx,'attack':first.attack,'intensity':first.intensity,'condition':first.condition,'effective_start':estart,'effective_end_exclusive':eend,'effective_ticks':len(active),'alarm_in_effective_window':bool((active.anomaly_count>0).any()),'alarm_ticks_effective':int((active.anomaly_count>0).sum()),'alarm_ticks_pre_attack':int((f[f.tick<84].anomaly_count>0).sum()),'alarm_ticks_transition_interval_60_119':int((f[(f.tick>=60)&(f.tick<120)].anomaly_count>0).sum())})
        marker=(str(first.attack),str(first.intensity)); is_max=(seed==int(max_reject.seed) and idx==int(max_reject.scenario_index) and first.condition==max_reject.condition)
        if (marker in selected and first.condition=='fpr_crp' and marker not in extracted) or is_max:
            shutil.copyfile(p,a.output/'tick_extracts'/p.name); extracted.add(marker)
    c=pd.DataFrame(checks); pd.DataFrame(manifest).to_csv(a.output/'raw_file_manifest.csv',index=False); pd.DataFrame(ranges).to_csv(a.output/'thermal_ranges_reentry.csv',index=False); pd.DataFrame(windows).to_csv(a.output/'effective_window_diagnostics.csv',index=False); pd.DataFrame(details).groupby(['attack','intensity','condition'],as_index=False).agg({'ticks':'sum','mean_confidence':'mean','min_confidence':'min','anomaly_ticks':'sum','crp_rounds':'sum','crp_nonconverged':'sum','crp_excluded_sum':'sum','rejections':'sum','rejection_reasons':lambda x:';'.join(sorted(set(';'.join(x).split(';'))-set([''])))}).to_csv(a.output/'diagnostics_by_attack.csv',index=False)
    c.to_csv(a.output/'raw_audit_by_run.csv',index=False); audit={'files':len(c),'ticks':int(c.ticks.sum()),'all_temperature_continuity':bool(c.temperature_continuity.all()),'all_plant_equation':bool(c.plant_equation.all()),'all_fan_continuity':bool(c.fan_continuity.all()),'all_finite':bool(c.finite.all()),'all_metrics_match':bool(c.metrics_match.all()),'all_bounds_slew':bool(c.bounds_slew.all()),'raw_manifest_entries':len(manifest),'extract_count':len(list((a.output/'tick_extracts').glob('*.csv'))),'note':'Sensor arrays were not stored in original raw logs; plant innovations were deterministically reconstructed from frozen seeds/configuration.'}; (a.output/'raw_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
if __name__=='__main__': main()
