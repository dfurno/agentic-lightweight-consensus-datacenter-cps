#!/usr/bin/env python3
"""Preregistered diagnostic CRP replay over fixed thermal-pilot trajectories."""
from __future__ import annotations
import argparse, hashlib, json, platform, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.consensus.crp import CRPState, _proximity
from src.consensus.fpr import dominance_scores, fuzzy_preference_relation, reliability_scores
from src.consensus.owa import reliability_ordered_owa_weights
from src.simulation.thermal_pilot import _attack, exogenous_arrays

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def agg(x,a,b,stable_ids,policy='stable_id'):
    d=dominance_scores(fuzzy_preference_relation(reliability_scores(x),10)); order=np.argsort(-d) if policy=='legacy' else np.lexsort((np.asarray(stable_ids),-d))
    return float(reliability_ordered_owa_weights(len(x),a,b)@x[order])

def dynamic_prediction(previous_estimate,cfg,previous_inputs):
    """Predict observation t from estimate t-1 and transition inputs t-1.

    No transition exists at tick zero, represented by ``previous_inputs=None``.
    """
    if previous_inputs is None:
        return float(previous_estimate)
    ambient,load,fan=previous_inputs
    return float(previous_estimate+cfg['exchange']*(ambient-previous_estimate)+load-cfg['fan_gain']*fan)

def step(values,state,cfg,variant,previous_estimate,previous_inputs,ordering='stable_id'):
    pred=state.update_ewma(values); temporal=np.maximum(0,1-np.abs(values-pred)/cfg['m_z_c'])
    predicted=dynamic_prediction(previous_estimate,cfg,previous_inputs)
    dynamic=np.maximum(0,1-np.abs(values-predicted)/cfg['m_z_c'])
    active=np.ones(len(values),bool); last_p=np.zeros(len(values)); converged=False
    for rounds in range(1,cfg['max_rounds']+1):
        idx=np.flatnonzero(active); p,C=_proximity(values[idx],cfg['m_z_c']); last_p=np.zeros(len(values)); last_p[idx]=p
        if C>=cfg['tau_c'] or len(idx)<=cfg['min_sensors']:
            converged=C>=cfg['tau_c']; break
        if variant=='temporal_only': rsub=temporal[idx]
        elif variant=='social_only': rsub=p
        elif variant=='published_blend': rsub=cfg['alpha_crp']*p+(1-cfg['alpha_crp'])*temporal[idx]
        else: rsub=cfg['alpha_crp']*p+(1-cfg['alpha_crp'])*dynamic[idx]
        weak=idx[rsub<cfg['tau_r']]
        if not len(weak) or len(idx)-len(weak)<cfg['min_sensors']: break
        active[weak]=False
    if variant=='temporal_only': reliability=temporal
    elif variant=='social_only': reliability=last_p
    elif variant=='published_blend': reliability=cfg['alpha_crp']*last_p+(1-cfg['alpha_crp'])*temporal
    else: reliability=cfg['alpha_crp']*last_p+(1-cfg['alpha_crp'])*dynamic
    idx=np.flatnonzero(active); flags=reliability<cfg['tau_r']
    state.below_streak=np.where(flags,state.below_streak+1,0.0); persistent=state.below_streak>=cfg['persistence_l']
    confidence=float(reliability[idx].mean()) if len(idx) else 0.; confidence=confidence if converged else confidence*.5
    dom=dominance_scores(fuzzy_preference_relation(reliability_scores(values[idx]),10))
    exact_tie=any(np.count_nonzero(dom==score)>1 for score in np.unique(dom))
    ordered_dom=np.sort(dom); minimum_dominance_gap=float(np.min(np.diff(ordered_dom))) if len(dom)>1 else float('inf')
    return {'estimate':agg(values[idx],cfg['alpha'],cfg['beta'],idx,ordering),'confidence':confidence,'flags':flags,'persistent':persistent,
            'excluded':~active,'rounds':rounds,'converged':converged,'dynamic_prediction':predicted,'exact_tie':exact_tie,'minimum_dominance_gap':minimum_dominance_gap}

def safe_div(a,b): return float(a/b) if b else None
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); z=ap.parse_args()
    if z.output.exists(): raise SystemExit('output exists')
    pre=yaml.safe_load(z.config.read_text()); base_path=Path(pre['base_config']); base=yaml.safe_load(base_path.read_text()); paths=sorted(Path(pre['input_run']).glob('checkpoints/*fpr_crp.ticks.csv'))
    if len(paths)!=pre['maximum_trajectories']: raise SystemExit('trajectory budget mismatch')
    c={**base['crp'],'exchange':base['plant']['thermal_exchange_per_step'],'fan_gain':base['plant']['fan_gain_c_per_step']}
    detail=[]; gates=[]; manifest=[]
    for path in paths:
        f=pd.read_csv(path); seed=int(f.seed.iloc[0]); si=int(f.scenario_index.iloc[0]); arrays=exogenous_arrays(seed,si,base); history=[]
        observed_reconstructed=[]; intensity=0 if f.attack.iloc[0]=='nominal' else float(base['attacks']['definitions'][f.attack.iloc[0]][f.intensity.iloc[0]])
        for row in f.itertuples():
            clean=row.true_temperature_c+arrays['sensor_noise'][row.tick]; history.append(clean.copy()); observed_reconstructed.append(_attack(clean,history,row.attack,intensity,row.tick,base,arrays['attacked']))
        observed_reconstructed=np.asarray(observed_reconstructed)
        if 'observed_sensor_values' in f:
            observed_saved=np.asarray([json.loads(x) for x in f.observed_sensor_values]); obs_delta=float(np.max(np.abs(observed_reconstructed-observed_saved)))
        else:
            obs_delta=np.nan
        for variant in pre['variants']:
            state=CRPState(lam=c['ewma_lambda']); previous=float(f.estimate_c.iloc[0]); previous_inputs=None
            legacy_state=CRPState(lam=c['ewma_lambda']); legacy_previous=previous
            tp=fp=fn=tn=0; false_transition=0; nonconv=excluded=rounds_total=0; errors=[]
            for i,row in enumerate(f.itertuples()):
                r=step(observed_reconstructed[i],state,c,variant,previous,previous_inputs); previous=r['estimate']
                legacy_r=None
                if variant=='published_blend':
                    legacy_r=step(observed_reconstructed[i],legacy_state,c,variant,legacy_previous,previous_inputs,ordering='legacy')
                    legacy_previous=legacy_r['estimate']
                previous_inputs=(row.ambient_c,row.load_c_per_step,row.fan_after)
                mask=np.zeros(base['sensors']['count'],bool)
                if row.attack!='nominal' and base['attacks']['start_tick']<=row.tick<base['attacks']['end_tick_exclusive']:
                    mask[arrays['attacked']]=not (row.attack=='freeze' and row.tick-base['attacks']['start_tick']>=int(intensity))
                flags=r['persistent']; tp+=int((flags&mask).sum()); fp+=int((flags&~mask).sum()); fn+=int((~flags&mask).sum()); tn+=int((~flags&~mask).sum())
                transition=base['plant']['transition_start_tick']<=row.tick<base['plant']['transition_end_tick']; false_transition+=int(transition and not mask.any() and flags.any())
                errors.append(abs(r['estimate']-row.true_temperature_c)); nonconv+=int(not r['converged']); excluded+=int(r['excluded'].sum()); rounds_total+=r['rounds']
                if variant=='published_blend':
                    gates.append({'file':path.name,'tick':row.tick,'estimate_delta':legacy_r['estimate']-row.estimate_c,'confidence_delta':legacy_r['confidence']-row.confidence,
                      'persistent_match':None if not hasattr(row,'crp_persistent_flags') else legacy_r['persistent'].tolist()==json.loads(row.crp_persistent_flags),
                      'excluded_match':int(legacy_r['excluded'].sum())==row.crp_excluded,'rounds_match':legacy_r['rounds']==row.crp_rounds,'observation_max_delta':obs_delta,'exact_tie':legacy_r['exact_tie'],'minimum_dominance_gap':legacy_r['minimum_dominance_gap']})
            detail.append({'file':path.name,'variant':variant,'ticks':len(f),'mae_c':float(np.mean(errors)),'precision':safe_div(tp,tp+fp),'recall':safe_div(tp,tp+fn),
              'true_positive_reporter_ticks':tp,'false_positive_reporter_ticks':fp,'false_negative_reporter_ticks':fn,'true_negative_reporter_ticks':tn,
              'false_alarm_transition_ticks':false_transition,'nonconverged_ticks':nonconv,'reporter_exclusions':excluded,'total_rounds':rounds_total})
        fan_continuity=bool(np.allclose(f.fan_after.iloc[:-1].to_numpy(float),f.fan_before.iloc[1:].to_numpy(float),rtol=0,atol=1e-12))
        gates.append({'file':path.name,'tick':-1,'estimate_delta':0.0,'confidence_delta':0.0,'persistent_match':None,
          'excluded_match':True,'rounds_match':True,'observation_max_delta':obs_delta,'exact_tie':False,
          'minimum_dominance_gap':float('inf'),'fan_state_continuity':fan_continuity})
        manifest.append({'path':str(path),'sha256':sha(path)})
    z.output.mkdir(parents=True); d=pd.DataFrame(detail); g=pd.DataFrame(gates); d.to_csv(z.output/'trajectory_variant_metrics.csv',index=False); g.to_csv(z.output/'published_gate.csv',index=False)
    d.groupby('variant',as_index=False).agg(trajectories=('file','nunique'),ticks=('ticks','sum'),mean_mae_c=('mae_c','mean'),nonconverged_ticks=('nonconverged_ticks','sum'),reporter_exclusions=('reporter_exclusions','sum'),total_rounds=('total_rounds','sum'),false_alarm_transition_ticks=('false_alarm_transition_ticks','sum')).to_csv(z.output/'variant_summary.csv',index=False)
    comparison=g.tick>=0; near_tie=(g.minimum_dominance_gap<=1e-12)&comparison
    # The saved published fields used legacy unstable tie ordering.  This gate is
    # deliberately separate from the four stable-ID comparison variants.
    gate={'rows':int(comparison.sum()),'observations_saved_in_original':bool(g.observation_max_delta.notna().any()),'observations_match':None if not g.observation_max_delta.notna().any() else bool((g.observation_max_delta.dropna()<=1e-12).all()),'estimate_match_all_ticks':bool((g.loc[comparison,'estimate_delta'].abs()<=1e-12).all()),'estimate_match_outside_numeric_ties':bool((g.loc[comparison&~near_tie,'estimate_delta'].abs()<=1e-12).all()),'numeric_tie_affected_estimate_mismatches':int((near_tie & (g.estimate_delta.abs()>1e-12)).sum()),'numeric_tie_tolerance':1e-12,'confidence_match':bool((g.loc[comparison,'confidence_delta'].abs()<=1e-12).all()),'persistent_flags_recoverable':bool(g.persistent_match.notna().any()),'persistent_match':None if not g.persistent_match.notna().any() else bool(g.persistent_match.dropna().all()),'excluded_count_match':bool(g.loc[comparison,'excluded_match'].all()),'rounds_match':bool(g.loc[comparison,'rounds_match'].all()),'fan_after_t_minus_1_equals_fan_before_t':bool(g.loc[g.tick==-1,'fan_state_continuity'].all())}; gate['passed']=gate['estimate_match_outside_numeric_ties'] and gate['confidence_match'] and gate['excluded_count_match'] and gate['rounds_match'] and gate['fan_after_t_minus_1_equals_fan_before_t']
    (z.output/'gate.json').write_text(json.dumps(gate,indent=2)+'\n'); (z.output/'input_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    commit=subprocess.run(['git','rev-parse','HEAD'],check=True,text=True,capture_output=True).stdout.strip(); prov={'run_id':pre['run_id'],'source_code_commit':commit,'command':' '.join(sys.argv),'config_sha256':sha(z.config),'base_config_sha256':sha(base_path),'runner_sha256':sha(__file__),'input_manifest_sha256':sha(z.output/'input_manifest.json'),'python':platform.python_version(),'interpreter':sys.executable,'numpy':np.__version__,'pandas':pd.__version__,'llm_calls':0,'gpu_calls':0,'limitations':['offline diagnostic replay on fixed physical trajectories and actions','not a counterfactual closed-loop comparison','published raw lacks combined reliability and instantaneous anomaly flags']}; (z.output/'provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
if __name__=='__main__': main()
