#!/usr/bin/env python3
"""Selective old-vs-stable-ID FPR ordering audit for paper-facing evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.consensus.crp import CRPState, _proximity
from src.consensus.fpr import dominance_scores, fuzzy_preference_relation, reliability_scores
from src.consensus.owa import reliability_ordered_owa_weights
from src.revision.artifacts import scenario_parts
from src.simulation.thermal_pilot import _attack, exogenous_arrays


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(values: np.ndarray, alpha: float, beta: float, stable_ids: np.ndarray, policy: str):
    reliability = reliability_scores(values)
    dominance = dominance_scores(fuzzy_preference_relation(reliability, 10.0))
    old = np.argsort(-dominance)
    new = np.lexsort((stable_ids, -dominance))
    order = old if policy == "legacy" else new
    weights = reliability_ordered_owa_weights(values.size, alpha, beta)
    ties = [np.flatnonzero(dominance == score) for score in np.unique(dominance)
            if np.count_nonzero(dominance == score) > 1]
    return float(weights @ values[order]), old, new, ties


def crp_tick(values: np.ndarray, state: CRPState, alpha: float, beta: float,
             alpha_crp: float, m_z: float, tau_c=.7, tau_r=.5, max_rounds=3,
             min_sensors=3, policy="legacy"):
    prediction = state.update_ewma(values)
    temporal = np.maximum(0.0, 1.0 - np.abs(values - prediction) / m_z)
    active = np.ones(values.size, dtype=bool)
    last_p = np.zeros(values.size)
    for _ in range(max_rounds):
        idx = np.flatnonzero(active)
        p_sub, degree = _proximity(values[idx], m_z)
        last_p = np.zeros(values.size); last_p[idx] = p_sub
        if degree >= tau_c or idx.size <= min_sensors:
            break
        weak = idx[(alpha_crp * p_sub + (1-alpha_crp) * temporal[idx]) < tau_r]
        if weak.size == 0 or idx.size - weak.size < min_sensors:
            break
        active[weak] = False
    idx = np.flatnonzero(active)
    estimate, old, new, ties = aggregate(values[idx], alpha, beta, idx, policy)
    return estimate, old, new, ties, idx


def metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    err = np.abs(pred-truth)
    return {"mae": float(err.mean()), "rmse": float(np.sqrt(np.mean((pred-truth)**2))),
            "median_absolute_error": float(np.median(err)),
            "p95_absolute_error": float(np.percentile(err, 95)), "max_absolute_error": float(err.max())}


def trace_values(path: Path):
    frame = pd.read_csv(path)
    cols = sorted((c for c in frame if c.startswith("sensor_")), key=lambda x: int(x.split("_")[1]))
    count = int(scenario_parts(path.stem).get("sensors_per_group", len(cols)))
    return frame, frame[cols[:count]].to_numpy(float), frame.temperature_ground_truth.to_numpy(float)


def scan_historical(family: str, scenarios: list[str], trace_dir: Path, crp: bool=False):
    rows=[]; derived=[]; manifests=[]
    for scenario in scenarios:
        path=trace_dir/f"{scenario}.csv"; frame, matrix, truth=trace_values(path); parts=scenario_parts(scenario)
        alpha=float(parts.get("alpha",.3)); beta=float(parts.get("beta",.8))
        old_pred=[]; new_pred=[]; old_state=CRPState(); new_state=CRPState()
        for tick, values in enumerate(matrix):
            if crp:
                old, oo, no, ties, retained=crp_tick(values,old_state,alpha,beta,.5,4.,policy="legacy")
                new, _, _, _, _=crp_tick(values,new_state,alpha,beta,.5,4.,policy="stable_id")
            else:
                retained=np.arange(values.size); old,oo,no,ties=aggregate(values,alpha,beta,retained,"legacy"); new,*_=aggregate(values,alpha,beta,retained,"stable_id")
            old_pred.append(old); new_pred.append(new)
            rows.append({"family":family,"case":scenario,"tick":tick,"exact_tie":bool(ties),
                         "tie_distinct_measurements":any(np.unique(values[retained][g]).size>1 for g in ties),
                         "order_changed":not np.array_equal(oo,no),"aggregate_delta":new-old})
        before=metrics(np.asarray(old_pred),truth); after=metrics(np.asarray(new_pred),truth)
        for key in before:
            derived.append({"family":family,"case":scenario,"metric":key,"before":before[key],"after":after[key],
                            "delta":after[key]-before[key],"display_3dp_changed":round(before[key],3)!=round(after[key],3)})
        manifests.append({"path":str(path),"sha256":sha256(path)})
    return rows,derived,manifests


def paper_mapping(derived: pd.DataFrame) -> pd.DataFrame:
    """Map every FPR-dependent manuscript cell/series to recomputed values."""
    historical={"main_campaign","robust_extension","historical_fpr_ablation","corrected_fpr_single","corrected_crp"}
    parts=pd.DataFrame([scenario_parts(x) if family in historical else {} for x,family in zip(derived.case,derived.family)],index=derived.index)
    enriched=pd.concat([derived,parts],axis=1)
    specs=[
      ("tab_consensus_perf","FPR--OWA global", "main_campaign",[],4),
      ("fig_heatmap_attack_method","FPR--OWA by attack", "main_campaign",["attack_type"],3),
      ("fig_mae_vs_beta","FPR--OWA by attack ratio", "main_campaign",["attack_ratio"],4),
      ("tab_sensors","FPR--OWA by sensors/zone", "main_campaign",["sensors_per_group"],4),
      ("OWA configuration audit","FPR--OWA by alpha,beta", "main_campaign",["alpha","beta"],4),
      ("tab_robust","FPR--OWA ten-seed global", "robust_extension",[],3),
      ("ablation","full FPR--OWA global", "historical_fpr_ablation",[],3),
      ("tab_crp","single-round FPR--OWA", "corrected_fpr_single",[],4),
      ("tab_crp","FPR--OWA + CRP", "corrected_crp",[],4),
    ]
    rows=[]
    for artifact,label,family,groups,precision in specs:
      subset=enriched[enriched.family==family]
      iterator=[((),subset)] if not groups else subset.groupby(groups,dropna=False)
      for key,group in iterator:
       keys=key if isinstance(key,tuple) else (key,)
       suffix=", ".join(f"{name}={value}" for name,value in zip(groups,keys)) or "global"
       for metric,mgroup in group.groupby("metric"):
        before=float(mgroup.before.mean()); after=float(mgroup.after.mean())
        rows.append({"artifact":artifact,"cell_or_series":f"{label}; {suffix}; {metric}","source_family":family,
          "cases":int(mgroup.case.nunique()),"display_precision":precision,"before":before,"after":after,"delta":after-before,
          "before_display":f"{before:.{precision}f}","after_display":f"{after:.{precision}f}",
          "changes_at_display_precision":f"{before:.{precision}f}"!=f"{after:.{precision}f}"})
    # The robustness table also displays the between-seed SD of per-seed MAE.
    robust=enriched[(enriched.family=="robust_extension")&(enriched.metric=="mae")]
    seedmeans=robust.groupby("seed")[["before","after"]].mean()
    before=float(seedmeans.before.std(ddof=1)); after=float(seedmeans.after.std(ddof=1))
    rows.append({"artifact":"tab_robust","cell_or_series":"FPR--OWA MAE between-seed std","source_family":"robust_extension",
      "cases":int(robust.case.nunique()),"display_precision":3,"before":before,"after":after,"delta":after-before,
      "before_display":f"{before:.3f}","after_display":f"{after:.3f}","changes_at_display_precision":f"{before:.3f}"!=f"{after:.3f}"})
    return pd.DataFrame(rows)


def scan_saved_ticks(family: str, paths: list[Path], cfg: dict):
    rows=[]; derived=[]; manifests=[]
    for path in paths:
        frame=pd.read_csv(path); old_pred=[]; new_pred=[]; old_state=CRPState(lam=cfg['crp']['ewma_lambda']); history=[]
        seed=int(frame.seed.iloc[0]); scenario_index=int(frame.scenario_index.iloc[0]); arrays=exogenous_arrays(seed,scenario_index,cfg)
        intensity=0 if frame.attack.iloc[0]=='nominal' else float(cfg['attacks']['definitions'][frame.attack.iloc[0]][frame.intensity.iloc[0]])
        for row in frame.itertuples():
            if hasattr(row,'observed_sensor_values'):
                values=np.asarray(json.loads(row.observed_sensor_values),float)
            else:
                clean=row.true_temperature_c+arrays['sensor_noise'][row.tick]; history.append(clean.copy())
                values=_attack(clean,history,row.attack,intensity,row.tick,cfg,arrays['attacked'])
            if hasattr(row,'crp_excluded_flags'):
                retained=np.flatnonzero(~np.asarray(json.loads(row.crp_excluded_flags),bool))
            else:
                _,_,_,_,retained=crp_tick(values,old_state,.3,.8,.5,4.,policy='legacy')
            old,oo,no,ties=aggregate(values[retained],.3,.8,retained,"legacy")
            new,*_=aggregate(values[retained],.3,.8,retained,"stable_id")
            old_pred.append(old); new_pred.append(new)
            rows.append({"family":family,"case":path.stem,"tick":int(row.tick),"exact_tie":bool(ties),
                         "tie_distinct_measurements":any(np.unique(values[retained][g]).size>1 for g in ties),
                         "order_changed":not np.array_equal(oo,no),"aggregate_delta":new-old})
        truth=frame.true_temperature_c.to_numpy(float); before=metrics(np.asarray(old_pred),truth); after=metrics(np.asarray(new_pred),truth)
        for key in before:
            derived.append({"family":family,"case":path.stem,"metric":key,"before":before[key],"after":after[key],
                            "delta":after[key]-before[key],"display_3dp_changed":round(before[key],3)!=round(after[key],3)})
        manifests.append({"path":str(path),"sha256":sha256(path)})
    return rows,derived,manifests


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--original",type=Path,required=True); ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--thermal-pilot",type=Path,required=True); ap.add_argument("--hot-start",type=Path,required=True); args=ap.parse_args()
    if args.output.exists(): raise SystemExit("output exists")
    metrics_frame=pd.read_csv(args.original/"results/metrics.csv")
    main_scenarios=sorted(metrics_frame.loc[metrics_frame.method=="fpr_owa","scenario"].unique())
    corrected=pd.read_csv("revision/evidence/20260903_bias_selective_replay/recomposed/crp_estimation_per_scenario.csv")
    crp_scenarios=sorted(corrected.loc[corrected.method=="crp","scenario"].unique())
    ablation=pd.read_csv("revision/evidence/20260903_bias_selective_replay/dependents/fpr_ablation.csv")
    ablation_scenarios=sorted(ablation.scenario.unique())
    all_rows=[]; all_derived=[]; manifests=[]
    robust=pd.read_csv(args.original/"outputs/seed_extension_metrics.csv")
    robust_scenarios=sorted(robust.loc[robust.method=="fpr_owa","scenario"].unique())
    for payload in [scan_historical("main_campaign",main_scenarios,args.original/"results/traces"),
                    scan_historical("robust_extension",robust_scenarios,args.original/"results/traces"),
                    scan_historical("corrected_fpr_single",crp_scenarios,args.original/"results/traces"),
                    scan_historical("corrected_crp",crp_scenarios,args.original/"results/traces",True),
                    scan_historical("historical_fpr_ablation",ablation_scenarios,args.original/"results/traces")]:
        r,d,m=payload; all_rows+=r; all_derived+=d; manifests+=m
    thermal_cfg=yaml.safe_load(Path('configs/thermal_pilot.yaml').read_text())
    pilot_paths=sorted(args.thermal_pilot.glob("checkpoints/*fpr_crp.ticks.csv"))
    hot_paths=sorted(args.hot_start.glob("checkpoints/*fpr_crp.ticks.csv"))
    for payload in [scan_saved_ticks("thermal_pilot",pilot_paths,thermal_cfg),scan_saved_ticks("hot_start",hot_paths,thermal_cfg)]:
        r,d,m=payload; all_rows+=r; all_derived+=d; manifests+=m
    args.output.mkdir(parents=True)
    ticks=pd.DataFrame(all_rows); derived=pd.DataFrame(all_derived)
    derived.to_csv(args.output/"all_derived_metrics.csv",index=False)
    paper_mapping(derived).to_csv(args.output/"paper_artifact_mapping.csv",index=False)
    ticks.loc[ticks.order_changed | (ticks.aggregate_delta.abs()>1e-15)].to_csv(args.output/"changed_ticks.csv",index=False)
    derived.loc[(derived.delta.abs()>1e-15) | derived.display_3dp_changed].to_csv(args.output/"changed_derived_metrics.csv",index=False)
    summary=ticks.groupby("family").agg(cases=("case","nunique"),ticks=("tick","size"),exact_tie_ticks=("exact_tie","sum"),
        ties_with_distinct_measurements=("tie_distinct_measurements","sum"),changed_order_ticks=("order_changed","sum"),
        changed_aggregate_ticks=("aggregate_delta",lambda x:int((x.abs()>1e-15).sum())),maximum_absolute_delta=("aggregate_delta",lambda x:float(x.abs().max()))).reset_index()
    dm=derived.groupby("family").agg(derived_metrics_changed=("delta",lambda x:int((x.abs()>1e-15).sum())),
        displayed_3dp_values_changed=("display_3dp_changed","sum")).reset_index()
    summary.merge(dm,on="family").to_csv(args.output/"family_summary.csv",index=False)
    derived.groupby(["family","metric"],as_index=False).agg(before=("before","mean"),after=("after","mean")).assign(
        delta=lambda x:x.after-x.before,display_3dp_changed=lambda x:x.before.round(3)!=x.after.round(3)).to_csv(args.output/"aggregate_metrics_before_after.csv",index=False)
    unique={m["path"]:m for m in manifests}
    (args.output/"input_manifest.json").write_text(json.dumps(list(unique.values()),indent=2)+"\n")
    commit=subprocess.run(["git","rev-parse","HEAD"],check=True,text=True,capture_output=True).stdout.strip()
    provenance={"run_id":args.output.name,"source_code_commit":commit,"command":" ".join(sys.argv),"runner_sha256":sha256(Path(__file__)),
        "input_manifest_sha256":sha256(args.output/"input_manifest.json"),"python":platform.python_version(),"interpreter":sys.executable,
        "numpy":np.__version__,"pandas":pd.__version__,"dependencies":{"scipy":__import__("scipy").__version__},"llm_calls":0,"gpu_calls":0}
    (args.output/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")

if __name__=="__main__": main()
