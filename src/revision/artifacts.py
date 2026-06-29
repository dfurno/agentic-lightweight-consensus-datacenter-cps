from __future__ import annotations

import json
import math
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.consensus.baselines import hampel_filter_then_mean, median, mean, run_baseline, trimmed_mean
from src.consensus.fpr import reliability_scores, robust_center_scale
from src.consensus.fpr_owa import fpr_owa_consensus
from src.consensus.owa import direct_owa
from src.evaluation.statistics import bootstrap_ci
from src.utils.io import ensure_dir, read_yaml


RESULTS = Path("results")
OUTPUTS = Path("outputs")


@dataclass(frozen=True)
class RevisionPaths:
    results: Path = RESULTS
    outputs: Path = OUTPUTS


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, value: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def trace_files(results: Path = RESULTS) -> list[Path]:
    return sorted((results / "traces").glob("*.csv"))


def metadata_files(results: Path = RESULTS) -> list[Path]:
    return sorted((results / "traces").glob("*.metadata.json"))


def sensor_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in frame.columns if col.startswith("sensor_")]


def label_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in frame.columns if col.startswith("attack_label_")]


def scenario_from_trace(path: Path) -> str:
    return path.stem


def metric_schema(frame: pd.DataFrame) -> list[dict[str, str]]:
    return [{"column": col, "dtype": str(dtype)} for col, dtype in frame.dtypes.items()]


def scenario_parts(scenario: str) -> dict[str, object]:
    # Handles names like full_seed42_bias_0p2_noise0p25_s9_a0p3_b0p8.
    parts = scenario.split("_")
    out: dict[str, object] = {"scenario": scenario}
    try:
        seed_token = next(part for part in parts if part.startswith("seed"))
        out["seed"] = int(seed_token.removeprefix("seed"))
        idx = parts.index(seed_token)
        out["attack_type"] = parts[idx + 1]
        out["attack_ratio"] = float(parts[idx + 2].replace("p", "."))
        noise_token = next(part for part in parts if part.startswith("noise"))
        out["sensor_noise_level"] = float(noise_token.removeprefix("noise").replace("p", "."))
        sensor_token = next(part for part in parts if part.startswith("s") and part[1:].isdigit())
        out["sensors_per_group"] = int(sensor_token.removeprefix("s"))
        alpha_token = next(part for part in parts if part.startswith("a") and len(part) > 1)
        beta_token = next(part for part in parts if part.startswith("b") and len(part) > 1)
        out["alpha"] = float(alpha_token.removeprefix("a").replace("p", "."))
        out["beta"] = float(beta_token.removeprefix("b").replace("p", "."))
    except (StopIteration, ValueError, IndexError):
        pass
    return out


def inventory(paths: RevisionPaths = RevisionPaths()) -> None:
    out = ensure_dir(paths.outputs)
    results = paths.results
    tables = {
        "metrics": read_csv_if_exists(results / "metrics.csv"),
        "metrics_by_scenario": read_csv_if_exists(results / "metrics_by_scenario.csv"),
        "safety_metrics": read_csv_if_exists(results / "safety_metrics.csv"),
        "latency_metrics": read_csv_if_exists(results / "latency_metrics.csv"),
        "realtime_metrics": read_csv_if_exists(results / "realtime_metrics.csv"),
        "realtime_safety_metrics": read_csv_if_exists(results / "realtime_safety_metrics.csv"),
        "token_usage": read_csv_if_exists(results / "token_usage.csv"),
    }
    first_trace = trace_files(results)[0] if trace_files(results) else None
    trace_schema = metric_schema(pd.read_csv(first_trace, nrows=20)) if first_trace else []
    configs = {
        "experiments": read_yaml("configs/experiments.yaml") if Path("configs/experiments.yaml").exists() else {},
        "simulation": read_yaml("configs/simulation.yaml") if Path("configs/simulation.yaml").exists() else {},
        "sla": read_yaml("configs/sla.yaml") if Path("configs/sla.yaml").exists() else {},
        "llm": read_yaml("configs/llm.yaml") if Path("configs/llm.yaml").exists() else {},
    }
    full = configs["experiments"].get("full", {})
    grid_expected = (
        len(full.get("seeds", []))
        * len(full.get("attack_ratios", []))
        * len(full.get("attack_types", []))
        * len(full.get("sensor_noise_levels", []))
        * len(full.get("sensors_per_group", []))
        * len(full.get("owa_parameters", []))
    )
    trace_flags = {
        "ground_truth_attack_labels": any(item["column"].startswith("attack_label_") for item in trace_schema),
        "per_sensor_reliability": False,
        "anomaly_flag_set": False,
        "zone_confidence": False,
        "absolute_estimation_error": False,
        "recomputable_from_trace": bool(first_trace),
    }
    params = parameter_table(configs)
    params.to_csv(out / "params_table.csv", index=False)
    lines = [
        "# Major-revision inventory",
        "",
        "## Artifact locations",
    ]
    for name, frame in tables.items():
        path = results / f"{name}.csv"
        status = "present" if path.exists() else "missing"
        lines.append(f"- `{path}`: {status}; rows={len(frame)}; columns={list(frame.columns)}")
    lines.extend(
        [
            f"- `{results / 'traces'}`: {len(trace_files(results))} trace CSV files and {len(metadata_files(results))} metadata JSON files.",
            "",
            "## First trace schema",
            f"- source file: `{first_trace}`" if first_trace else "- source file: missing",
        ]
    )
    lines.extend(f"- {item['column']}: {item['dtype']}" for item in trace_schema)
    lines.extend(
        [
            "",
            "## FATAL diagnostic-data availability",
            f"- Ground-truth attacked-sensor labels: {trace_flags['ground_truth_attack_labels']}.",
            "- Per-sensor reliability, anomaly flags, confidence, and absolute error are not stored as raw columns in traces, "
            "but are recomputable from sensor readings, ground truth, and the configured FPR-OWA operator.",
            f"- Recompute from traces possible: {trace_flags['recomputable_from_trace']}.",
            "",
            "## Scenario grid",
            f"- Expected full-grid scenarios from config: {grid_expected}.",
            f"- Trace CSV files currently present: {len(trace_files(results))}.",
            f"- Metrics scenarios currently present: {tables['metrics']['scenario'].nunique() if 'scenario' in tables['metrics'] else 0}.",
            "",
            "## Latency-unit reconciliation",
            "The `latency_metrics.csv` consensus latency is a scenario-level wall-clock loop over all ticks for a method, "
            "including Python iteration overhead. Realtime `mean_loop_latency_ms`, `p95_loop_latency_ms`, and "
            "`p99_loop_latency_ms` are per-control-tick measurements inside the event-triggered loop. Therefore, "
            "operator-level seconds in Fig. 3 and sub-millisecond p95 loop latency in the realtime table do not time "
            "the same code region. See `outputs/latency_reconciliation.md` after Phase B for the numeric summary.",
            "",
            "## Environment",
            f"- Python: {platform.python_version()}",
            f"- Platform: {platform.platform()}",
        ]
    )
    write_text(out / "_inventory.md", "\n".join(lines) + "\n")


def parameter_table(configs: dict[str, dict]) -> pd.DataFrame:
    simulation = configs.get("simulation", {})
    experiments = configs.get("experiments", {})
    sla = configs.get("sla", {})
    llm = configs.get("llm", {})
    rows: list[dict[str, object]] = []

    def add(name: str, value: object, group: str, note: str = "") -> None:
        rows.append({"parameter": name, "value": json.dumps(value) if isinstance(value, (dict, list)) else value, "group": group, "note": note})

    add("fpr_kappa", 10.0, "consensus", "Default in fpr_owa_consensus unless explicitly swept.")
    add("anomaly_floor_tau", 0.15, "consensus", "max(0.15, median(reliability)-2*std(reliability)).")
    for params in experiments.get("full", {}).get("owa_parameters", []):
        add("owa_alpha_beta", params, "consensus")
    add("thermal", simulation.get("thermal", {}), "thermal_model")
    add("attacks", simulation.get("attacks", {}), "attack_injection")
    add("temperature_sla_c", sla.get("temperature_sla_c"), "verifier")
    add("emergency_temperature_c", sla.get("emergency_temperature_c"), "verifier")
    add("min_confidence", sla.get("min_confidence"), "verifier")
    add("max_consensus_age_seconds", sla.get("max_consensus_age_seconds"), "verifier")
    add("actuators", sla.get("actuators", {}), "verifier")
    add("prediction", sla.get("prediction", {}), "verifier")
    add("realtime_llm_policy", experiments.get("realtime", {}).get("llm_policy", {}), "event_trigger")
    add("realtime_deadlines_ms", experiments.get("realtime", {}).get("deadlines_ms", []), "event_trigger")
    add("llm_model", llm.get("primary_model"), "llm")
    add("llm_base_url", llm.get("base_url"), "llm")
    add("llm_timeout_seconds", llm.get("timeout_seconds"), "llm")
    return pd.DataFrame(rows)


def reproduce(paths: RevisionPaths = RevisionPaths()) -> None:
    out = ensure_dir(paths.outputs)
    results = paths.results
    metrics = read_csv_if_exists(results / "metrics.csv")
    safety = read_csv_if_exists(results / "safety_metrics.csv")
    latency = read_csv_if_exists(results / "latency_metrics.csv")
    realtime = read_csv_if_exists(results / "realtime_metrics.csv")
    rows: list[dict[str, object]] = []

    def add(quantity: str, manuscript: float | None, recomputed: float | None, tolerance: float = 5e-4) -> None:
        if recomputed is None or (isinstance(recomputed, float) and math.isnan(recomputed)):
            status = "unavailable"
            diff = math.nan
        elif manuscript is None:
            status = "computed_no_manuscript_value"
            diff = math.nan
        else:
            diff = abs(float(recomputed) - float(manuscript))
            status = "pass" if diff <= tolerance else "mismatch"
        rows.append(
            {
                "quantity": quantity,
                "manuscript_value": manuscript,
                "recomputed_value": recomputed,
                "abs_diff": diff,
                "status": status,
            }
        )

    if not metrics.empty:
        global_by_method = metrics.groupby("method", as_index=False)[["mae", "rmse", "p95_absolute_error"]].mean(numeric_only=True)
        lookup = global_by_method.set_index("method")
        add("global_mae_mean", 1.5366, lookup.loc["mean", "mae"] if "mean" in lookup.index else None, 5e-3)
        add("global_mae_fpr_owa", 1.6171, lookup.loc["fpr_owa", "mae"] if "fpr_owa" in lookup.index else None, 5e-3)
        add("global_rmse_mean", 1.6210, lookup.loc["mean", "rmse"] if "mean" in lookup.index else None, 5e-3)
        for method, manuscript in [("direct_owa", 2.3906), ("median", 2.3952), ("mean", 2.4419)]:
            add(f"global_p95_ae_{method}", manuscript, lookup.loc[method, "p95_absolute_error"] if method in lookup.index else None, 5e-3)
        bias = metrics[(metrics["attack_type"] == "bias") & (metrics["attack_ratio"] > 0)]
        bias_lookup = bias.groupby("method")["mae"].mean(numeric_only=True)
        add("bias_attack_mae_fpr_owa", 1.6468, bias_lookup.get("fpr_owa", math.nan), 5e-3)
        add("bias_attack_mae_mean", 1.7763, bias_lookup.get("mean", math.nan), 5e-3)
        nz = metrics[metrics["method"] == "mean"].groupby("sensors_per_group")["mae"].mean(numeric_only=True)
        add("redundancy_mean_mae_n5", 1.8268, nz.get(5, math.nan), 5e-3)
        add("redundancy_mean_mae_n13", 1.1522, nz.get(13, math.nan), 5e-3)
    if not safety.empty:
        by_variant = safety.groupby("agent_variant", as_index=False).sum(numeric_only=True).set_index("agent_variant")
        add("react_without_verifier_unsafe_executed", 4409, by_variant.get("unsafe_actions_executed", pd.Series()).get("react_without_verifier", math.nan), 0.5)
        add("react_with_verifier_blocked", 4358, by_variant.get("unsafe_actions_blocked_by_verifier", pd.Series()).get("react_with_verifier", math.nan), 0.5)
        add("react_with_verifier_unsafe_executed", 0, by_variant.get("unsafe_actions_executed", pd.Series()).get("react_with_verifier", math.nan), 0.5)
    if not latency.empty:
        lat = latency.groupby("method", as_index=False).mean(numeric_only=True).set_index("method")
        add("agent_react_with_verifier_latency_seconds", 7.15, lat.get("deterministic_simulation_latency_seconds", pd.Series()).get("agent_react_with_verifier", math.nan), 0.15)
        add("agent_self_refinement_latency_seconds", 12.95, lat.get("deterministic_simulation_latency_seconds", pd.Series()).get("agent_react_with_verifier_and_self_refinement", math.nan), 0.20)
    if not realtime.empty:
        rt = realtime.groupby("deadline_ms", as_index=False).mean(numeric_only=True)
        first = rt.iloc[0] if len(rt) else pd.Series(dtype=float)
        add("realtime_event_triggered_invocation_rate", 0.0396, float(first.get("llm_invocation_rate", math.nan)), 5e-3)
        add("realtime_event_triggered_mean_loop_latency_ms", 285.4, float(first.get("mean_loop_latency_ms", math.nan)), 5.0)
        add("realtime_event_triggered_deadline_miss_rate", 0.0396, float(first.get("deadline_miss_rate", math.nan)), 5e-3)
        add("realtime_event_triggered_mae", 1.5090, float(first.get("mae", math.nan)), 5e-3)
        add("realtime_event_triggered_sla_violation_rate", 0.0, float(first.get("thermal_sla_violation_rate", math.nan)), 1e-9)
    pd.DataFrame(rows).to_csv(out / "reproduction_check.csv", index=False)


def _baseline_scores(values: np.ndarray, method: str) -> np.ndarray:
    clean = np.asarray(values, dtype=float)
    center, scale = robust_center_scale(clean)
    if method == "mad_z":
        return np.abs(clean - center) / scale
    if method == "hampel_residual":
        filtered = hampel_filter_then_mean(clean)
        return np.abs(clean - filtered) / scale
    if method == "trimmed_residual":
        filtered = trimmed_mean(clean)
        return np.abs(clean - filtered) / scale
    raise ValueError(method)


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=bool)
    pred = np.asarray(y_pred, dtype=bool)
    tp = float(np.sum(truth & pred))
    fp = float(np.sum(~truth & pred))
    tn = float(np.sum(~truth & ~pred))
    fn = float(np.sum(truth & ~pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "false_positive_rate": fpr}


def roc_pr_points(y_true: np.ndarray, scores: np.ndarray, method: str, attack_type: str, attack_ratio: float) -> pd.DataFrame:
    truth = np.asarray(y_true, dtype=bool)
    score = np.asarray(scores, dtype=float)
    thresholds = np.unique(np.quantile(score, np.linspace(0, 1, 101)))
    rows = []
    positives = max(float(np.sum(truth)), 1.0)
    negatives = max(float(np.sum(~truth)), 1.0)
    for threshold in thresholds:
        pred = score >= threshold
        tp = float(np.sum(truth & pred))
        fp = float(np.sum(~truth & pred))
        fn = float(np.sum(truth & ~pred))
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / positives
        fpr = fp / negatives
        rows.append(
            {
                "method": method,
                "attack_type": attack_type,
                "attack_ratio": attack_ratio,
                "threshold": float(threshold),
                "fpr": fpr,
                "tpr": recall,
                "precision": precision,
                "recall": recall,
                "curve": "roc_pr",
            }
        )
    return pd.DataFrame(rows)


def auc_from_curve(x: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(x)
    return float(np.trapezoid(np.asarray(y)[order], np.asarray(x)[order]))


def average_precision(points: pd.DataFrame) -> float:
    ordered = points.sort_values("recall")
    return float(np.trapezoid(ordered["precision"], ordered["recall"]))


def detection_delay(labels: np.ndarray, flags: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    flags = np.asarray(flags, dtype=bool)
    if not labels.any():
        return math.nan
    onset = int(np.argmax(labels))
    after = np.where(flags[onset:] & labels[onset:])[0]
    return float(after[0]) if after.size else math.nan


def diagnostic_records(max_traces: int | None = None, paths: RevisionPaths = RevisionPaths()) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    files = trace_files(paths.results)
    if max_traces is not None:
        files = files[:max_traces]
    for trace_path in files:
        frame = pd.read_csv(trace_path)
        parts = scenario_parts(scenario_from_trace(trace_path))
        sensors = sensor_columns(frame)
        labels = label_columns(frame)
        if not sensors or not labels:
            continue
        n = min(len(sensors), len(labels))
        sensors = sensors[:n]
        labels = labels[:n]
        alpha = float(parts.get("alpha", 0.3))
        beta = float(parts.get("beta", 0.8))
        truth_temp = frame["temperature_ground_truth"].to_numpy(dtype=float) if "temperature_ground_truth" in frame else np.full(len(frame), np.nan)
        for tick, row in frame.iterrows():
            values = row[sensors].to_numpy(dtype=float)
            true_flags = row[labels].astype(bool).to_numpy()
            result = fpr_owa_consensus(values, alpha=alpha, beta=beta, timestamp=str(row.get("timestamp", "")))
            fpr_scores = 1.0 - result.reliability_scores
            baseline = {
                "fpr_owa": (fpr_scores, result.anomaly_flags.astype(bool)),
                "mad_z": (_baseline_scores(values, "mad_z"), _baseline_scores(values, "mad_z") >= 3.0),
                "hampel_residual": (_baseline_scores(values, "hampel_residual"), _baseline_scores(values, "hampel_residual") >= 3.0),
                "trimmed_residual": (_baseline_scores(values, "trimmed_residual"), _baseline_scores(values, "trimmed_residual") >= 3.0),
            }
            abs_error = abs(result.aggregated_value - truth_temp[tick]) if not np.isnan(truth_temp[tick]) else math.nan
            for method, (scores, flags) in baseline.items():
                for idx in range(n):
                    records.append(
                        {
                            **parts,
                            "tick": tick,
                            "sensor_index": idx,
                            "method": method,
                            "attack_label": bool(true_flags[idx]),
                            "anomaly_score": float(scores[idx]),
                            "anomaly_flag": bool(flags[idx]),
                            "confidence": float(result.confidence) if method == "fpr_owa" else float(max(0.0, 1.0 - np.mean(scores) / 5.0)),
                            "absolute_error": float(abs_error),
                        }
                    )
    return pd.DataFrame(records)


def phase_b(paths: RevisionPaths = RevisionPaths(), max_traces: int | None = None) -> None:
    out = ensure_dir(paths.outputs)
    records = diagnostic_records(max_traces=max_traces, paths=paths)
    if records.empty:
        write_text(out / "phase_b_status.md", "No diagnostic records could be computed from available traces.\n")
        return
    records.to_csv(out / "diagnostic_sensor_tick_records.csv.gz", index=False)
    anomaly_detection(records, out)
    calibration(records, out)
    statistics_outputs(paths.results, out)
    degradation_outputs(paths.results, out)
    latency_outputs(paths.results, out)
    write_manifest(out)
    write_summary(out)


def anomaly_detection(records: pd.DataFrame, out: Path) -> None:
    rows = []
    curve_rows = []
    for keys, group in records.groupby(["method", "attack_type", "attack_ratio"], dropna=False):
        method, attack_type, attack_ratio = keys
        y = group["attack_label"].to_numpy(dtype=bool)
        flags = group["anomaly_flag"].to_numpy(dtype=bool)
        scores = group["anomaly_score"].to_numpy(dtype=float)
        binary = binary_metrics(y, flags)
        points = roc_pr_points(y, scores, str(method), str(attack_type), float(attack_ratio))
        auroc = auc_from_curve(points["fpr"].to_numpy(), points["tpr"].to_numpy())
        auprc = average_precision(points)
        delay_values = []
        for _, scenario_group in group.groupby(["scenario", "sensor_index"]):
            delay_values.append(detection_delay(scenario_group["attack_label"].to_numpy(), scenario_group["anomaly_flag"].to_numpy()))
        rows.append(
            {
                "method": method,
                "attack_type": attack_type,
                "attack_ratio": attack_ratio,
                "auroc": auroc,
                "auprc": auprc,
                **binary,
                "mean_detection_delay_ticks": float(np.nanmean(delay_values)) if np.isfinite(delay_values).any() else math.nan,
            }
        )
        curve_rows.append(points)
    metrics = pd.DataFrame(rows)
    curves = pd.concat(curve_rows, ignore_index=True)
    metrics.to_csv(out / "anomaly_detection_metrics.csv", index=False)
    curves.to_csv(out / "roc_pr_curves.csv", index=False)
    plot_curve(curves, out / "fig_anomaly_roc.pdf", "fpr", "tpr", "ROC curve", "False-positive rate", "True-positive rate")
    plot_curve(curves, out / "fig_anomaly_pr.pdf", "recall", "precision", "Precision-recall curve", "Recall", "Precision")
    f1 = metrics[["method", "attack_type", "attack_ratio", "f1"]].rename(columns={"f1": "anomaly_f1"})
    f1.to_csv(out / "f1_vs_beta.csv", index=False)


def plot_curve(curves: pd.DataFrame, path: Path, x: str, y: str, title: str, xlabel: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for method, group in curves.groupby("method"):
        reduced = group.groupby(x, as_index=False)[y].mean(numeric_only=True).sort_values(x)
        ax.plot(reduced[x], reduced[y], label=str(method))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def calibration(records: pd.DataFrame, out: Path) -> None:
    rows = []
    fpr = records[records["method"] == "fpr_owa"].copy()
    if fpr.empty:
        pd.DataFrame().to_csv(out / "calibration.csv", index=False)
        return
    fpr["correctness"] = 1.0 / (1.0 + fpr["absolute_error"].fillna(fpr["absolute_error"].median()))
    bins = np.linspace(0.0, 1.0, 11)
    fpr["confidence_bin"] = pd.cut(fpr["confidence"], bins=bins, include_lowest=True)
    total = len(fpr)
    ece = 0.0
    max_ce = 0.0
    for bin_label, group in fpr.groupby("confidence_bin", observed=True):
        if group.empty:
            continue
        pred = float(group["confidence"].mean())
        empirical = float(group["correctness"].mean())
        gap = abs(pred - empirical)
        ece += len(group) / total * gap
        max_ce = max(max_ce, gap)
        rows.append(
            {
                "method": "fpr_owa",
                "confidence_bin": str(bin_label),
                "mean_confidence": pred,
                "empirical_correctness_proxy": empirical,
                "mae": float(group["absolute_error"].mean()),
                "count": int(len(group)),
                "ece": ece,
                "max_calibration_error": max_ce,
            }
        )
    for method, group in records.groupby("method"):
        clean = group.dropna(subset=["confidence", "absolute_error"])
        if len(clean) > 2:
            rows.append(
                {
                    "method": method,
                    "confidence_bin": "global_correlation",
                    "mean_confidence": float(clean["confidence"].mean()),
                    "empirical_correctness_proxy": math.nan,
                    "mae": float(clean["absolute_error"].mean()),
                    "count": int(len(clean)),
                    "pearson_confidence_abs_error": float(stats.pearsonr(clean["confidence"], clean["absolute_error"]).statistic),
                    "spearman_confidence_abs_error": float(stats.spearmanr(clean["confidence"], clean["absolute_error"]).statistic),
                }
            )
    cal = pd.DataFrame(rows)
    cal.to_csv(out / "calibration.csv", index=False)
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    reliability = cal[cal["confidence_bin"].astype(str).str.startswith("(") | cal["confidence_bin"].astype(str).str.startswith("[")]
    if not reliability.empty:
        ax.plot(reliability["mean_confidence"], reliability["empirical_correctness_proxy"], marker="o", label="FPR-OWA")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Empirical correctness proxy")
    ax.set_title("Confidence calibration")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "fig_calibration_reliability.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    sample = fpr.sample(min(len(fpr), 4000), random_state=42)
    ax.scatter(sample["confidence"], sample["absolute_error"], s=4, alpha=0.15)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Absolute error")
    ax.set_title("Confidence vs. absolute error")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "fig_confidence_vs_error.pdf")
    plt.close(fig)


def statistics_outputs(results: Path, out: Path) -> None:
    metrics = read_csv_if_exists(results / "metrics.csv")
    if metrics.empty:
        return
    rows = []
    for keys, group in metrics.groupby(["method", "attack_type"], dropna=False):
        method, attack_type = keys
        for metric in ["mae", "rmse", "p95_absolute_error"]:
            values = group[metric].to_numpy(dtype=float)
            lo, hi = bootstrap_ci(values, seed=42)
            rows.append(
                {
                    "method": method,
                    "attack_type": attack_type,
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "bootstrap95_low": lo,
                    "bootstrap95_high": hi,
                    "n": int(len(values)),
                }
            )
    pd.DataFrame(rows).to_csv(out / "stats_ci.csv", index=False)
    tests = []
    pivot = metrics.pivot_table(index="scenario", columns="method", values="mae", aggfunc="mean")
    methods = [col for col in ["mean", "median", "trimmed_mean", "hampel_filter_then_mean", "direct_owa", "fpr_owa"] if col in pivot]
    if len(methods) >= 3 and len(pivot.dropna()) >= 2:
        friedman = stats.friedmanchisquare(*(pivot.dropna()[method].to_numpy() for method in methods))
        tests.append({"test": "friedman", "contrast": "all_methods_mae", "statistic": float(friedman.statistic), "p_value": float(friedman.pvalue)})
    for a, b in [("mean", "fpr_owa"), ("mean", "median"), ("mean", "direct_owa")]:
        if a in pivot and b in pivot:
            paired = pivot[[a, b]].dropna()
            if len(paired) > 1:
                test = stats.wilcoxon(paired[a], paired[b], zero_method="zsplit")
                tests.append({"test": "wilcoxon_signed_rank", "contrast": f"{a}_vs_{b}", "statistic": float(test.statistic), "p_value": float(test.pvalue)})
    pd.DataFrame(tests).to_csv(out / "stats_tests.csv", index=False)
    anova = []
    total_ss = float(np.sum((metrics["mae"] - metrics["mae"].mean()) ** 2))
    for factor in ["method", "attack_type", "attack_ratio", "sensor_noise_level", "sensors_per_group"]:
        means = metrics.groupby(factor)["mae"].transform("mean")
        ss = float(np.sum((means - metrics["mae"].mean()) ** 2))
        anova.append({"factor": factor, "eta_squared_proxy": ss / total_ss if total_ss else math.nan})
    pd.DataFrame(anova).to_csv(out / "anova_effects.csv", index=False)
    ranks = pivot.rank(axis=1, method="average").mean().sort_values()
    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    ax.scatter(ranks.values, np.arange(len(ranks)))
    ax.set_yticks(np.arange(len(ranks)), ranks.index)
    ax.set_xlabel("Average rank (lower is better)")
    ax.set_title("Critical-difference style rank summary")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "fig_critical_difference.pdf")
    plt.close(fig)


def degradation_outputs(results: Path, out: Path) -> None:
    metrics = read_csv_if_exists(results / "metrics.csv")
    if metrics.empty:
        return
    mae_beta = metrics.groupby(["method", "attack_ratio"], as_index=False)["mae"].mean(numeric_only=True)
    mae_beta.to_csv(out / "mae_vs_beta.csv", index=False)
    ab = metrics.groupby(["method", "alpha", "beta"], as_index=False)[["mae", "rmse", "p95_absolute_error"]].mean(numeric_only=True)
    ab.to_csv(out / "owa_ab_split.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for method, group in mae_beta.groupby("method"):
        ax.plot(group["attack_ratio"], group["mae"], marker="o", label=method)
    ax.set_xlabel("Attack ratio beta_a")
    ax.set_ylabel("MAE")
    ax.set_title("MAE degradation vs. attack ratio")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_mae_vs_beta.pdf")
    plt.close(fig)


def latency_outputs(results: Path, out: Path) -> None:
    latency = read_csv_if_exists(results / "latency_metrics.csv")
    realtime = read_csv_if_exists(results / "realtime_metrics.csv")
    rows = []
    if not latency.empty:
        for method, group in latency.groupby("method"):
            rows.append(
                {
                    "component": "scenario_operator_or_agent_wall_time",
                    "method": method,
                    "mean_seconds": float((group["fuzzy_consensus_latency_seconds"] + group["deterministic_simulation_latency_seconds"]).mean()),
                    "p95_seconds": float(np.percentile(group["fuzzy_consensus_latency_seconds"] + group["deterministic_simulation_latency_seconds"], 95)),
                    "definition": "Scenario-level wall-clock over all ticks for consensus methods, or per agent decision for agent variants.",
                }
            )
    if not realtime.empty:
        dedup = realtime.drop_duplicates("scenario")
        for col in ["mean_loop_latency_ms", "p95_loop_latency_ms", "p99_loop_latency_ms", "mean_llm_latency_seconds", "p95_llm_latency_seconds"]:
            if col in dedup:
                rows.append(
                    {
                        "component": col,
                        "method": "event_triggered_realtime_loop",
                        "mean_seconds": float(dedup[col].mean() / 1000.0) if col.endswith("_ms") else float(dedup[col].mean()),
                        "p95_seconds": float(np.percentile(dedup[col], 95) / 1000.0) if col.endswith("_ms") else float(np.percentile(dedup[col], 95)),
                        "definition": "Per-control-tick timing summary emitted by realtime suite.",
                    }
                )
    pd.DataFrame(rows).to_csv(out / "latency_decomposition.csv", index=False)
    write_text(
        out / "latency_reconciliation.md",
        "# Latency reconciliation\n\n"
        "`results/latency_metrics.csv` times scenario-level consensus/agent loops. For consensus methods, the reported seconds cover "
        "running a method across every tick in one scenario and include Python iteration overhead. By contrast, "
        "`results/realtime_metrics.csv` stores per-control-tick loop summaries from the event-triggered control loop. "
        "The reported p95 loop latency excludes rare LLM ticks when the 95th percentile falls in the deterministic fast-path mass; "
        "p99 and deadline-miss rates expose the slow LLM tail. These measurements are therefore complementary rather than contradictory.\n",
    )
    if not realtime.empty:
        fig, ax = plt.subplots(figsize=(5.0, 3.2))
        dedup = realtime.drop_duplicates("scenario")
        values = np.sort(dedup["mean_loop_latency_ms"].to_numpy(dtype=float))
        y = np.linspace(0, 1, len(values), endpoint=True)
        ax.plot(values, y)
        ax.set_xlabel("Mean loop latency per scenario (ms)")
        ax.set_ylabel("CDF")
        ax.set_title("Realtime loop latency CDF")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / "fig_latency_cdf.pdf")
        plt.close(fig)


def write_manifest(out: Path) -> None:
    descriptions = {
        "_inventory.md": "Inventory of existing artifacts, schemas, diagnostic availability, and latency-unit reconciliation notes.",
        "params_table.csv": "Parameter table extracted from configs and code defaults.",
        "reproduction_check.csv": "Recomputed headline quantities versus manuscript values.",
        "anomaly_detection_metrics.csv": "Sensor-tick anomaly diagnostic metrics per method, attack type, and attack ratio.",
        "roc_pr_curves.csv": "Threshold curve points for ROC and precision-recall plots.",
        "calibration.csv": "Confidence calibration, ECE proxy, and confidence-error correlations.",
        "stats_ci.csv": "Bootstrap confidence intervals and descriptive statistics.",
        "stats_tests.csv": "Friedman and Wilcoxon tests where computable from available artifacts.",
        "anova_effects.csv": "Variance-decomposition proxy effect sizes.",
        "mae_vs_beta.csv": "MAE degradation curves by attack ratio.",
        "f1_vs_beta.csv": "Anomaly F1 degradation curves by attack ratio.",
        "owa_ab_split.csv": "OWA alpha/beta split metrics.",
        "latency_decomposition.csv": "Latency definitions and component summaries.",
        "latency_reconciliation.md": "Narrative reconciliation of scenario-level and per-tick latency units.",
        "rule_supervisor_realtime_metrics.csv": "Realtime metrics for the deterministic rule-based supervisor using the same event triggers and verifier.",
        "rule_supervisor_safety_metrics.csv": "Per-scenario safety summary for the deterministic rule-based supervisor.",
        "supervisor_comparison.csv": "C3 comparison between deterministic rule-based supervision and LangGraph/Gemma event-triggered supervision.",
        "fig_supervisor_comparison.pdf": "Vector comparison figure for C3 supervisor results.",
        "llm_value_findings.md": "Short interpretation guide for whether the LLM adds measured value over a rule-based supervisor.",
        "fpr_ablation.csv": "C6 FPR-OWA component ablation: reliability-only, dominance-only, OWA-only, and full FPR-OWA.",
        "fig_fpr_ablation.pdf": "Vector figure summarizing FPR-OWA component ablation.",
        "compute_cost.csv": "C8 CPU wall-time and peak-memory micro-benchmark for consensus operators and CRP snapshot conversion.",
        "fig_compute_cost.pdf": "Vector figure summarizing lightweight compute-cost scaling.",
        "adversarial_results.csv": "C4 adversarial verifier checks for unsafe, stale, low-confidence, and process-invalid proposed actions.",
        "failure_modes.csv": "C4 verifier failure-mode and non-bypass enforcement scenarios.",
        "safety_argument.md": "Empirical and architectural safety argument for verifier-gated actuation and fallback behavior.",
    }
    files = []
    for path in sorted(out.rglob("*")):
        if path.is_file():
            rel = path.relative_to(out).as_posix()
            files.append({"file": rel, "description": descriptions.get(rel, "Major-revision generated artifact."), "reviewer_item": "major_revision"})
    write_json(out / "MANIFEST.json", {"files": files})


def write_summary(out: Path) -> None:
    lines = [
        "# Major-revision experimental summary",
        "",
        "This folder contains generated artifacts for the major-revision campaign. All quantitative values are derived from local CSV traces/results.",
        "Use `MANIFEST.json` to map files to review items.",
        "",
        "Priority checks:",
        "- `_inventory.md` reports whether sensor-tick attack labels and recomputable diagnostics are available.",
        "- `reproduction_check.csv` is the sanity gate against manuscript headline values.",
        "- `anomaly_detection_metrics.csv` and `calibration.csv` address the diagnostic-validation item.",
        "- `latency_reconciliation.md` distinguishes scenario-level operator timings from per-tick realtime loop timings.",
        "- `fpr_ablation.csv`, `compute_cost.csv`, and `adversarial_results.csv` close the C6, C8, and C4 reviewer items when present.",
    ]
    write_text(out / "SUMMARY.md", "\n".join(lines) + "\n")


def bundle(paths: RevisionPaths = RevisionPaths()) -> None:
    out = ensure_dir(paths.outputs)
    repro = ensure_dir(out / "repro_bundle")
    for file in ["requirements.txt", "requirements-gpu.txt", "pyproject.toml", "Makefile"]:
        src = Path(file)
        if src.exists():
            shutil.copy2(src, repro / src.name)
    for directory in ["configs", "scripts", "src", "tests"]:
        src_dir = Path(directory)
        dst_dir = repro / directory
        if src_dir.exists() and not dst_dir.exists():
            shutil.copytree(src_dir, dst_dir, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    write_json(
        repro / "environment.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "note": "GPU/CUDA/model server details should be collected on the remote execution host.",
        },
    )
    write_manifest(out)
    write_summary(out)
