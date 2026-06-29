from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.utils.io import ensure_dir


def save_basic_figures(metrics: pd.DataFrame, figures_dir: str | Path = "results/figures") -> None:
    ensure_dir(os.environ["MPLCONFIGDIR"])
    out = ensure_dir(figures_dir)
    if metrics.empty:
        return
    grouped = metrics.groupby("method", as_index=False)["mae"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(grouped["method"], grouped["mae"])
    ax.set_ylabel("MAE (C)")
    ax.set_xlabel("Method")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out / "mae_by_method.png", dpi=200)
    fig.savefig(out / "mae_by_method.pdf")
    plt.close(fig)
