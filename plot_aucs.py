#!/usr/bin/env python3
"""
Plot AUCs (MISF, MIST, BG) by regime from outputs_ft_all_metrics.csv

- Averages AUCs over multiple runs for each (tag, method)
- Produces a single figure with 3 subplots: AUC_MISF, AUC_MIST, AUC_BG
- Each subplot shows per-method lines across regimes (ZS → FT_STRUCT → FT_LEAKMIN)
- Saves a 300 dpi PNG

Usage:
  python plot_aucs_from_outputs_ft.py --csv ./outputs_ft_all_metrics.csv --out aucs_by_regime.png
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REGIME_ORDER = ["ZS", "FT_STRUCT", "FT_LEAKMIN"]  # expected tags
AUC_COLUMNS  = ["AUC_MISF", "AUC_MIST", "AUC_BG"]

def normalize_tag(val: str) -> str:
    u = str(val).upper().replace("-", "_").strip()
    if u in {"ZS", "ZERO_SHOT"}: return "ZS"
    if u in {"FT_STRUCT", "FTSTRUCT"}: return "FT_STRUCT"
    if u in {"FT_LEAKMIN", "FTLEAKMIN"}: return "FT_LEAKMIN"
    return u

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to outputs_ft_all_metrics.csv")
    ap.add_argument("--out", default="aucs_by_regime.png", help="Output PNG (300 dpi)")
    ap.add_argument("--title", default="Structure-level AUCs by Regime", help="Figure title")
    args = ap.parse_args()

    # 1) Load
    df = pd.read_csv(args.csv)

    # 2) Check columns
    needed = {"tag", "method"} | set(AUC_COLUMNS)
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}. Found: {list(df.columns)}")

    # 3) Clean and aggregate per (tag, method)
    work = df[["tag", "method"] + AUC_COLUMNS].copy()
    work["tag"] = work["tag"].apply(normalize_tag)
    for c in AUC_COLUMNS:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna(subset=AUC_COLUMNS)

    agg = (work
           .groupby(["tag", "method"], as_index=False)[AUC_COLUMNS]
           .mean())

    # 4) Pivot each AUC into method × regime matrix (ordered regimes)
    pivots = {}
    methods_order = sorted(agg["method"].unique().tolist())
    for c in AUC_COLUMNS:
        p = agg.pivot_table(index="method", columns="tag", values=c)
        p = p.reindex(index=methods_order, columns=REGIME_ORDER)
        pivots[c] = p

    # 5) Plot: 3 subplots in one row (single figure)
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.2), sharey=True)  # fits a CVPR column width nicely
    x = np.arange(len(REGIME_ORDER))
    markers = ["o", "s", "D", "^", "v", "P", "X", "*"]

    for ax, c, pretty in zip(
        axes,
        AUC_COLUMNS,
        ["AUC_MISF (finder) ↑", "AUC_MIST (timing) ↑", "AUC_BG (background) ↓"]
    ):
        pivot = pivots[c]
        for i, (mname, row) in enumerate(pivot.iterrows()):
            y = row.values.astype(float)
            ax.plot(x, y, marker=markers[i % len(markers)], linewidth=1.8, label=str(mname))
        ax.set_xticks(x, REGIME_ORDER)
        ax.set_xlabel("Regime")
        ax.set_title(pretty)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

    axes[0].set_ylabel("AUC")
    # single legend for all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="CAM", loc="upper center", ncols=min(len(labels), 4), framealpha=0.9)
    fig.suptitle(args.title, y=1.04)
    fig.tight_layout()
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
