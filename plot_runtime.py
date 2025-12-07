#!/usr/bin/env python3
"""
Plot ms_per_img (runtime) per method × regime from outputs_ft_all_metrics.csv

- Averages ms_per_img across multiple runs per (tag, method)
- Groups bars by regime with one bar per method
- Saves a 300 dpi PNG

Usage:
  python plot_runtime.py --csv ./outputs_ft_all_metrics.csv --out runtime.png
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REGIME_ORDER = ["ZS", "FT_STRUCT", "FT_LEAKMIN"]

def normalize_tag(val: str) -> str:
    u = str(val).upper().replace("-", "_").strip()
    if u in {"ZS", "ZERO_SHOT"}: return "ZS"
    if u in {"FT_STRUCT", "FTSTRUCT"}: return "FT_STRUCT"
    if u in {"FT_LEAKMIN", "FTLEAKMIN"}: return "FT_LEAKMIN"
    return u

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to outputs_ft_all_metrics.csv")
    ap.add_argument("--out", default="runtime.png", help="Output PNG (300 dpi)")
    args = ap.parse_args()

    # 1) Load
    df = pd.read_csv(args.csv)

    # 2) Check columns
    needed = {"tag", "method", "ms_per_img"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}. Found: {list(df.columns)}")

    # 3) Normalize & aggregate
    df["tag"] = df["tag"].apply(normalize_tag)
    df["ms_per_img"] = pd.to_numeric(df["ms_per_img"], errors="coerce")
    work = df.dropna(subset=["ms_per_img"])
    agg = work.groupby(["tag", "method"], as_index=False)["ms_per_img"].mean()

    # Pivot: regimes as x-axis, methods as series
    pivot = agg.pivot_table(index="tag", columns="method", values="ms_per_img")
    pivot = pivot.reindex(REGIME_ORDER)

    # 4) Plot
    plt.figure(figsize=(6.5, 3.5))
    bar_width = 0.25
    x = np.arange(len(REGIME_ORDER))

    methods = pivot.columns.tolist()
    for i, m in enumerate(methods):
        vals = pivot[m].values
        plt.bar(x + i*bar_width, vals, width=bar_width, label=m)

    plt.xticks(x + bar_width*(len(methods)-1)/2, REGIME_ORDER)
    plt.ylabel("Runtime (ms / image) ↓")
    plt.xlabel("Regime")
    plt.title("Runtime of CAM Methods")
    # Legend outside plot, on the right
    plt.legend(title="CAM", framealpha=0.9, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
