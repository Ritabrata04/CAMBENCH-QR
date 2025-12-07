#!/usr/bin/env python3
"""
Plot DtS by regime from outputs_ft_all_metrics.csv

- Averages DtS over multiple runs for each (tag, method)
- Plots per-method lines across regimes (ZS → FT_STRUCT → FT_LEAKMIN)
- Saves a 300 dpi PNG
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REGIME_ORDER = ["ZS", "FT_STRUCT", "FT_LEAKMIN"]  # expected tags

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to outputs_ft_all_metrics.csv")
    ap.add_argument("--out", default="dts_by_regime.png", help="Output PNG (300 dpi)")
    ap.add_argument("--title", default="Distance-to-Structure (DtS)", help="Figure title")
    args = ap.parse_args()

    # 1) Load
    df = pd.read_csv(args.csv)

    # 2) Make sure required columns are present
    needed = {"tag", "method", "DtS"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}. Found: {list(df.columns)}")

    # 3) Clean and aggregate: mean DtS per (tag, method)
    work = df[["tag", "method", "DtS"]].copy()
    work["tag"] = work["tag"].astype(str).str.upper().str.replace("-", "_")
    work["DtS"] = pd.to_numeric(work["DtS"], errors="coerce")
    work = work.dropna(subset=["DtS"])

    agg = (work
           .groupby(["tag", "method"], as_index=False)["DtS"]
           .mean())

    # 4) Pivot to method rows × regime columns in desired order
    pivot = agg.pivot_table(index="method", columns="tag", values="DtS")
    # keep only known regimes, in order
    pivot = pivot.reindex(columns=REGIME_ORDER)

    # 5) Plot
    plt.figure(figsize=(6, 3.4))  # fits a single CVPR column width nicely
    x = np.arange(len(REGIME_ORDER))

    markers = ["o", "s", "D", "^", "v", "P", "X", "*"]
    for i, (mname, row) in enumerate(pivot.iterrows()):
        y = row.values.astype(float)  # may contain NaN if a method missing a regime
        plt.plot(x, y, marker=markers[i % len(markers)], linewidth=1.8, label=str(mname))

    plt.xticks(x, REGIME_ORDER)
    plt.xlabel("Regime")
    plt.ylabel("Distance-to-Structure (DtS) ↓")
    plt.title(args.title)
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    plt.legend(title="CAM", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
