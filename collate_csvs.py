#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collate all paper CSV outputs into one exportable bundle.

What it does:
- Finds all .csv under the specified roots (default: outputs_ft, outputs_zero_shot, outputs_more)
- Copies them into <base>/<out_dir>/csv/ with collision-proof names (prefix by source + flattened relpath)
- Builds a manifest (index) with: source root, relpath, dest filename, rows/cols, file size, md5, columns, and a detected "type"
- (Optional) builds a small "quick_summary.csv" if it finds common metrics tables
- Writes a README and produces a ZIP archive you can upload

Usage (defaults are fine for your setup):
  python3 collate_csvs.py
  python3 collate_csvs.py --base /home/ritabrata/qr_stuff --roots outputs_ft outputs_zero_shot outputs_more --out packaged_csvs

Output:
  /home/ritabrata/qr_stuff/<out_dir>/manifest.csv
  /home/ritabrata/qr_stuff/<out_dir>/quick_summary.csv (if inputs exist)
  /home/ritabrata/qr_stuff/<out_dir>/<out_dir>.zip
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import pandas as pd

# ----------------------- Helpers -----------------------

def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def detect_table_type(cols: List[str], filename: str) -> str:
    c = set([x.lower() for x in cols])

    # Strong signals
    if {"method", "fmr", "tmr", "bl"} <= c:
        if "structurescore" in c:
            return "methods_metrics_with_structscore"
        return "methods_metrics"
    if {"auc_ins_qr", "auc_del_qr"} <= c or {"auc_ins_bg", "auc_del_bg"} <= c or "faithfulness" in filename.lower():
        return "faithfulness_summary_or_detail"
    if {"drop_finder", "drop_timing"} & c:
        return "causal_occlusion"
    if {"p", "fmr", "tmr"} <= c and "bl" in c:
        return "stress_monotonic"
    if {"thr", "iou"} <= c:
        return "threshold_robustness"
    if {"size", "fmr", "tmr", "bl"} <= c:
        return "cross_resolution"
    if {"ms_per_img"} <= c or {"cuda_mb"} <= c:
        return "runtime_memory"
    if "index" in filename.lower():
        return "index_or_catalog"

    # Weaker filename hints
    fn = filename.lower()
    if fn.startswith("metrics_"):
        return "metrics_misc"
    if fn.startswith("faithfulness_") or fn.endswith("_faithfulness.csv"):
        return "faithfulness_summary_or_detail"
    if "causal" in fn:
        return "causal_occlusion"
    if "stress" in fn:
        return "stress_monotonic"
    if "thr_" in fn or "threshold" in fn:
        return "threshold_robustness"
    if "xres" in fn:
        return "cross_resolution"
    return "other"

def flatten_name(src_root_name: str, rel_path: Path) -> str:
    """
    Make a collision-proof filename: <root>__<relpath with separators as '__'>
    """
    rel_str = str(rel_path).replace(os.sep, "__")
    return f"{src_root_name}__{rel_str}"

def safe_read_csv(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p)
    except Exception as e:
        # fallback: try python engine
        try:
            return pd.read_csv(p, engine="python")
        except Exception:
            # return empty with a marker col
            return pd.DataFrame({"__read_error__": [str(e)]})

# ----------------------- Quick Summary -----------------------

def build_quick_summary(collected_dir: Path, manifest_df: pd.DataFrame) -> pd.DataFrame:
    """
    Optionally build a tiny cross-file summary if common tables are present:
      - Any metrics_*.csv: group by [method] and compute means of [FMR, TMR, BL, StructureScore] if available.
      - Any runtime_memory_*.csv: report mean ms/img per regime/method.
      - Any faithfulness_*_summary.csv: keep method-wise AUC_ins/deletion QR/BG averages.

    Returns a concatenated DataFrame (may be empty).
    """
    rows = []

    # 1) Per-method structural metrics (from metrics_*.csv)
    for rec in manifest_df.itertuples():
        if not rec.dest_name.lower().startswith(("outputs_ft__metrics_", "outputs_zero_shot__metrics_", "outputs_more__metrics_")):
            continue
        df = safe_read_csv(collected_dir / "csv" / rec.dest_name)
        if df.empty: 
            continue
        cols = set([c.lower() for c in df.columns])
        if "method" in cols:
            keep = [c for c in df.columns if c.lower() in {"method", "fmr", "tmr", "bl", "structurescore"}]
            gd = (df[keep]
                  .groupby("method", as_index=False)
                  .mean(numeric_only=True))
            gd["source_csv"] = rec.dest_name
            rows.append(("per_method_metrics", gd))

    # 2) Runtime/memory
    for rec in manifest_df.itertuples():
        if "runtime_memory" not in rec.dest_name.lower():
            continue
        df = safe_read_csv(collected_dir / "csv" / rec.dest_name)
        if df.empty:
            continue
        cols = set([c.lower() for c in df.columns])
        if {"method", "ms_per_img"} <= cols:
            keep = [c for c in df.columns if c.lower() in {"regime", "method", "ms_per_img", "cuda_mb"}]
            gd = df[keep].groupby(["regime", "method"], as_index=False).mean(numeric_only=True)
            gd["source_csv"] = rec.dest_name
            rows.append(("runtime_memory", gd))

    # 3) Faithfulness summaries
    for rec in manifest_df.itertuples():
        if not rec.dest_name.lower().startswith("outputs_more__faithfulness_") or not rec.dest_name.lower().endswith("_summary.csv"):
            continue
        df = safe_read_csv(collected_dir / "csv" / rec.dest_name)
        if df.empty:
            continue
        cols = set([c.lower() for c in df.columns])
        if {"regime", "method"} <= cols:
            keep_cols = [c for c in df.columns if c.lower() in {
                "regime", "method", "auc_ins", "auc_del", "auc_ins_qr", "auc_del_qr", "auc_ins_bg", "auc_del_bg"
            }]
            gd = df[keep_cols].copy()
            gd["source_csv"] = rec.dest_name
            rows.append(("faithfulness", gd))

    if not rows:
        return pd.DataFrame()

    # Concatenate with a section label
    out = []
    for label, df in rows:
        df2 = df.copy()
        df2.insert(0, "__section__", label)
        out.append(df2)
    all_summary = pd.concat(out, ignore_index=True)
    return all_summary

# ----------------------- Main -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="/home/ritabrata/qr_stuff",
                        help="Base directory that holds your outputs_* folders.")
    parser.add_argument("--roots", type=str, nargs="+",
                        default=["outputs_ft", "outputs_zero_shot", "outputs_more"],
                        help="Relative folders under --base to scan for CSVs.")
    parser.add_argument("--out", type=str, default="packaged_csvs",
                        help="Name of output folder created under --base.")
    parser.add_argument("--zip", action="store_true", default=True,
                        help="Create a zip archive of the packaged CSVs.")
    args = parser.parse_args()

    base = Path(args.base).resolve()
    outdir = base / args.out
    csv_out = outdir / "csv"
    outdir.mkdir(parents=True, exist_ok=True)
    csv_out.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Base: {base}")
    print(f"[INFO] Roots: {args.roots}")
    print(f"[INFO] Out:   {outdir}")

    manifest_rows: List[Dict] = []

    total_found = 0
    total_copied = 0

    for root_name in args.roots:
        src_root = base / root_name
        if not src_root.exists():
            print(f"[WARN] Missing root: {src_root}")
            continue

        csvs = list(src_root.rglob("*.csv"))
        total_found += len(csvs)
        print(f"[SCAN] {root_name}: {len(csvs)} CSV files")

        for p in csvs:
            rel = p.relative_to(src_root)
            dest_name = flatten_name(root_name, rel)
            dest = csv_out / dest_name

            # copy with metadata
            shutil.copy2(p, dest)
            total_copied += 1

            # attempt to read to get schema info
            df = safe_read_csv(dest)
            rows = int(df.shape[0]) if not df.empty else 0
            cols = int(df.shape[1]) if not df.empty else 0
            colnames = list(df.columns)
            fsize = dest.stat().st_size
            md5 = md5sum(dest)
            dtype = detect_table_type(colnames, dest_name)

            manifest_rows.append({
                "src_root": root_name,
                "rel_path": str(rel),
                "dest_name": dest_name,
                "rows": rows,
                "cols": cols,
                "size_bytes": fsize,
                "md5": md5,
                "detected_type": dtype,
                "columns_json": json.dumps(colnames, ensure_ascii=False),
            })

    # Write manifest
    manifest = pd.DataFrame(manifest_rows)
    mpath = outdir / "manifest.csv"
    manifest.to_csv(mpath, index=False)
    print(f"[OK] Wrote manifest: {mpath}  (files: {len(manifest_rows)})")

    # Quick summary (optional, best-effort)
    try:
        quick = build_quick_summary(outdir, manifest)
        if not quick.empty:
            qpath = outdir / "quick_summary.csv"
            quick.to_csv(qpath, index=False)
            print(f"[OK] Wrote quick summary: {qpath}")
        else:
            print("[INFO] Quick summary skipped (no recognized inputs).")
    except Exception as e:
        print(f"[WARN] Quick summary failed: {e}")

    # README
    readme = outdir / "README_PACKAGED.txt"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    readme.write_text(
        f"Packaged CSV bundle for Structural XAI QR paper\n"
        f"Timestamp: {ts}\n"
        f"\n"
        f"Folders:\n"
        f"  csv/                ← all CSV files, flattened + prefixed by source folder\n"
        f"  manifest.csv        ← index with paths, md5, shapes, detected type, columns\n"
        f"  quick_summary.csv   ← (optional) compact summary of metrics/runtime/faithfulness\n"
        f"\n"
        f"How to use:\n"
        f"  • Upload the ZIP (<out_dir>.zip) to ChatGPT and ask for analysis.\n"
        f"  • I will use manifest.csv to navigate and cross-check tables.\n"
    )
    print(f"[OK] Wrote README: {readme}")

    # Zip it
    if args.zip:
        zip_path = outdir / f"{args.out}.zip"
        # make_archive wants base name without extension and a dir
        root_for_zip = str(outdir / args.out)
        # Create a temp folder with a stable name inside outdir to avoid zipping parent junk
        temp_pack = outdir / args.out
        if temp_pack.exists():
            shutil.rmtree(temp_pack)
        temp_pack.mkdir(parents=True, exist_ok=True)
        # copy manifest, quick summary, README, and csv folder into temp_pack
        shutil.copy2(mpath, temp_pack / "manifest.csv")
        if (outdir / "quick_summary.csv").exists():
            shutil.copy2(outdir / "quick_summary.csv", temp_pack / "quick_summary.csv")
        shutil.copy2(readme, temp_pack / "README_PACKAGED.txt")
        shutil.copytree(csv_out, temp_pack / "csv", dirs_exist_ok=True)
        shutil.make_archive(root_for_zip, "zip", temp_pack)
        print(f"[ZIP] Created: {zip_path}")
        # clean temp
        shutil.rmtree(temp_pack, ignore_errors=True)

    print(f"\n[SUMMARY] Found {total_found} CSVs, copied {total_copied}.")
    print(f"[NEXT] Upload: {outdir}/{args.out}.zip")

if __name__ == "__main__":
    # pandas wide print for debug if needed
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 160)
    main()
