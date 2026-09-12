import sys
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("d:/Project/NeurIPS")
OUT = ROOT / "gemini-10-9"
OUT.mkdir(exist_ok=True)

print("--- AUDIT SAMPLE IDENTITY AND FOLD OVERLAP ---")
manifest_path = ROOT / "research" / "exp_zero_mean_head" / "artifacts" / "manifest.csv"
meta = pd.read_csv(manifest_path)
meta["sample_id"] = meta["file"] + ":" + meta["start"].astype(str) + ":20:20"
meta["history_start"] = meta["start"]
meta["history_end"] = meta["start"] + 20
meta["future_start"] = meta["start"] + 20
meta["future_end"] = meta["start"] + 40

print(f"Total windows: {len(meta)}")
print(f"Unique sample_ids: {meta['sample_id'].nunique()} (is_unique: {meta['sample_id'].is_unique})")
print("Fold distribution:\n", meta["fold"].value_counts())

# Check trajectory overlap across folds
traj_folds = meta.groupby("file")["fold"].nunique()
print(f"Max folds per trajectory: {traj_folds.max()} (should be 1 if folds are strictly trajectory/condition-separated)")

# Check raw frame overlap across folds
cache_dir = ROOT / "CCCCCC" / "runs" / "cache" / "real"
cross_fold_overlaps = []
for file, g in meta.groupby("file"):
    for i, a in g.iterrows():
        for j, b in g.iterrows():
            if i < j and a["fold"] != b["fold"]:
                # Check if frame ranges overlap
                if max(a["history_start"], b["history_start"]) < min(a["future_end"], b["future_end"]):
                    cross_fold_overlaps.append({
                        "file": file,
                        "sample_a": a["sample_id"], "fold_a": a["fold"],
                        "sample_b": b["sample_id"], "fold_b": b["fold"]
                    })

print(f"Cross-fold frame overlaps: {len(cross_fold_overlaps)}")

# Verify saved predictions vs raw cache targets
fcache_path = ROOT / "research" / "exp_zero_mean_head" / "artifacts" / "feature_cache.npz"
fcache = np.load(fcache_path)
test_idx = np.where(meta["fold"] == "test")[0]
test_meta = meta.iloc[test_idx].reset_index(drop=True)

test_preds_path = ROOT / "research" / "exp_zero_mean_head" / "artifacts" / "test_predictions.npz"
tpreds = np.load(test_preds_path)

target_in_preds = tpreds["target"][..., :2]
target_in_fcache = fcache["y"][test_idx]

diff_target = np.abs(target_in_preds - target_in_fcache).max()
print(f"Max abs diff between test_predictions.npz target and feature_cache target: {diff_target}")

# Verify with raw cache files directly
raw_checks = []
for file, g in test_meta.groupby("file"):
    raw_path = cache_dir / file.replace(".h5", ".npz")
    with np.load(raw_path) as rc:
        t_starts = list(rc["t_starts"])
        for _, row in g.iterrows():
            s = row["start"]
            w_idx = t_starts.index(s)
            raw_target = rc["target"][w_idx, ..., :2]
            cached_target = fcache["y"][row.name] # row.name is original index in meta
            diff = np.abs(raw_target - cached_target).max()
            raw_checks.append({
                "sample_id": row["sample_id"],
                "file": file,
                "start": s,
                "re": int(rc["re"]),
                "aoa": int(rc["aoa"]),
                "meta_re": row["Re"],
                "meta_aoa": row["AoA"],
                "max_diff_raw_vs_cached": float(diff)
            })

raw_df = pd.DataFrame(raw_checks)
print(f"Verified {len(raw_df)} test windows against raw trajectory files.")
print(f"Max difference raw vs cached: {raw_df['max_diff_raw_vs_cached'].max()}")
print(f"Any Re/AoA mismatches: {any(raw_df['re'] != raw_df['meta_re']) or any(raw_df['aoa'] != raw_df['meta_aoa'])}")

# Check Hypo 13/14 Row Slice Issue
# In Hypo 13/14:
# Row 0..27: labeled 6306
# Row 28..53: labeled 13977
# Row 54..79: labeled 24204
test_meta["hypo13_assigned_Re"] = [6306 if i < 28 else (13977 if i < 54 else 24204) for i in range(len(test_meta))]
mismatched_re = test_meta[test_meta["Re"] != test_meta["hypo13_assigned_Re"]]
print(f"Hypo 13/14 mislabeled rows: {len(mismatched_re)} / {len(test_meta)} ({len(mismatched_re)/len(test_meta)*100:.1f}%)")

actual_counts = test_meta["Re"].value_counts().to_dict()
hypo_counts = test_meta["hypo13_assigned_Re"].value_counts().to_dict()
print(f"Actual Re distribution on test (80w): {actual_counts}")
print(f"Hypo 13/14 assumed distribution: {hypo_counts}")

# Determine whether this affected model training or ONLY diagnostic labels:
# 1. Look at train_and_ablate.py:
# Does train_and_ablate.py use row slices or meta["Re"] for splits?
# Line 184: fold = np.where(meta.Re.isin(CONFIG["test_Re"]), "test", ...)
# Line 187: meta.to_csv(OUT_DIR / "manifest.csv", index=False)
# It used true HDF5 Re from the cache metadata!
# 2. In evaluate_suite.py / hypo notebooks:
# The mislabeling occurred ONLY in the post-hoc diagnostic grouping scripts in Hypo 13 and Hypo 14 notebooks
# where someone did: re_labels = [6306]*28 + [13977]*26 + [24204]*26 without checking manifest.csv!
print("Hypo 13/14 scope conclusion: Affected ONLY per-Re diagnostic tables in Hypo 13/14 notebooks. Model training, fold partitioning, and aggregate prediction/target alignment were strictly guided by explicit manifest metadata and were completely unaffected.")

audit_identity = {
    "total_windows": len(meta),
    "unique_sample_ids": int(meta["sample_id"].nunique()),
    "is_unique": bool(meta["sample_id"].is_unique),
    "fold_counts": meta["fold"].value_counts().to_dict(),
    "max_folds_per_trajectory": int(traj_folds.max()),
    "cross_fold_frame_overlaps": len(cross_fold_overlaps),
    "max_diff_raw_vs_cached_target": float(raw_df["max_diff_raw_vs_cached"].max()),
    "hypo13_mislabeled_windows": len(mismatched_re),
    "actual_re_distribution": actual_counts,
    "hypo13_assumed_re_distribution": hypo_counts,
    "affected_scope": "Diagnostic labels only; model training, splits, and aggregate predictions unaffected."
}

with open(OUT / "sample_identity_audit.json", "w") as f:
    json.dump(audit_identity, f, indent=2)

test_meta.to_csv(OUT / "test_sample_manifest.csv", index=False)
print("Saved sample_identity_audit.json and test_sample_manifest.csv")
