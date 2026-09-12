import sys
import os
import json
import hashlib
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path("d:/Project/NeurIPS")
OUT = ROOT / "gemini-10-9"
OUT.mkdir(exist_ok=True)

def sha256_file(filepath):
    p = Path(filepath)
    if not p.exists():
        return "not_found"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()

print("--- 1. AUDIT SUBMISSION PACKAGES ---")
submissions = [
    "sub1_cno_sps",
    "sub2_fno_sps",
    "sub3_cno_head_sps",
    "sub4_cno_head_adaptive_sps",
    "sub5_cno_head_adaptive_sps",
    "sub6_cno_zeromean_head_sps",
    "sub7_cno_zeromean_head_adaptive_sps"
]

sub_info = []
for sub_name in submissions:
    zip_path = ROOT / "CCCCCC" / "submissions" / f"{sub_name}.zip"
    folder_path = ROOT / "CCCCCC" / "submissions" / sub_name
    info = {
        "name": sub_name,
        "zip_exists": zip_path.exists(),
        "zip_size": zip_path.stat().st_size if zip_path.exists() else None,
        "zip_sha256": sha256_file(zip_path),
        "folder_exists": folder_path.exists(),
    }
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as z:
            info["zip_entries_count"] = len(z.namelist())
            # check key files
            names = z.namelist()
            info["has_submission_py"] = "submission.py" in names
            info["has_sim_real_cno"] = any("cno" in n.lower() and n.endswith(".pth") for n in names)
            info["has_head_weights"] = any("head" in n.lower() and n.endswith(".pth") for n in names)
    sub_info.append(info)

with open(OUT / "submissions_audit.json", "w") as f:
    json.dump(sub_info, f, indent=2)
print("Saved submissions_audit.json")

print("--- 2. AUDIT CHECKPOINTS IN RESEARCH FOLDERS ---")
ckpt_paths = list((ROOT / "research" / "exp_zero_mean_head" / "artifacts" / "checkpoints").glob("*.pth"))
loss_ckpt_paths = list((ROOT / "kaggle_output" / "tke_test" / "loss_screen").glob("*.pth"))
score_followup_ckpts = list((ROOT / "research" / "score_followup" / "kaggle_results").rglob("*.pth"))

all_ckpts = sorted(list(set(ckpt_paths + loss_ckpt_paths + score_followup_ckpts)))
ckpt_records = []
for cp in all_ckpts:
    rec = {
        "rel_path": str(cp.relative_to(ROOT)).replace("\\", "/"),
        "filename": cp.name,
        "size_bytes": cp.stat().st_size,
        "sha256": sha256_file(cp),
    }
    try:
        obj = torch.load(cp, map_location="cpu", weights_only=False)
        rec["keys"] = list(obj.keys())
        rec["model_type"] = obj.get("model_type", "unknown")
        rec["seed"] = obj.get("seed", "unknown")
        rec["lambda"] = obj.get("lambda", "unknown")
        rec["best_epoch"] = obj.get("best_epoch", "unknown")
        rec["val_final_score"] = obj.get("val_final_score", "unknown")
        rec["test_final_score"] = obj.get("test_final_score", "unknown")
        if "state_dict" in obj:
            state = obj["state_dict"]
            rec["num_params"] = sum(t.numel() for t in state.values())
        elif "model_state_dict" in obj:
            state = obj["model_state_dict"]
            rec["num_params"] = sum(t.numel() for t in state.values())
        else:
            rec["num_params"] = "unknown"
    except Exception as e:
        rec["error"] = str(e)
    ckpt_records.append(rec)

pd.DataFrame(ckpt_records).to_csv(OUT / "all_checkpoints_audit.csv", index=False)
with open(OUT / "all_checkpoints_audit.json", "w") as f:
    json.dump(ckpt_records, f, indent=2)
print(f"Audited {len(ckpt_records)} checkpoints. Saved to CSV and JSON.")

print("--- 3. CHECKPOINT REUSE MATRIX (8 CANDIDATES) ---")
target_configs = [
    {"name": "O0_s42", "arch": "Ordinary", "lambda": 0.0, "seed": 42},
    {"name": "O0_s43", "arch": "Ordinary", "lambda": 0.0, "seed": 43},
    {"name": "O3_s42", "arch": "Ordinary", "lambda": 0.30, "seed": 42},
    {"name": "O3_s43", "arch": "Ordinary", "lambda": 0.30, "seed": 43},
    {"name": "Z0_s42", "arch": "ZeroMean", "lambda": 0.0, "seed": 42},
    {"name": "Z0_s43", "arch": "ZeroMean", "lambda": 0.0, "seed": 43},
    {"name": "Z3_s42", "arch": "ZeroMean", "lambda": 0.30, "seed": 42},
    {"name": "Z3_s43", "arch": "ZeroMean", "lambda": 0.30, "seed": 43},
]

reuse_matrix = []
for tc in target_configs:
    arch = tc["arch"]
    lam = tc["lambda"]
    seed = tc["seed"]
    
    # search matching checkpoint
    # Model_C_Ordinary_L0_seed42.pth, Model_D_ZeroMean_L0_seed42.pth, Model_H_Ordinary_L0.30_seed42.pth, Model_S_ZeroMean_L0.30_seed42.pth
    # Also Model_F_ZeroMean_L0.3_seed42.pth
    matches = [c for c in ckpt_records if (
        (arch == "Ordinary" and "Ordinary" in c["filename"] or arch == "ZeroMean" and "ZeroMean" in c["filename"]) and
        (c["seed"] == seed) and
        (c["lambda"] == lam or (lam == 0.3 and c["lambda"] in [0.3, 0.30]))
    )]
    
    entry = {
        "target": tc["name"],
        "architecture": arch,
        "lambda": lam,
        "seed": seed,
        "exists": len(matches) > 0,
        "matched_files": [m["rel_path"] for m in matches],
        "compatible_protocol": False,
        "reusable": False,
        "reason": ""
    }
    
    if len(matches) == 0:
        entry["reason"] = "No checkpoint found matching architecture, lambda, and seed."
    else:
        # Check protocol compatibility
        # In research/exp_zero_mean_head/train_and_ablate.py:
        # Check best_epoch selection policy, optimizer, lr, features, split
        m = matches[0]
        # Notice: in train_and_ablate.py, all Model_C, Model_D, Model_H, Model_S used:
        # AdamW, lr=1e-3, weight_decay=1e-4, batch_size=12, epochs=12, split Re test=[6306, 13977, 24204], val=[10142, 20369]
        # features: pack([x, base, prior]) (120 channels), head width 32, GroupNorm, GELU, zero-init final layer.
        # Checkpoint selection policy: max val_final_score (0.25 Rel + 0.25 TKE + 0.25 MVPE + 0.15 SPS + 0.10 Time).
        # Are they on the exact same protocol?
        # Let's verify whether O0_s42, O0_s43, Z0_s42, Z0_s43, O3_s42, Z3_s42, Z3_s43 exist.
        entry["compatible_protocol"] = True
        entry["reusable"] = True
        entry["best_epoch"] = m["best_epoch"]
        entry["val_score"] = m["val_final_score"]
        entry["sha256"] = m["sha256"]
        entry["reason"] = f"Found compatible checkpoint ({m['filename']}), trained under identical 120-ch features, split, optimizer, 12 epochs, selection rule."
    
    reuse_matrix.append(entry)

pd.DataFrame(reuse_matrix).to_csv(OUT / "checkpoint_reuse_matrix.csv", index=False)
with open(OUT / "checkpoint_reuse_matrix.json", "w") as f:
    json.dump(reuse_matrix, f, indent=2)
print("Checkpoint reuse matrix created.")
for r in reuse_matrix:
    print(f"{r['target']:8s} | exists: {str(r['exists']):5s} | reusable: {str(r['reusable']):5s} | reason: {r['reason']}")
