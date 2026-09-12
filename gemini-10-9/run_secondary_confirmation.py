import sys
import json
import time
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path("d:/Project/NeurIPS")
OUT = ROOT / "gemini-10-9"
OUT.mkdir(exist_ok=True)

# Starting kit scoring
sys.path.insert(0, str(ROOT / "realpde_t1_starting_kit_v9" / "realpde_t1_starting_kit_v9"))
import scoring as sc

# Exact sub7 pipeline
spec = importlib.util.spec_from_file_location("exact", str(ROOT / "research/phase1_audit/exact_sub7/submission.py"))
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)

# Load manifest and feature cache
manifest = pd.read_csv(ROOT / "research/exp_zero_mean_head/artifacts/manifest.csv")
manifest["sample_id"] = manifest["file"] + ":" + manifest["start"].astype(str) + ":20:20"
feature_cache = np.load(ROOT / "research/exp_zero_mean_head/artifacts/feature_cache.npz")

train_idx = np.flatnonzero(manifest["fold"] == "train")
test_idx = np.flatnonzero(manifest["fold"] == "test")
test_manifest = manifest.iloc[test_idx].reset_index(drop=True)

# Training history-std quartile edges for u
x_train = feature_cache["x"][train_idx]
std_u_train = np.std(x_train[..., 0], axis=1, ddof=1)
q_edges_u = np.quantile(std_u_train, [0.25, 0.50, 0.75])

# Freeze inputs and point prediction on the 80 test windows
x_test = feature_cache["x"][test_idx]
y_test = feature_cache["y"][test_idx]
x_test_pad = np.pad(x_test, ((0,0),(0,0),(0,0),(0,0),(0,1)))

# Run predict to get point prediction
pred_dict = s.predict(x_test_pad)
p_point = pred_dict["prediction"][..., :2]

# Point metrics verification
rel_l2 = float(sc.rel_l2_per_sample(p_point, y_test, 2).mean())
tke = float(sc.tke_rel_l2_per_sample(p_point, y_test, 2).mean())
mvpe = float(sc.mvpe_rel_l2_per_sample(p_point, y_test).mean())

# Precompute history stds and horizon factors
uin = x_test[..., 0]
vin = x_test[..., 1]
std_u = np.std(uin, axis=1, ddof=1, keepdims=True)
std_v = np.std(vin, axis=1, ddof=1, keepdims=True)
h_factors = np.sqrt(np.arange(1, 21, dtype=np.float32) / 10.0).reshape(1, 20, 1, 1)

# Frozen v-channel half-width (identical for both)
hw_v = np.clip(0.0032 + 0.85 * std_v * h_factors, 0.0032, 0.040)

def evaluate_uq(slope_u, name):
    hw_u_raw = 0.0075 + slope_u * std_u * h_factors
    hw_u = np.clip(hw_u_raw, 0.0075, 0.040)
    hw = np.stack([hw_u, hw_v], axis=-1)
    lo = p_point - hw
    hi = p_point + hw

    # Aggregate SPS & coverage
    sps_raw, cov_overall = sc.aggregate_sps(p_point, y_test, 2, lower=lo, upper=hi)
    sps_score = float(sc.score_sps(sps_raw))
    cov_overall = float(cov_overall)

    # Width statistics (double half-width for full interval width)
    full_w = hw * 2.0
    mean_w = float(np.mean(full_w))
    median_w = float(np.median(full_w))
    mean_w_u = float(np.mean(full_w[..., 0]))
    mean_w_v = float(np.mean(full_w[..., 1]))

    # Coverage by channel u and v
    mask_scored = (y_test != 0)
    inside = (y_test >= lo) & (y_test <= hi) & mask_scored
    
    cov_u = float(np.sum(inside[..., 0]) / np.sum(mask_scored[..., 0]))
    cov_v = float(np.sum(inside[..., 1]) / np.sum(mask_scored[..., 1]))

    # Coverage by horizon (1 to 20)
    horizon_cov = []
    for h in range(20):
        c_h = float(np.sum(inside[:, h]) / np.sum(mask_scored[:, h]))
        horizon_cov.append(c_h)

    # Coverage by training-defined history-std quartile on u
    std_u_flat = np.std(uin, axis=1, ddof=1) # [N, 32, 64]
    q_cov_u = {}
    q_masks = [
        ("Q1", std_u_flat <= q_edges_u[0]),
        ("Q2", (std_u_flat > q_edges_u[0]) & (std_u_flat <= q_edges_u[1])),
        ("Q3", (std_u_flat > q_edges_u[1]) & (std_u_flat <= q_edges_u[2])),
        ("Q4", std_u_flat > q_edges_u[2]),
    ]
    for q_name, q_m in q_masks:
        q_m_20 = np.repeat(q_m[:, np.newaxis, ...], 20, axis=1) # [N, 20, 32, 64]
        scored_q = mask_scored[..., 0] & q_m_20
        ins_q = inside[..., 0] & q_m_20
        cov_q = float(np.sum(ins_q) / np.sum(scored_q))
        q_cov_u[q_name] = cov_q

    # Benchmark GPU latency
    times = []
    if torch.cuda.is_available():
        u_t = torch.from_numpy(uin).cuda()
        v_t = torch.from_numpy(vin).cuda()
        hf_t = torch.from_numpy(h_factors).cuda()
        # Warmup
        for _ in range(5):
            su_t = torch.std(u_t, dim=1, keepdim=True)
            sv_t = torch.std(v_t, dim=1, keepdim=True)
            hwu_t = torch.clamp(0.0075 + slope_u * su_t * hf_t, 0.0075, 0.040)
            hwv_t = torch.clamp(0.0032 + 0.85 * sv_t * hf_t, 0.0032, 0.040)
            torch.cuda.synchronize()
        for _ in range(20):
            t0 = time.perf_counter()
            su_t = torch.std(u_t, dim=1, keepdim=True)
            sv_t = torch.std(v_t, dim=1, keepdim=True)
            hwu_t = torch.clamp(0.0075 + slope_u * su_t * hf_t, 0.0075, 0.040)
            hwv_t = torch.clamp(0.0032 + 0.85 * sv_t * hf_t, 0.0032, 0.040)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
    latency_ms = float(np.median(times)) if times else 0.0

    return {
        "name": name,
        "slope_u": slope_u,
        "sps_score": sps_score,
        "coverage": cov_overall,
        "mean_width": mean_w,
        "median_width": median_w,
        "mean_width_u": mean_w_u,
        "mean_width_v": mean_w_v,
        "coverage_u": cov_u,
        "coverage_v": cov_v,
        "horizon_coverage": horizon_cov,
        "q_coverage_u": q_cov_u,
        "latency_ms": latency_ms,
        "lo": lo,
        "hi": hi
    }

print("Evaluating Baseline (s_u = 0.85)...")
res_base = evaluate_uq(0.85, "baseline_sub7")
print(f"Baseline SPS: {res_base['sps_score']:.6f}, cov: {res_base['coverage']*100:.2f}%, mean_w: {res_base['mean_width']:.6f}")

print("Evaluating Frozen Candidate (s_u = 0.70)...")
res_cand = evaluate_uq(0.70, "candidate_narrow_0.70")
print(f"Candidate SPS: {res_cand['sps_score']:.6f}, cov: {res_cand['coverage']*100:.2f}%, mean_w: {res_cand['mean_width']:.6f}")
print(f"Delta SPS (Candidate - Baseline): {res_cand['sps_score'] - res_base['sps_score']:+.6f}")

# Grouped stability calculations
# 1. Per Trajectory
traj_results = []
for traj, g in test_manifest.groupby("file"):
    idxs = g.index.values
    p_tr = p_point[idxs]
    y_tr = y_test[idxs]
    
    # Baseline
    lo_b = res_base["lo"][idxs]
    hi_b = res_base["hi"][idxs]
    sps_b_raw, cov_b = sc.aggregate_sps(p_tr, y_tr, 2, lower=lo_b, upper=hi_b)
    sps_b = float(sc.score_sps(sps_b_raw))
    w_b = float(np.mean((hi_b - lo_b)))
    
    # Candidate
    lo_c = res_cand["lo"][idxs]
    hi_c = res_cand["hi"][idxs]
    sps_c_raw, cov_c = sc.aggregate_sps(p_tr, y_tr, 2, lower=lo_c, upper=hi_c)
    sps_c = float(sc.score_sps(sps_c_raw))
    w_c = float(np.mean((hi_c - lo_c)))
    
    delta = sps_c - sps_b
    sign = "+" if delta > 1e-5 else ("-" if delta < -1e-5 else "=")
    re_val = int(g["Re"].iloc[0])
    aoa_val = int(g["AoA"].iloc[0])
    
    traj_results.append({
        "trajectory": traj,
        "Re": re_val,
        "AoA": aoa_val,
        "windows": len(idxs),
        "base_sps": sps_b,
        "cand_sps": sps_c,
        "delta_sps": delta,
        "sign": sign,
        "base_cov": float(cov_b),
        "cand_cov": float(cov_c),
        "base_width": w_b,
        "cand_width": w_c
    })

traj_df = pd.DataFrame(traj_results)
traj_df.to_csv(OUT / "confirmation_per_trajectory.csv", index=False)

# 2. Per Reynolds group
re_results = []
for re_val, g in test_manifest.groupby("Re"):
    idxs = g.index.values
    p_re = p_point[idxs]
    y_re = y_test[idxs]
    
    lo_b = res_base["lo"][idxs]
    hi_b = res_base["hi"][idxs]
    sps_b_raw, cov_b = sc.aggregate_sps(p_re, y_re, 2, lower=lo_b, upper=hi_b)
    sps_b = float(sc.score_sps(sps_b_raw))
    w_b = float(np.mean((hi_b - lo_b)))
    
    lo_c = res_cand["lo"][idxs]
    hi_c = res_cand["hi"][idxs]
    sps_c_raw, cov_c = sc.aggregate_sps(p_re, y_re, 2, lower=lo_c, upper=hi_c)
    sps_c = float(sc.score_sps(sps_c_raw))
    w_c = float(np.mean((hi_c - lo_c)))
    
    delta = sps_c - sps_b
    sign = "+" if delta > 1e-5 else ("-" if delta < -1e-5 else "=")
    
    re_results.append({
        "Re": int(re_val),
        "trajectories": g["file"].nunique(),
        "windows": len(idxs),
        "base_sps": sps_b,
        "cand_sps": sps_c,
        "delta_sps": delta,
        "sign": sign,
        "base_cov": float(cov_b),
        "cand_cov": float(cov_c),
        "base_width": w_b,
        "cand_width": w_c
    })

re_df = pd.DataFrame(re_results)
re_df.to_csv(OUT / "confirmation_per_reynolds.csv", index=False)

# 3. Per AoA group
aoa_results = []
for aoa_val, g in test_manifest.groupby("AoA"):
    idxs = g.index.values
    p_aoa = p_point[idxs]
    y_aoa = y_test[idxs]
    
    lo_b = res_base["lo"][idxs]
    hi_b = res_base["hi"][idxs]
    sps_b_raw, cov_b = sc.aggregate_sps(p_aoa, y_aoa, 2, lower=lo_b, upper=hi_b)
    sps_b = float(sc.score_sps(sps_b_raw))
    w_b = float(np.mean((hi_b - lo_b)))
    
    lo_c = res_cand["lo"][idxs]
    hi_c = res_cand["hi"][idxs]
    sps_c_raw, cov_c = sc.aggregate_sps(p_aoa, y_aoa, 2, lower=lo_c, upper=hi_c)
    sps_c = float(sc.score_sps(sps_c_raw))
    w_c = float(np.mean((hi_c - lo_c)))
    
    delta = sps_c - sps_b
    sign = "+" if delta > 1e-5 else ("-" if delta < -1e-5 else "=")
    
    aoa_results.append({
        "AoA": int(aoa_val),
        "trajectories": g["file"].nunique(),
        "windows": len(idxs),
        "base_sps": sps_b,
        "cand_sps": sps_c,
        "delta_sps": delta,
        "sign": sign,
        "base_cov": float(cov_b),
        "cand_cov": float(cov_c),
        "base_width": w_b,
        "cand_width": w_c
    })

aoa_df = pd.DataFrame(aoa_results)
aoa_df.to_csv(OUT / "confirmation_per_aoa.csv", index=False)

# Save summary JSON
del res_base["lo"], res_base["hi"]
del res_cand["lo"], res_cand["hi"]

summary_data = {
    "point_metrics": {
        "rel_l2": rel_l2,
        "tke": tke,
        "mvpe": mvpe
    },
    "aggregate_comparison": {
        "baseline": res_base,
        "candidate": res_cand,
        "delta_sps": res_cand["sps_score"] - res_base["sps_score"],
        "delta_coverage": res_cand["coverage"] - res_base["coverage"],
        "delta_mean_width": res_cand["mean_width"] - res_base["mean_width"]
    },
    "per_trajectory": traj_results,
    "per_reynolds": re_results,
    "per_aoa": aoa_results
}

with open(OUT / "secondary_confirmation_results.json", "w", encoding="utf-8") as f:
    json.dump(summary_data, f, indent=2)

print("\nConfirmation analysis complete. Artifacts written to gemini-10-9/.")
