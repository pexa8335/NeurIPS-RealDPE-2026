import sys
import os
import json
import time
import hashlib
import zipfile
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path("d:/Project/NeurIPS")
OUT = ROOT / "gemini-10-9"
OUT.mkdir(exist_ok=True)

# Add starting kit scoring
sys.path.insert(0, str(ROOT / "realpde_t1_starting_kit_v9" / "realpde_t1_starting_kit_v9"))
import scoring as sc

# Add models
sys.path.insert(0, str(ROOT / "research" / "exp_zero_mean_head"))
from models import OrdinaryResidualHead, ZeroMeanResidualHead

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()

print("=" * 80)
print("RUNNING COMPREHENSIVE PHASE 1 AUDIT IN gemini-10-9")
print("=" * 80)

# Load Manifest and Caches
manifest_path = ROOT / "research" / "exp_zero_mean_head" / "artifacts" / "manifest.csv"
meta = pd.read_csv(manifest_path)
meta["sample_id"] = meta["file"] + ":" + meta["start"].astype(str) + ":20:20"

fcache_path = ROOT / "research" / "exp_zero_mean_head" / "artifacts" / "feature_cache.npz"
fcache = np.load(fcache_path)
x = fcache["x"]
y = fcache["y"]
base_cached = fcache["base"]
train_idx = np.where(meta["fold"] == "train")[0]
test_idx = np.where(meta["fold"] == "test")[0]
val_idx = np.where(meta["fold"] == "validation")[0]

test_meta = meta.iloc[test_idx].reset_index(drop=True)
xt = x[test_idx]
yt = y[test_idx]
bt_cached = base_cached[test_idx]

# Check exact sub7 deployment
sub7_zip = ROOT / "CCCCCC" / "submissions" / "sub7_cno_zeromean_head_adaptive_sps.zip"
sub7_extracted = ROOT / "CCCCCC" / "submissions" / "sub7_cno_zeromean_head_adaptive_sps"

# Import sub7 module
spec = importlib.util.spec_from_file_location("sub7_pkg", sub7_extracted / "submission.py")
sub7_pkg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sub7_pkg)

# Package Inference on test set
x3 = np.pad(xt, ((0,0),(0,0),(0,0),(0,0),(0,1)))
print("Running packaged sub7 prediction on test set (80 windows)...")
sub7_out = sub7_pkg.predict(x3)
p_pkg = sub7_out["prediction"][..., :2]
lower_pkg = sub7_out["lower"][..., :2]
upper_pkg = sub7_out["upper"][..., :2]

# Measure Live Research Path Inference (batch=16)
cno_model, h42, h43, norm_dict, dev, decay, *extra = sub7_pkg._get_models()

with torch.inference_mode():
    xx_t = torch.from_numpy(x3).to(dev)
    # Live CNO
    bb_list = []
    for i in range(0, len(xx_t), 16):
        batch_in = (xx_t[i:i+16] - torch.tensor(sub7_pkg._MEAN_IN, device=dev)) / torch.tensor(sub7_pkg._STD_IN, device=dev)
        bb_list.append(cno_model(batch_in) * torch.tensor(sub7_pkg._STD_TGT, device=dev) + torch.tensor(sub7_pkg._MEAN_TGT, device=dev))
    live_cno = torch.cat(bb_list)[..., :2]
    
    # Feature construction
    uv = xx_t[..., :2]
    mean_uv = uv.mean(1, keepdim=True)
    prior = uv.mean(1, keepdim=True) + decay * (uv[:, -1:] - uv.mean(1, keepdim=True))
    mask_hist = torch.all(uv == 0.0, dim=(1, 4))
    active_mask = (~mask_hist)[:, None, :, :, None]
    prior = prior * active_mask
    
    ff = torch.cat([sub7_pkg._pack_torch((q - norm_dict["nm"]) / norm_dict["ns"]) for q in [uv, live_cno, prior]], dim=1)
    
    # Head correction (ZeroMean, unmasked)
    cc_list = []
    for i in range(0, len(ff), 16):
        corr_batch = 0.5 * (h42(ff[i:i+16]) + h43(ff[i:i+16])) * norm_dict["rs"]
        cc_list.append(corr_batch)
    live_corr = torch.cat(cc_list)
    
    live_off = (live_cno + live_corr).cpu().numpy()
    live_on = ((live_cno + live_corr) * active_mask).cpu().numpy()

# Parity Checks
print("--- G0 PARITY AUDIT ---")
diff_live_vs_pkg = np.abs(live_on - p_pkg)
meaningful_idx = np.abs(p_pkg) >= 1e-3
max_abs_parity = float(diff_live_vs_pkg.max())
max_rel_parity = float((diff_live_vs_pkg[meaningful_idx] / np.abs(p_pkg[meaningful_idx])).max())

def calc_raw_metrics(pred, target):
    l2 = sc.rel_l2_per_sample(pred, target, 2)
    tke = sc.tke_rel_l2_per_sample(pred, target, 2)
    mvpe = sc.mvpe_rel_l2_per_sample(pred, target)
    return np.array([l2.mean(), tke.mean(), mvpe.mean()])

m_pkg = calc_raw_metrics(p_pkg, yt)
m_live = calc_raw_metrics(live_on, yt)
m_delta = m_live - m_pkg

parity_results = {
    "live_research_vs_package_batch16": {
        "max_abs_diff": max_abs_parity,
        "max_rel_diff_meaningful": max_rel_parity,
        "allclose_1e6": bool(np.allclose(live_on, p_pkg, atol=1e-6, rtol=1e-5)),
        "raw_metric_delta_l2_tke_mvpe": [float(v) for v in m_delta],
        "metric_delta_within_1e6": bool(np.max(np.abs(m_delta)) <= 1e-6)
    }
}
print("Parity live research vs package:", json.dumps(parity_results, indent=2))

# G1 — A1 ZeroMean Algebra Check
print("--- G1.A1 ZEROMEAN ALGEBRA AUDIT ---")
corr_np = live_corr.cpu().numpy()
base_np = live_cno.cpu().numpy()

corr_mean_t = corr_np.mean(axis=1) # mean over time
base_mean_t = base_np.mean(axis=1)
off_mean_t = live_off.mean(axis=1)
on_mean_t = live_on.mean(axis=1)

mask_np = mask_hist.cpu().numpy()

max_corr_mean = float(np.abs(corr_mean_t).max())
max_shift_before_mask = float(np.abs(off_mean_t - base_mean_t).max())
max_shift_after_mask = float(np.abs(on_mean_t - base_mean_t).max())
masked_pixels_final_mean = float(np.abs(on_mean_t[mask_np]).max())

algebra_results = {
    "max_abs_time_mean_correction_before_mask": max_corr_mean,
    "max_abs_time_mean_shift_base_plus_corr_before_mask": max_shift_before_mask,
    "satisfies_mean_preservation_before_mask_1e6": bool(max_shift_before_mask <= 1e-6),
    "max_abs_time_mean_shift_after_persistent_zero_mask": max_shift_after_mask,
    "masked_pixels_final_temporal_mean": masked_pixels_final_mean
}
print("A1 Algebra results:", json.dumps(algebra_results, indent=2))

# G1 — A2 Guardrail OFF vs ON
print("--- G1.A2 PERSISTENT-ZERO GUARDRAIL OFF VS ON ---")
e_off_per_sample = np.stack([sc.rel_l2_per_sample(live_off, yt, 2),
                             sc.tke_rel_l2_per_sample(live_off, yt, 2),
                             sc.mvpe_rel_l2_per_sample(live_off, yt)], axis=1)

e_on_per_sample = np.stack([sc.rel_l2_per_sample(live_on, yt, 2),
                            sc.tke_rel_l2_per_sample(live_on, yt, 2),
                            sc.mvpe_rel_l2_per_sample(live_on, yt)], axis=1)

delta_on_minus_off = e_on_per_sample - e_off_per_sample # negative is improvement
regressions_strict = (delta_on_minus_off > 0).mean(axis=0)
regressions_1e7 = (delta_on_minus_off > 1e-7).mean(axis=0)

num_masked_pixels = int(mask_np.sum())
total_pixels = int(mask_np.size)
fraction_masked = float(num_masked_pixels / total_pixels)

# Check future target nonzero in masked pixels
future_target_nonzero = np.any(yt != 0.0, axis=(1, 4)) # (N, 32, 64)
fraction_masked_future_nonzero = float(future_target_nonzero[mask_np].mean())

# MSE inside vs outside mask
sel_mask = np.broadcast_to(mask_np[:, None, :, :, None], yt.shape)
mse_off_inside = float(np.mean((live_off - yt)[sel_mask] ** 2))
mse_on_inside = float(np.mean((live_on - yt)[sel_mask] ** 2))
mse_off_outside = float(np.mean((live_off - yt)[~sel_mask] ** 2))
mse_on_outside = float(np.mean((live_on - yt)[~sel_mask] ** 2))

# Per trajectory breakdown
traj_table = test_meta.copy()
for idx, name in enumerate(["RelL2", "TKE", "MVPE"]):
    traj_table[f"{name}_off"] = e_off_per_sample[:, idx]
    traj_table[f"{name}_on"] = e_on_per_sample[:, idx]
    traj_table[f"{name}_delta"] = delta_on_minus_off[:, idx]

traj_summary = traj_table.groupby("file")[[f"{n}_{s}" for n in ["RelL2", "TKE", "MVPE"] for s in ["off", "on", "delta"]]].mean()
traj_summary.to_csv(OUT / "guardrail_per_trajectory.csv")

# Per horizon breakdown
horizon_rows = []
for h in range(1, 21):
    sub_off = live_off[:, :h]
    sub_on = live_on[:, :h]
    sub_y = yt[:, :h]
    
    l2_off = float(sc.rel_l2_per_sample(sub_off, sub_y, 2).mean())
    l2_on = float(sc.rel_l2_per_sample(sub_on, sub_y, 2).mean())
    
    mvpe_off = float(sc.mvpe_rel_l2_per_sample(sub_off, sub_y).mean())
    mvpe_on = float(sc.mvpe_rel_l2_per_sample(sub_on, sub_y).mean())
    
    if h > 1:
        tke_off = float(sc.tke_rel_l2_per_sample(sub_off, sub_y, 2).mean())
        tke_on = float(sc.tke_rel_l2_per_sample(sub_on, sub_y, 2).mean())
    else:
        tke_off = None
        tke_on = None
    
    single_l2_off = float(sc.rel_l2_per_sample(live_off[:, h-1:h], yt[:, h-1:h], 2).mean())
    single_l2_on = float(sc.rel_l2_per_sample(live_on[:, h-1:h], yt[:, h-1:h], 2).mean())
    
    horizon_rows.append({
        "horizon": h,
        "RelL2_off": l2_off, "RelL2_on": l2_on, "RelL2_delta": l2_on - l2_off,
        "TKE_off": tke_off, "TKE_on": tke_on, "TKE_delta": (tke_on - tke_off) if tke_on is not None else None,
        "MVPE_off": mvpe_off, "MVPE_on": mvpe_on, "MVPE_delta": mvpe_on - mvpe_off,
        "single_frame_RelL2_delta": single_l2_on - single_l2_off
    })

pd.DataFrame(horizon_rows).to_csv(OUT / "guardrail_per_horizon.csv", index=False)

guardrail_results = {
    "aggregate_errors": {
        "OFF": [float(v) for v in e_off_per_sample.mean(axis=0)],
        "ON": [float(v) for v in e_on_per_sample.mean(axis=0)],
        "delta_ON_minus_OFF": [float(v) for v in delta_on_minus_off.mean(axis=0)]
    },
    "regression_rates_strict": [float(v) for v in regressions_strict],
    "regression_rates_1e7": [float(v) for v in regressions_1e7],
    "masked_pixels_count": num_masked_pixels,
    "total_pixels_count": total_pixels,
    "fraction_masked": fraction_masked,
    "fraction_masked_future_nonzero": fraction_masked_future_nonzero,
    "region_mse": {
        "inside_mask_off": mse_off_inside,
        "inside_mask_on": mse_on_inside,
        "outside_mask_off": mse_off_outside,
        "outside_mask_on": mse_on_outside
    }
}
print("A2 Guardrail results:", json.dumps(guardrail_results, indent=2))

# G1 — A3 Constant vs Adaptive SPS
print("--- G1.A3 CONSTANT VS ADAPTIVE SPS AUDIT ---")
# Constant SPS (sub6 rule)
hw_const = 0.85 * 0.010925
lower_const = p_pkg - hw_const
upper_const = p_pkg + hw_const

# Time interval calculation on GPU
torch.cuda.synchronize()
# Warmup
hw_adaptive_t = torch.clamp(sub7_pkg._state["base_hw"] + 0.85 * uv.std(dim=1, keepdim=True) * sub7_pkg._state["h_factors"],
                            min=sub7_pkg._state["base_hw"], max=sub7_pkg._state["max_hw"])
torch.cuda.synchronize()

times_const = []
for _ in range(5):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    hw_c = torch.full_like(uv, 0.85 * 0.010925)
    torch.cuda.synchronize()
    times_const.append(time.perf_counter() - t0)

times_adapt = []
for _ in range(5):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    hw_a = torch.clamp(sub7_pkg._state["base_hw"] + 0.85 * uv.std(dim=1, keepdim=True) * sub7_pkg._state["h_factors"],
                       min=sub7_pkg._state["base_hw"], max=sub7_pkg._state["max_hw"])
    torch.cuda.synchronize()
    times_adapt.append(time.perf_counter() - t0)

sps_const, cov_const = sc.aggregate_sps(p_pkg, yt, 2, lower=lower_const, upper=upper_const)
sps_adapt, cov_adapt = sc.aggregate_sps(p_pkg, yt, 2, lower=lower_pkg, upper=upper_pkg)

width_const = upper_const - lower_const
width_adapt = upper_pkg - lower_pkg

interval_results = {
    "constant_sps": {
        "sps_score": sc.score_sps(sps_const),
        "coverage": float(cov_const),
        "mean_width": float(width_const.mean(dtype=np.float64)),
        "median_width": float(np.median(width_const)),
        "width_channel_u_v": [float(v) for v in width_const.mean(axis=(0,1,2,3), dtype=np.float64)],
        "gpu_time_median_sec": float(np.median(times_const)),
        "gpu_time_range_sec": [min(times_const), max(times_const)]
    },
    "adaptive_sps": {
        "sps_score": sc.score_sps(sps_adapt),
        "coverage": float(cov_adapt),
        "mean_width": float(width_adapt.mean(dtype=np.float64)),
        "median_width": float(np.median(width_adapt)),
        "width_channel_u_v": [float(v) for v in width_adapt.mean(axis=(0,1,2,3), dtype=np.float64)],
        "gpu_time_median_sec": float(np.median(times_adapt)),
        "gpu_time_range_sec": [min(times_adapt), max(times_adapt)]
    }
}
print("A3 Interval results:", json.dumps(interval_results, indent=2))

# G1 — A4 SPS Calibration Structure
print("--- G1.A4 RESIDUAL SPS CALIBRATION STRUCTURE ---")
# Training history std quartile edges (ddof=1)
train_x = x[train_idx]
std_train = np.std(train_x, axis=1, ddof=1) # (339, 32, 64, 2)
# Compute quantiles separately for u and v
edges_u = np.quantile(std_train[..., 0], [0.25, 0.50, 0.75])
edges_v = np.quantile(std_train[..., 1], [0.25, 0.50, 0.75])

std_test = np.std(xt, axis=1, ddof=1) # (80, 32, 64, 2)

scored = yt != 0.0
inside = (yt >= lower_pkg) & (yt <= upper_pkg)
width = width_adapt

# Compute elementwise SPS contribution
# Accuracy factors per window
e_on = e_on_per_sample
factors = np.sum([w * (0.5 / (0.5 + e_on[:, j])) for j, w in enumerate([0.5, 0.3, 0.2])], axis=0)[:, None, None, None, None]
nil = width / sc.SIGMA_GLOBAL
elem_contrib = inside * np.exp(-nil) * factors

cal_rows = []
def add_group(dimension, group_label, sel_mask):
    total_scored_sel = sel_mask & scored
    n = int(total_scored_sel.sum())
    if n > 0:
        cov = float(inside[total_scored_sel].mean())
        mae = float(np.abs(p_pkg - yt)[total_scored_sel].mean())
        w_mean = float(width[total_scored_sel].mean())
        contrib_score = float(100.0 * elem_contrib[total_scored_sel].mean())
        cal_rows.append({
            "dimension": dimension,
            "group": str(group_label),
            "n_scored": n,
            "coverage": cov,
            "mae": mae,
            "mean_width": w_mean,
            "sps_contribution_score": contrib_score
        })

# 1. By channel
add_group("channel", "u", (np.arange(2)[None, None, None, None, :] == 0))
add_group("channel", "v", (np.arange(2)[None, None, None, None, :] == 1))

# 2. By history std quantiles
for b in range(4):
    if b == 0:
        sel_u = std_test[..., 0] <= edges_u[0]
        sel_v = std_test[..., 1] <= edges_v[0]
    elif b == 1:
        sel_u = (std_test[..., 0] > edges_u[0]) & (std_test[..., 0] <= edges_u[1])
        sel_v = (std_test[..., 1] > edges_v[0]) & (std_test[..., 1] <= edges_v[1])
    elif b == 2:
        sel_u = (std_test[..., 0] > edges_u[1]) & (std_test[..., 0] <= edges_u[2])
        sel_v = (std_test[..., 1] > edges_v[1]) & (std_test[..., 1] <= edges_v[2])
    else:
        sel_u = std_test[..., 0] > edges_u[2]
        sel_v = std_test[..., 1] > edges_v[2]
    
    # Broadcast to (80, 20, 32, 64, 2)
    sel_u_full = np.zeros_like(yt, dtype=bool)
    sel_u_full[..., 0] = sel_u[:, None, :, :]
    add_group("history_std_quantile", f"u:Q{b+1}", sel_u_full)
    
    sel_v_full = np.zeros_like(yt, dtype=bool)
    sel_v_full[..., 1] = sel_v[:, None, :, :]
    add_group("history_std_quantile", f"v:Q{b+1}", sel_v_full)

# 3. By horizon
for h in range(20):
    sel_h = np.zeros_like(yt, dtype=bool)
    sel_h[:, h, ...] = True
    add_group("horizon", f"h{h+1}", sel_h)

# 4. By spatial region (persistent_zero_mask and complement)
sel_mask_region = np.broadcast_to(mask_np[:, None, :, :, None], yt.shape)
add_group("region", "persistent_zero_mask", sel_mask_region)
add_group("region", "complement", ~sel_mask_region)

# 5. By Re and AoA (descriptive)
for re_val in sorted(test_meta["Re"].unique()):
    sel_re = np.zeros_like(yt, dtype=bool)
    idx_re = np.where(test_meta["Re"] == re_val)[0]
    sel_re[idx_re] = True
    add_group("Re", str(re_val), sel_re)

for aoa_val in sorted(test_meta["AoA"].unique()):
    sel_aoa = np.zeros_like(yt, dtype=bool)
    idx_aoa = np.where(test_meta["AoA"] == aoa_val)[0]
    sel_aoa[idx_aoa] = True
    add_group("AoA", str(aoa_val), sel_aoa)

cal_df = pd.DataFrame(cal_rows)
cal_df.to_csv(OUT / "calibration_diagnostics.csv", index=False)
print("Saved calibration_diagnostics.csv")
print(cal_df[cal_df["dimension"].isin(["channel", "history_std_quantile"])].to_string(index=False))

# Save full audit summary json
full_summary = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "parity": parity_results,
    "algebra_a1": algebra_results,
    "guardrail_a2": guardrail_results,
    "intervals_a3": interval_results,
    "training_std_edges": {
        "u": [float(v) for v in edges_u],
        "v": [float(v) for v in edges_v]
    }
}
with open(OUT / "full_phase1_audit_summary.json", "w") as f:
    json.dump(full_summary, f, indent=2)

print("=" * 80)
print("PHASE 1 FULL AUDIT COMPLETE AND PERSISTED IN gemini-10-9")
print("=" * 80)
