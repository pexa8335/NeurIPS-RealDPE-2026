from pathlib import Path
import sys, importlib.util, time, json
import numpy as np
import pandas as pd
import torch

ROOT = Path("d:/Project/NeurIPS")
OUT = ROOT / "gemini-10-9"
OUT.mkdir(exist_ok=True)

# Starting kit scoring
sys.path.insert(0, str(ROOT / 'realpde_t1_starting_kit_v9/realpde_t1_starting_kit_v9'))
import scoring as sc

# Exact sub7 pipeline
spec = importlib.util.spec_from_file_location('exact', ROOT / 'research/phase1_audit/exact_sub7/submission.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)

# Load manifests and feature cache
manifest = pd.read_csv(ROOT / 'research/exp_zero_mean_head/artifacts/manifest.csv')
feature_cache = np.load(ROOT / 'research/exp_zero_mean_head/artifacts/feature_cache.npz')

train_idx = np.flatnonzero(manifest.fold.eq('train'))
val_idx = np.flatnonzero(manifest.fold.eq('validation'))
test_idx = np.flatnonzero(manifest.fold.eq('test'))

# Calculate training Q-edges for u
x_train = feature_cache['x'][train_idx]
std_u_train = np.std(x_train[..., 0], axis=1, ddof=1)
q_edges_u = np.quantile(std_u_train, [0.25, 0.50, 0.75])
print(f"Training history-std Q edges for u: {q_edges_u}")

protocol = {
    "goal": "Test narrow hypothesis: current history-std contribution to u-channel interval width is under-scaled for high-variance inputs.",
    "frozen_components": {
        "point_predictor": "Exact sub7 (Frozen CNO + Dual-Seed ZeroMean Head L0.30 + Persistent-Zero Guardrail)",
        "v_channel_formula": "hw_v = clip(0.0032 + 0.85 * std_v * sqrt(h / 10), min=0.0032, max=0.040)",
        "horizon_scaling": "sqrt(h / 10.0)",
        "base_hw_u": 0.0075,
        "max_hw": 0.040,
        "precision": "float32",
        "batch_size": 16
    },
    "candidates": {
        "baseline": "hw_raw_u = 0.0075 + 0.85 * std_u * sqrt(h/10)",
        "C1_mild": "hw_raw_u = 0.0075 + 0.95 * std_u * sqrt(h/10)",
        "C2_moderate": "hw_raw_u = 0.0075 + 1.05 * std_u * sqrt(h/10)",
        "C3_q4_boost": "hw_raw_u = 0.0075 + (0.85 + 0.20 * 1[std_u > 0.00949892]) * std_u * sqrt(h/10)"
    },
    "selection_partition": "validation (58 windows)",
    "evaluation_partition": "development test (80 windows)",
    "primary_endpoint": "SPS score",
    "tie_rule": "Retain current sub7 bounds"
}
(OUT / 'g2_uq_protocol.json').write_text(json.dumps(protocol, indent=2))

def compute_bounds(x_in, p_point, mode):
    uin = x_in[..., 0]
    vin = x_in[..., 1]
    std_u = np.std(uin, axis=1, ddof=1, keepdims=True)
    std_v = np.std(vin, axis=1, ddof=1, keepdims=True)
    h_factors = np.sqrt(np.arange(1, 21, dtype=np.float32) / 10.0).reshape(1, 20, 1, 1)

    hw_v = np.clip(0.0032 + 0.85 * std_v * h_factors, 0.0032, 0.040)

    if mode == 'baseline':
        hw_u_raw = 0.0075 + 0.85 * std_u * h_factors
    elif mode == 'C1_mild':
        hw_u_raw = 0.0075 + 0.95 * std_u * h_factors
    elif mode == 'C2_moderate':
        hw_u_raw = 0.0075 + 1.05 * std_u * h_factors
    elif mode == 'C3_q4_boost':
        scale_u = 0.85 + 0.20 * (std_u > q_edges_u[2]).astype(np.float32)
        hw_u_raw = 0.0075 + scale_u * std_u * h_factors
    else:
        raise ValueError(f"Unknown mode: {mode}")

    hw_u = np.clip(hw_u_raw, 0.0075, 0.040)
    hw = np.stack([hw_u, hw_v], axis=-1)
    lo = p_point - hw
    hi = p_point + hw
    return lo, hi, hw

def evaluate_partition(indices, name):
    x_part = feature_cache['x'][indices]
    y_part = feature_cache['y'][indices]
    x_part3 = np.pad(x_part, ((0,0),(0,0),(0,0),(0,0),(0,1)))
    res = s.predict(x_part3)
    p_point = res['prediction'][..., :2]

    rel_l2 = sc.rel_l2_per_sample(p_point, y_part, 2).mean()
    tke = sc.tke_rel_l2_per_sample(p_point, y_part, 2).mean()
    mvpe = sc.mvpe_rel_l2_per_sample(p_point, y_part).mean()

    results = {}
    base_sps = None

    for mode in ['baseline', 'C1_mild', 'C2_moderate', 'C3_q4_boost']:
        t_list = []
        for _ in range(5):
            t0 = time.perf_counter()
            lo, hi, hw = compute_bounds(x_part, p_point, mode)
            t1 = time.perf_counter()
            t_list.append(t1 - t0)
        med_lat_ms = float(np.median(t_list) * 1000)

        sps_raw, cov = sc.aggregate_sps(p_point, y_part, 2, lower=lo, upper=hi)
        sps_score = sc.score_sps(sps_raw)

        if mode == 'baseline':
            base_sps = sps_score

        scored = y_part != 0
        inside = (y_part >= lo) & (y_part <= hi)
        width = (hi - lo)

        std_u_sample = np.std(x_part[..., 0], axis=1, ddof=1)
        u_q_cov = {}
        for b in range(4):
            if b == 0:
                mask_q = std_u_sample <= q_edges_u[0]
            elif b == 1:
                mask_q = (std_u_sample > q_edges_u[0]) & (std_u_sample <= q_edges_u[1])
            elif b == 2:
                mask_q = (std_u_sample > q_edges_u[1]) & (std_u_sample <= q_edges_u[2])
            else:
                mask_q = std_u_sample > q_edges_u[2]
            mask_q_full = mask_q[:, None, :, :, None] & scored[..., :1]
            cov_q = float(inside[..., :1][mask_q_full].mean()) if mask_q_full.any() else 0.0
            u_q_cov[f"Q{b+1}"] = cov_q

        h_cov = []
        for h in range(20):
            sel_h = scored[:, h:h+1]
            cov_h = float(inside[:, h:h+1][sel_h].mean()) if sel_h.any() else 0.0
            h_cov.append(cov_h)

        results[mode] = {
            "mode": mode,
            "sps_score": float(sps_score),
            "delta_sps": float(sps_score - base_sps),
            "coverage": float(cov),
            "mean_width": float(width.mean(dtype=np.float64)),
            "median_width": float(np.median(width)),
            "u_width_mean": float(width[..., 0].mean(dtype=np.float64)),
            "v_width_mean": float(width[..., 1].mean(dtype=np.float64)),
            "u_q_coverage": u_q_cov,
            "horizon_coverage": h_cov,
            "latency_ms": med_lat_ms,
            "rel_l2": float(rel_l2),
            "tke": float(tke),
            "mvpe": float(mvpe)
        }
        print(f"[{name}] {mode:12s}: SPS={sps_score:.6f} (delta={sps_score-base_sps:+.6f}), cov={cov*100:.2f}%, mean_w={width.mean():.6f}")

    return results

print("\n--- EVALUATING VALIDATION PARTITION (Selection) ---")
val_results = evaluate_partition(val_idx, "VALIDATION")
(OUT / 'g2_uq_val_results.json').write_text(json.dumps(val_results, indent=2))

print("\n--- EVALUATING DEVELOPMENT TEST PARTITION (Diagnostic Comparison) ---")
test_results = evaluate_partition(test_idx, "TEST")
(OUT / 'g2_uq_test_results.json').write_text(json.dumps(test_results, indent=2))

summary_rows = []
for mode in ['baseline', 'C1_mild', 'C2_moderate', 'C3_q4_boost']:
    vr = val_results[mode]
    tr = test_results[mode]
    summary_rows.append({
        "Configuration": mode,
        "Val SPS": vr['sps_score'],
        "Val Delta SPS": vr['delta_sps'],
        "Val Coverage": vr['coverage'] * 100,
        "Test SPS": tr['sps_score'],
        "Test Delta SPS": tr['delta_sps'],
        "Test Coverage": tr['coverage'] * 100,
        "Test Mean Width": tr['mean_width'],
        "Test Median Width": tr['median_width'],
        "Test u Width": tr['u_width_mean'],
        "Test v Width": tr['v_width_mean'],
        "Test u Q1 Cov": tr['u_q_coverage']['Q1'] * 100,
        "Test u Q2 Cov": tr['u_q_coverage']['Q2'] * 100,
        "Test u Q3 Cov": tr['u_q_coverage']['Q3'] * 100,
        "Test u Q4 Cov": tr['u_q_coverage']['Q4'] * 100,
        "Latency ms": tr['latency_ms']
    })

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(OUT / 'g2_uq_comparison_summary.csv', index=False)
print("\n--- G2-UQ COMPARISON SUMMARY IN gemini-10-9 ---")
print(df_summary.to_string(index=False))
