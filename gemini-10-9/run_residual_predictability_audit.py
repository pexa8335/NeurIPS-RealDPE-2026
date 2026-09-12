import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

ROOT = Path("d:/Project/NeurIPS")
OUT = ROOT / "gemini-10-9"
OUT.mkdir(exist_ok=True)

# Starting kit scoring
sys.path.insert(0, str(ROOT / "realpde_t1_starting_kit_v9" / "realpde_t1_starting_kit_v9"))
import scoring as sc

# Load manifest and feature cache
manifest = pd.read_csv(ROOT / "research/exp_zero_mean_head/artifacts/manifest.csv")
fcache = np.load(ROOT / "research/exp_zero_mean_head/artifacts/feature_cache.npz")

x = fcache["x"] # [477, 20, 32, 64, 2]
y = fcache["y"] # [477, 20, 32, 64, 2]
base = fcache["base"] # [477, 20, 32, 64, 2] CNO base predictions

test_idx = np.flatnonzero(manifest["fold"] == "test")
train_idx = np.flatnonzero(manifest["fold"] == "train")
val_idx = np.flatnonzero(manifest["fold"] == "validation")

print(f"Loaded {len(test_idx)} test windows, {len(train_idx)} train windows, {len(val_idx)} val windows.")

# Focus on the 80 development test windows for oracle decomposition
y_test = y[test_idx] # [80, 20, 32, 64, 2]
base_test = base[test_idx] # [80, 20, 32, 64, 2]
x_test = x[test_idx] # [80, 20, 32, 64, 2]

# Temporal mean and fluctuation decomposition
y_mean = np.mean(y_test, axis=1, keepdims=True) # [80, 1, 32, 64, 2]
y_fluc = y_test - y_mean # [80, 20, 32, 64, 2]

base_mean = np.mean(base_test, axis=1, keepdims=True)
base_fluc = base_test - base_mean # [80, 20, 32, 64, 2]

# Baseline Error Breakdown
err_total = (base_test - y_test)**2
err_mean = (base_mean - y_mean)**2 # broadcast across 20 frames
err_fluc = (base_fluc - y_fluc)**2

mse_total = float(np.mean(err_total))
mse_mean = float(np.mean(err_mean))
mse_fluc = float(np.mean(err_fluc))

pct_mean = (mse_mean / mse_total) * 100.0
pct_fluc = (mse_fluc / mse_total) * 100.0

print(f"Total MSE: {mse_total:.8f}")
print(f"Mean MSE:  {mse_mean:.8f} ({pct_mean:.2f}% of total)")
print(f"Fluc MSE:  {mse_fluc:.8f} ({pct_fluc:.2f}% of total)")

def calc_tke_error(p, target):
    tke_sample = sc.tke_rel_l2_per_sample(p, target, 2)
    return float(np.mean(tke_sample))

baseline_tke_error = calc_tke_error(base_test, y_test)
print(f"Baseline TKE RelL2 Error: {baseline_tke_error:.6f}")

fluc_energy_target = float(np.mean(y_fluc**2))
fluc_energy_base = float(np.mean(base_fluc**2))
energy_ratio = fluc_energy_base / fluc_energy_target
print(f"Fluctuation Energy (CNO / Target): {fluc_energy_base:.8f} / {fluc_energy_target:.8f} (ratio: {energy_ratio:.4f})")

results = []

# --- 1. Global Amplitude Oracle (Per-window scalar alpha) ---
fluc_pred_1 = np.zeros_like(base_fluc)
alphas = []
for i in range(len(test_idx)):
    num = np.sum(base_fluc[i] * y_fluc[i])
    den = np.sum(base_fluc[i]**2) + 1e-10
    a = float(num / den)
    alphas.append(a)
    fluc_pred_1[i] = a * base_fluc[i]

mse_1 = float(np.mean((fluc_pred_1 - y_fluc)**2))
rec_1 = ((mse_fluc - mse_1) / mse_fluc) * 100.0
tke_1 = calc_tke_error(base_mean + fluc_pred_1, y_test)
results.append({
    "transformation": "1. Global Amplitude Oracle (Scalar alpha per window)",
    "degrees_of_freedom": "1 scalar / window",
    "recovered_fluc_error_pct": rec_1,
    "remaining_fluc_mse": mse_1,
    "tke_rel_l2_error": tke_1,
    "notes": f"Mean optimal alpha = {np.mean(alphas):.4f} (median: {np.median(alphas):.4f})"
})

# --- 2. Local Spatial Amplitude Map (Per-pixel scaling A(x,y) per window) ---
fluc_pred_2 = np.zeros_like(base_fluc)
for i in range(len(test_idx)):
    num = np.sum(base_fluc[i] * y_fluc[i], axis=0, keepdims=True) # [1, 32, 64, 2]
    den = np.sum(base_fluc[i]**2, axis=0, keepdims=True) + 1e-10
    A = np.clip(num / den, -5.0, 5.0)
    fluc_pred_2[i] = A * base_fluc[i]

mse_2 = float(np.mean((fluc_pred_2 - y_fluc)**2))
rec_2 = ((mse_fluc - mse_2) / mse_fluc) * 100.0
tke_2 = calc_tke_error(base_mean + fluc_pred_2, y_test)
results.append({
    "transformation": "2. Local Spatial Amplitude Map (Per-pixel A(x,y) per window)",
    "degrees_of_freedom": "32x64x2 = 4,096 params / window",
    "recovered_fluc_error_pct": rec_2,
    "remaining_fluc_mse": mse_2,
    "tke_rel_l2_error": tke_2,
    "notes": "Optimal per-pixel spatial envelope matching"
})

# --- 3. Temporal Phase Shift Oracle (Lag tau in [-5, +5] frames) ---
fluc_pred_3 = np.zeros_like(base_fluc)
lags_chosen = []
for i in range(len(test_idx)):
    best_lag = 0
    best_err = float("inf")
    best_shifted = base_fluc[i]
    for lag in range(-5, 6):
        shifted = np.roll(base_fluc[i], lag, axis=0)
        err = np.mean((shifted - y_fluc[i])**2)
        if err < best_err:
            best_err = err
            best_lag = lag
            best_shifted = shifted
    lags_chosen.append(best_lag)
    fluc_pred_3[i] = best_shifted

mse_3 = float(np.mean((fluc_pred_3 - y_fluc)**2))
rec_3 = ((mse_fluc - mse_3) / mse_fluc) * 100.0
tke_3 = calc_tke_error(base_mean + fluc_pred_3, y_test)
results.append({
    "transformation": "3. Temporal Phase Shift Oracle (Lag tau in [-5, 5] frames)",
    "degrees_of_freedom": "1 discrete lag / window",
    "recovered_fluc_error_pct": rec_3,
    "remaining_fluc_mse": mse_3,
    "tke_rel_l2_error": tke_3,
    "notes": f"Lags distribution: {pd.Series(lags_chosen).value_counts().to_dict()}"
})

# --- 4. Spatial Rigid Shift Oracle ((dx, dy) in [-4, +4]^2) via fast torch.roll on GPU ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_fluc_t = torch.from_numpy(base_fluc).to(device) # [80, 20, 32, 64, 2]
y_fluc_t = torch.from_numpy(y_fluc).to(device)

fluc_pred_4_t = torch.zeros_like(base_fluc_t)
shifts_chosen = []

for i in range(len(test_idx)):
    b_i = base_fluc_t[i] # [20, 32, 64, 2]
    y_i = y_fluc_t[i]
    best_err = float("inf")
    best_shifted = b_i
    best_shift = (0, 0)
    for dy in range(-3, 4):
        for dx in range(-4, 5):
            shifted = torch.roll(b_i, shifts=(dy, dx), dims=(1, 2))
            err = torch.mean((shifted - y_i)**2).item()
            if err < best_err:
                best_err = err
                best_shift = (dy, dx)
                best_shifted = shifted
    shifts_chosen.append(best_shift)
    fluc_pred_4_t[i] = best_shifted

fluc_pred_4 = fluc_pred_4_t.cpu().numpy()
mse_4 = float(np.mean((fluc_pred_4 - y_fluc)**2))
rec_4 = ((mse_fluc - mse_4) / mse_fluc) * 100.0
tke_4 = calc_tke_error(base_mean + fluc_pred_4, y_test)
results.append({
    "transformation": "4. Spatial Rigid Alignment Oracle ((dx, dy) in [-4, 4] pixels)",
    "degrees_of_freedom": "2 discrete shifts / window",
    "recovered_fluc_error_pct": rec_4,
    "remaining_fluc_mse": mse_4,
    "tke_rel_l2_error": tke_4,
    "notes": "Coherent spatial advection offset search"
})

# --- 5. Combined Temporal Shift + Local Spatial Amplitude Map ---
fluc_pred_5 = np.zeros_like(base_fluc)
for i in range(len(test_idx)):
    shifted = fluc_pred_3[i] # already has best temporal shift
    num = np.sum(shifted * y_fluc[i], axis=0, keepdims=True)
    den = np.sum(shifted**2, axis=0, keepdims=True) + 1e-10
    A = np.clip(num / den, -5.0, 5.0)
    fluc_pred_5[i] = A * shifted

mse_5 = float(np.mean((fluc_pred_5 - y_fluc)**2))
rec_5 = ((mse_fluc - mse_5) / mse_fluc) * 100.0
tke_5 = calc_tke_error(base_mean + fluc_pred_5, y_test)
results.append({
    "transformation": "5. Combined Temporal Phase Shift + Local Spatial Amplitude Map",
    "degrees_of_freedom": "1 lag + 4,096 spatial weights / window",
    "recovered_fluc_error_pct": rec_5,
    "remaining_fluc_mse": mse_5,
    "tke_rel_l2_error": tke_5,
    "notes": "Joint phase synchronization + spatial envelope matching"
})

# --- 6. Combined Spatial Shift + Local Spatial Amplitude Map ---
fluc_pred_6 = np.zeros_like(base_fluc)
for i in range(len(test_idx)):
    shifted = fluc_pred_4[i] # already has best spatial shift
    num = np.sum(shifted * y_fluc[i], axis=0, keepdims=True)
    den = np.sum(shifted**2, axis=0, keepdims=True) + 1e-10
    A = np.clip(num / den, -5.0, 5.0)
    fluc_pred_6[i] = A * shifted

mse_6 = float(np.mean((fluc_pred_6 - y_fluc)**2))
rec_6 = ((mse_fluc - mse_6) / mse_fluc) * 100.0
tke_6 = calc_tke_error(base_mean + fluc_pred_6, y_test)
results.append({
    "transformation": "6. Combined Spatial Alignment + Local Spatial Amplitude Map",
    "degrees_of_freedom": "2 shifts + 4,096 spatial weights / window",
    "recovered_fluc_error_pct": rec_6,
    "remaining_fluc_mse": mse_6,
    "tke_rel_l2_error": tke_6,
    "notes": "Joint spatial displacement + envelope matching"
})

# --- 7. Combined Temporal Phase Shift + Spatial Shift + Local Amplitude Map ---
fluc_pred_7 = np.zeros_like(base_fluc)
for i in range(len(test_idx)):
    b_i = base_fluc_t[i]
    y_i = y_fluc_t[i]
    best_err = float("inf")
    best_shifted = b_i
    for lag in range(-3, 4):
        rolled_t = torch.roll(b_i, shifts=lag, dims=0)
        for dy in range(-2, 3):
            for dx in range(-3, 4):
                shifted = torch.roll(rolled_t, shifts=(dy, dx), dims=(1, 2))
                err = torch.mean((shifted - y_i)**2).item()
                if err < best_err:
                    best_err = err
                    best_shifted = shifted
    s_np = best_shifted.cpu().numpy()
    num = np.sum(s_np * y_fluc[i], axis=0, keepdims=True)
    den = np.sum(s_np**2, axis=0, keepdims=True) + 1e-10
    A = np.clip(num / den, -5.0, 5.0)
    fluc_pred_7[i] = A * s_np

mse_7 = float(np.mean((fluc_pred_7 - y_fluc)**2))
rec_7 = ((mse_fluc - mse_7) / mse_fluc) * 100.0
tke_7 = calc_tke_error(base_mean + fluc_pred_7, y_test)
results.append({
    "transformation": "7. Joint Full Alignment (Time Lag + Spatial Shift + Local Amplitude)",
    "degrees_of_freedom": "3 search params + 4,096 spatial weights / window",
    "recovered_fluc_error_pct": rec_7,
    "remaining_fluc_mse": mse_7,
    "tke_rel_l2_error": tke_7,
    "notes": "Comprehensive space-time alignment + local envelope"
})

# --- 8. POD Mode Energy Decomposition of Ground Truth Fluctuation ---
pod_energies_top1 = []
pod_energies_top3 = []
pod_energies_top5 = []
cno_proj_top1_r2 = []

for i in range(len(test_idx)):
    Y_mat = y_fluc[i].reshape(20, -1) # [20, 4096]
    U, S, Vt = np.linalg.svd(Y_mat, full_matrices=False) # S: [20]
    total_var = np.sum(S**2)
    top1 = (S[0]**2) / total_var
    top3 = np.sum(S[:3]**2) / total_var
    top5 = np.sum(S[:5]**2) / total_var
    pod_energies_top1.append(top1)
    pod_energies_top3.append(top3)
    pod_energies_top5.append(top5)
    
    # Project CNO fluctuation onto top 1 POD spatial mode
    v1 = Vt[0]
    C_mat = base_fluc[i].reshape(20, -1)
    proj_cno = C_mat @ v1
    proj_true = Y_mat @ v1
    corr = np.corrcoef(proj_cno, proj_true)[0, 1] if np.std(proj_cno) > 1e-6 and np.std(proj_true) > 1e-6 else 0.0
    cno_proj_top1_r2.append(corr**2)

print(f"\nTop 1 POD mode captures: {np.mean(pod_energies_top1)*100:.1f}% of fluctuation energy")
print(f"Top 3 POD modes capture: {np.mean(pod_energies_top3)*100:.1f}% of fluctuation energy")
print(f"Top 5 POD modes capture: {np.mean(pod_energies_top5)*100:.1f}% of fluctuation energy")
print(f"CNO correlation^2 (R^2) with top-1 POD mode temporal dynamics: {np.mean(cno_proj_top1_r2)*100:.1f}%")

# --- 9. Theoretical Coherent Dynamics Ceiling (Top-3 POD Modes of Future) ---
fluc_pred_pod3 = np.zeros_like(y_fluc)
for i in range(len(test_idx)):
    Y_mat = y_fluc[i].reshape(20, -1)
    U, S, Vt = np.linalg.svd(Y_mat, full_matrices=False)
    Y_recon = (U[:, :3] * S[:3]) @ Vt[:3, :]
    fluc_pred_pod3[i] = Y_recon.reshape(20, 32, 64, 2)

mse_pod3 = float(np.mean((fluc_pred_pod3 - y_fluc)**2))
rec_pod3 = ((mse_fluc - mse_pod3) / mse_fluc) * 100.0
tke_pod3 = calc_tke_error(base_mean + fluc_pred_pod3, y_test)

results.append({
    "transformation": "8. Theoretical Coherent Dynamics Ceiling (Top-3 POD Modes of Future)",
    "degrees_of_freedom": "Oracle top-3 spatial/temporal coherent modes",
    "recovered_fluc_error_pct": rec_pod3,
    "remaining_fluc_mse": mse_pod3,
    "tke_rel_l2_error": tke_pod3,
    "notes": f"Reconstituting top-3 coherent structures recovers {rec_pod3:.1f}% error"
})

# --- 10. Causal History Predictability Ceiling (Linear Regression from 20-frame Real History) ---
print("\nFitting Causal Linear Predictor from 20-frame Real History...")
x_train = x[train_idx]
y_train = y[train_idx]
y_train_mean = np.mean(y_train, axis=1, keepdims=True)
y_train_fluc = y_train - y_train_mean

X_tr_flat = x_train.reshape(len(train_idx), -1)
X_te_flat = x_test.reshape(len(test_idx), -1)

pca_hist = PCA(n_components=32, random_state=42)
Z_tr = pca_hist.fit_transform(X_tr_flat)
Z_te = pca_hist.transform(X_te_flat)

Y_tr_flat = y_train_fluc.reshape(len(train_idx), -1)
Y_te_flat = y_fluc.reshape(len(test_idx), -1)

pca_target = PCA(n_components=16, random_state=42)
W_tr = pca_target.fit_transform(Y_tr_flat)
W_te = pca_target.transform(Y_te_flat)

ridge = Ridge(alpha=100.0)
ridge.fit(Z_tr, W_tr)
W_te_pred = ridge.predict(Z_te)
Y_te_pred_flat = pca_target.inverse_transform(W_te_pred)
fluc_pred_causal = Y_te_pred_flat.reshape(len(test_idx), 20, 32, 64, 2)

mse_causal = float(np.mean((fluc_pred_causal - y_fluc)**2))
rec_causal = ((mse_fluc - mse_causal) / mse_fluc) * 100.0
tke_causal = calc_tke_error(base_mean + fluc_pred_causal, y_test)

results.append({
    "transformation": "9. Causal Linear History Model (PCA-32 history -> POD-16 future)",
    "degrees_of_freedom": "32x16 = 512 linear weights (trained on Train, held-out test)",
    "recovered_fluc_error_pct": rec_causal,
    "remaining_fluc_mse": mse_causal,
    "tke_rel_l2_error": tke_causal,
    "notes": "Truly causal prediction from 20 input frames only"
})

# Save results
df_results = pd.DataFrame(results)
df_results.to_csv(OUT / "oracle_decomposition_summary.csv", index=False)

summary_json = {
    "baseline_metrics": {
        "mse_total": mse_total,
        "mse_mean": mse_mean,
        "mse_fluc": mse_fluc,
        "pct_mean": pct_mean,
        "pct_fluc": pct_fluc,
        "baseline_tke_error": baseline_tke_error,
        "energy_ratio_cno_vs_target": energy_ratio
    },
    "pod_energy_distribution": {
        "mean_top1_pct": float(np.mean(pod_energies_top1) * 100),
        "mean_top3_pct": float(np.mean(pod_energies_top3) * 100),
        "mean_top5_pct": float(np.mean(pod_energies_top5) * 100),
        "cno_r2_with_dominant_mode_pct": float(np.mean(cno_proj_top1_r2) * 100)
    },
    "oracle_transformations": results
}

with open(OUT / "oracle_decomposition_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary_json, f, indent=2)

print("\n=== COMPLETE ORACLE DECOMPOSITION TABLE ===")
for r in results:
    print(f"| {r['transformation']} | {r['degrees_of_freedom']} | Recov MSE: {r['recovered_fluc_error_pct']:.2f}% | TKE Err: {r['tke_rel_l2_error']:.6f} |")
