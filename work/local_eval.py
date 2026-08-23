import sys, os, time
import h5py
import numpy as np

sys.path.insert(0, r"D:\Project\NeurIPS\work\test_sub")
import submission

DATA = r"D:\Project\NeurIPS\data"
T_IN = T_OUT = 20
SIGMA_GLOBAL = 0.0563870

def load_windows(paths):
    X, Y = [], []
    for p in paths:
        with h5py.File(p, "r") as f:
            u = f["u"][()].astype(np.float32)
            v = f["v"][()].astype(np.float32)
        u = u[:, ::2, ::2]; v = v[:, ::2, ::2]
        T = min(len(u), len(v))
        for i in range((T - T_IN - T_OUT) // 20):
            a, b = i * 20, i * 20 + T_IN
            xin = np.stack([u[a:b], v[a:b]], axis=-1)
            ytg = np.stack([u[b:b+T_OUT], v[b:b+T_OUT]], axis=-1)
            X.append(xin); Y.append(ytg)
    X = np.pad(np.stack(X).astype(np.float32), ((0,0),(0,0),(0,0),(0,0),(0,1)))
    Y = np.pad(np.stack(Y).astype(np.float32), ((0,0),(0,0),(0,0),(0,0),(0,1)))
    return X, Y

def rel_l2_err(P, G, ch=(0, 1)):
    d = P[..., ch] - G[..., ch]
    return np.sqrt((d**2).sum()) / (np.sqrt((G[..., ch]**2).sum()) + 1e-12)

def tke_field(a):
    m = a.mean(axis=1, keepdims=True)
    return 0.5 * (((a[..., 0]-m[..., 0])**2).mean(axis=1) + ((a[..., 1]-m[..., 1])**2).mean(axis=1))

def mvpe_err(P, G):
    # proxy: time-averaged velocity profiles at wake probe columns (x > 0.4 of domain width)
    xs = int(G.shape[3] * 0.45)
    pp = P[..., :2, :, xs:, :].mean(axis=1)
    gg = G[..., :2, :, xs:, :].mean(axis=1)
    return np.sqrt(((pp-gg)**2).sum()) / (np.sqrt((gg**2).sum()) + 1e-12)

def map_score(err):
    return 100.0 / (1.0 + 0.5 * err)

def sps_official(P, G, lower, upper, tke_e=None, mvpe_e=None):
    e_dm = rel_l2_err(P, G)
    e_tke = tke_e if tke_e is not None else (
        np.sqrt(((tke_field(P[..., :2])-tke_field(G[..., :2]))**2).mean())
        / (np.sqrt((tke_field(G[..., :2])**2).mean()) + 1e-12))
    e_mv = mvpe_e if mvpe_e is not None else mvpe_err(P, G)
    nil = (upper - lower) / SIGMA_GLOBAL
    expn = np.exp(-nil)
    out = {}
    for name, e, w in [("dm", e_dm, 0.5), ("tke", e_tke, 0.3), ("mvpe", mvpe_e if mvpe_e is not None else e_mv, 0.2)]:
        pm = e / (0.5 + e)
        inside = ((G[..., :2] >= lower[..., :2]) & (G[..., :2] <= upper[..., :2]))
        vals = (1 - pm) * expn[..., :2] * inside
        out[name] = float(vals.mean())
    return 100 * (0.5*out["dm"] + 0.3*out["tke"] + 0.2*out["mvpe"]), out

def main():
    X, G = load_windows([os.path.join(DATA, "holdout_11400_15.h5"), os.path.join(DATA, "holdout_17775_5.h5")])
    print(f"holdout windows: {len(X)}")

    # timed inference (exclude first call = warmup/load effects)
    t0 = time.time()
    preds = []
    for s in range(0, len(X), 8):
        o = submission.predict(X[s:s+8])
        preds.append(o)
    wall = time.time() - t0
    pred = np.concatenate([o["prediction"] for o in preds])
    lo15 = np.concatenate([o["lower"] for o in preds])
    hi15 = np.concatenate([o["upper"] for o in preds])
    t_per_sample = wall / len(X)
    r = t_per_sample / 0.72896
    time_score = 100 / (1 + np.sqrt(r))

    e_rel = rel_l2_err(pred, G)
    tkP, tkG = tke_field(pred[..., :2]), tke_field(G[..., :2])
    e_tke = np.sqrt(((tkP-tkG)**2).mean()) / (np.sqrt((tkG**2).mean()) + 1e-12)
    e_mv = mvpe_err(pred, G)

    print("\n=== OFFICIAL-STYLE SUBSCORES (local holdout, 82 windows) ===")
    print(f"rel_l2  err={e_rel:.4f}  -> rel_l2_score = {map_score(e_rel):.1f}")
    print(f"tke     err={e_tke:.4f}  -> tke_score    = {map_score(e_tke):.1f}")
    print(f"mvpe*   err={e_mv:.4f}  -> mvpe_score   = {map_score(e_mv):.1f}   (*proxy definition)")
    print(f"time    t={t_per_sample:.3f}s/sample r={r:.3f} -> time_score = {time_score:.1f}")

    # SPS: our calibrated band vs default +-5%
    sig = np.array(submission.RESID_SIGMA_UV, dtype=np.float32)
    lo_d = pred - 0.05*np.abs(pred); hi_d = pred + 0.05*np.abs(pred)
    lo_d[..., 2] = -10.; hi_d[..., 2] = 10.
    lo15[..., 2] = -10.; hi15[..., 2] = 10.
    sps_def, br_d = sps_official(pred, G, lo_d, hi_d, e_tke, e_mv)
    sps_cal, br_c = sps_official(pred, G, lo15.copy(), hi15.copy(), e_tke, e_mv)
    cov_d = ((G[..., :2] >= lo_d[..., :2]) & (G[..., :2] <= hi_d[..., :2])).mean()
    cov_c = ((G[..., :2] >= lo15[..., :2]) & (G[..., :2] <= hi15[..., :2])).mean()
    print("\n--- SPS ---")
    print(f"default band (+-5%):      coverage={cov_d:.3f} branches={ {k: round(v,3) for k,v in br_d.items()} } sps={sps_def:.1f}")
    print(f"calibrated band (k=1.5):  coverage={cov_c:.3f} branches={ {k: round(v,3) for k,v in br_c.items()} } sps={sps_cal:.1f}")

    print("\n--- reference: baseline BTC numbers from kernel log ---")
    e_rel_b, e_tke_b = 0.1404, 0.6652
    print(f"baseline rel_l2_score={map_score(e_rel_b):.1f}  tke_score={map_score(e_tke_b):.1f}")

if __name__ == "__main__":
    main()
