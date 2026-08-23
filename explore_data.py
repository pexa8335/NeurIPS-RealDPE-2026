import h5py
import numpy as np
import matplotlib.pyplot as plt

DATA = r"D:\Project\NeurIPS\data"
OUT = r"D:\Project\NeurIPS\explore"

def load(name):
    f = h5py.File(rf"{DATA}\{name}.h5", "r")
    d = {k: f[k][()] for k in f.keys()}
    f.close()
    return d

sim = load("sim_5025_5")
real = load("real_5025_5")

x, y = sim["x"], sim["y"]

# ---- Fig 1: instantaneous u-field snapshots, sim vs real ----
fig, axes = plt.subplots(2, 3, figsize=(16, 7))
for row, (d, label) in enumerate([(sim["u"], "SIM"), (real["u"].astype(np.float32), "REAL")]):
    for col, t in enumerate([100, 300, 500]):
        ax = axes[row, col]
        im = ax.pcolormesh(x, y, d[t], cmap="RdBu_r", shading="auto")
        plt.colorbar(im, ax=ax)
        ax.set_title(f"{label} u  frame t={t}")
        ax.set_aspect("equal")
plt.tight_layout()
plt.savefig(rf"{OUT}\fig1_snapshots.png", dpi=90)
plt.close()

# ---- Fig 2: vorticity (curl of velocity), sim vs real ----
def vort(u, v):
    du_dy = np.gradient(u, axis=1)
    dv_dx = np.gradient(v, axis=2)
    return dv_dx - du_dy

fig, axes = plt.subplots(1, 3, figsize=(17, 4))
w = vort(sim["u"], sim["v"])
im = axes[0].pcolormesh(x, y, w[400], cmap="RdBu_r", shading="auto", vmin=-15, vmax=15)
plt.colorbar(im, ax=axes[0]); axes[0].set_title("SIM vorticity t=400"); axes[0].set_aspect("equal")
wr = vort(real["u"].astype(np.float32), real["v"].astype(np.float32))
im = axes[1].pcolormesh(x, y, wr[400 % 868], cmap="RdBu_r", shading="auto", vmin=-15, vmax=15)
plt.colorbar(im, ax=axes[1]); axes[1].set_title("REAL vorticity t=400"); axes[1].set_aspect("equal")
diff = wr[400] - w[400]
im = axes[2].pcolormesh(x, y, diff, cmap="RdBu_r", shading="auto", vmin=-15, vmax=15)
plt.colorbar(im, ax=axes[2]); axes[2].set_title("REAL minus SIM"); axes[2].set_aspect("equal")
for ax in axes:
    ax.set_xlabel("x"); ax.set_ylabel("y")
plt.tight_layout()
plt.savefig(rf"{OUT}\fig2_vorticity.png", dpi=90)
plt.close()

# ---- Fig 3: time stats — mean field & fluctuation energy ----
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for col, (d, tag) in enumerate([((sim["u"], sim["v"]), "SIM"), ((real["u"].astype(np.float32), real["v"].astype(np.float32)), "REAL")]):
    um, vm = d
    umean = um.mean(axis=0); vmean = vm.mean(axis=0)
    speed_mean = np.sqrt(umean**2 + vmean**2)
    im = axes[0, col].pcolormesh(x, y, speed_mean, cmap="viridis", shading="auto")
    plt.colorbar(im, ax=axes[0, col]); axes[0, col].set_title(f"{tag} mean speed")
    axes[0, col].set_aspect("equal")
    fluct = 0.5 * (((um - umean)**2).mean(axis=0) + ((vm - vmean)**2).mean(axis=0))
    im = axes[1, col].pcolormesh(x, y, np.log10(fluct + 1e-6), cmap="inferno", shading="auto")
    plt.colorbar(im, ax=axes[1, col]); axes[1, col].set_title(f"{tag} log10 TKE-like")
    axes[1, col].set_aspect("equal")
axes[0, 2].axis("off")
stats_text = "\n".join([
    "STATS",
    f"sim  u: mean={sim['u'].mean():.3f} std={sim['u'].std():.3f}",
    f"real u: mean={real['u'].mean():.3f} std={real['u'].std():.3f}",
    f"sim  v: mean={sim['v'].mean():.3f} std={sim['v'].std():.3f}",
    f"real v: mean={real['v'].mean():.3f} std={real['v'].std():.3f}",
    f"sim  p: mean={sim['p'].mean():.3f} std={sim['p'].std():.3f}",
])
axes[0, 2].text(0.05, 0.6, stats_text, fontsize=13, family="monospace", verticalalignment="top")
plt.tight_layout()
plt.savefig(rf"{OUT}\fig3_stats.png", dpi=90)
plt.close()

# ---- Fig 4: time series at wake probes ----
probe_x = np.argmin(np.abs(x[32] - x.max() * 0.75))
probe_y = np.argmin(np.abs(y[:, 64]))
ts_sim_u = sim["u"][:, 32, probe_x]
ts_real_u = real["u"][:, 32, probe_x].astype(np.float32)
fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=False)
axes[0].plot(sim["t"], ts_sim_u, lw=0.6); axes[0].set_title("SIM: u(t) at mid-height probe")
axes[1].plot(real["t"], ts_real_u, lw=0.6, color="tab:red"); axes[1].set_title("REAL: u(t) at same probe")
axes[1].set_xlabel("t")
plt.tight_layout()
plt.savefig(rf"{OUT}\fig4_timeseries.png", dpi=90)
plt.close()

print("grid x range:", x.min(), x.max(), "| y range:", y.min(), y.max())
print("dt sim:", np.diff(sim["t"])[:3], " dt real:", np.diff(real["t"])[:3])
print("done -> figs saved in explore/")
