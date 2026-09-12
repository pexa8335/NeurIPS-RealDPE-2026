# Hypothesis 02: Exhaustive Audit of Parameter Space, Metadata Integrity, Temporal Structure, and Incumbent Performance

**Project**: NeurIPS 2026 RealPDE Track 1 (Sim2Real Airfoil Dynamics)  
**Deliverable**: `hypothesis_02_reynolds_aoa_distribution_v2.md`  
**Date**: 2026-09-11  
**Audit Target**: Complete archive of 182 HDF5 files (`train_real`: 82, `train_sim`: 100) and Incumbent Model `sub7_cno_zeromean_head_adaptive_sps`  
**Prior Notebook Audited**: `hypothesis/hypo_02_reynolds_aoa_distribution.ipynb` (SHA-256: `918664b22c7a364be14b8a6a2dae3ecfc9517173e44ebfa1d1ffaa0a20a4be21`)  
**Artifact Directory**: [`research/hypothesis02_audit/`](file:///d:/Project/NeurIPS/research/hypothesis02_audit/)

---

## 1. Executive Conclusion

A prior project notebook (`hypo_02_reynolds_aoa_distribution.ipynb`) declared Hypothesis 2 as globally "ACCEPTED", asserting that all 182 trajectories share an identical length $T=607$ with $\Delta t = 0.05\text{ s}$, that filename labels represent exact physical ground truth, and that 10 absent conditions ($Re=15225, 27975$) are "almost certainly" the hidden test set.

This exhaustive re-audit inspected **100% of the 182 HDF5 files** directly from [`archive.zip`](file:///d:/Project/NeurIPS/archive.zip) (reading all metadata scalars, full coordinate arrays, and temporal vector fields), cross-referenced the current incumbent benchmark (`sub7`), and evaluated the decision gate for held-Reynolds-out training. 

### Key Findings:
1. **Filename $\neq$ Physical Ground Truth in Real Data** (*Directly Observed*):
   - In `train_sim`, nominal filenames match internal HDF5 scalar metadata exactly (100/100 files, $0.0\%$ mismatch).
   - In `train_real`, filename Reynolds numbers **fail to match internal HDF5 metadata in 78 out of 82 files (95.12% mismatch)**.
   - Only the $Re=3750$ operating condition matches ($4/4$ files). All other 17 operating groups exhibit a systematic, monotonically increasing positive offset: $\Delta Re = \text{internal\_Re} - \text{nominal\_Re}$ ranges from $+3$ at $Re=5025$ to $+61$ at $Re=26700$ ($\approx +0.26\%$ affine shift).
   - Crucially, internal Reynolds numbers are **not noisy or jittered across trajectories**: each nominal operating condition maps deterministically to exactly one integer internal Reynolds number across all angles of attack.
   - Angle of attack (`AoA`) matches internal metadata in **100% of all 182 files** (0 mismatches across both Real and Sim).

2. **Temporal Regularity Claims Were Methodologically False** (*Directly Observed*):
   - The prior claim that all files share $T=607$ and $\Delta t = 0.05\text{ s}$ is **completely false**.
   - The prior notebook inspected exactly two Real files and two Sim files, and happened to pick `10125_0.h5`—which is **the only file in the entire dataset with $T=607$**.
   - In reality:
     - `train_sim`: All 100 files have $T=1000$ frames ($t \in [0.08, 20.08]\text{ s}$, duration $20.00\text{ s}$, uniform $\Delta t = 0.020020\text{ s}$).
     - `train_real`: 79 files ($96.3\%$) have $T=868$ frames ($t \in [0.08, 17.42]\text{ s}$, duration $17.34\text{ s}$, $\Delta t = 0.020000\text{ s} = 50\text{ Hz}$).
     - Exactly 3 isolated truncated trajectories exist in Real data: `10125_0.h5` ($T=607$), `21600_5.h5` ($T=492$), and `24150_20.h5` ($T=282$).
     - The true sampling rate is **50 Hz ($\Delta t = 0.02\text{ s}$)**, not $20\text{ Hz}$ ($\Delta t = 0.05\text{ s}$).

3. **Sim vs Real Parameter Space Coverage** (*Directly Observed*):
   - `train_sim` spans a complete orthogonal $20 \times 5 = 100$ lattice ($Re \in [3750, 27975], \Delta Re = 1275$; $AoA \in \{0^\circ, 5^\circ, 10^\circ, 15^\circ, 20^\circ\}$).
   - `train_real` covers 82 nominal conditions (82% of the lattice). Exactly 18 nominal conditions are absent from Real data:
     - Two complete Reynolds groups: $Re=15225$ (5 files) and $Re=27975$ (5 files).
     - Eight scattered configurations: $(3750, 15^\circ), (17775, 0^\circ), (22875, 5^\circ), (22875, 20^\circ), (24150, 5^\circ), (25425, 5^\circ), (26700, 5^\circ), (26700, 20^\circ)$.
     - At high Reynolds numbers ($Re \ge 22875$), $AoA = 5^\circ$ is completely absent across four consecutive Reynolds intervals.
   - Calling missing conditions the "hidden test set" is unevidenced speculation. They are formally designated: **"Sim-supported conditions absent from the observed Real training set."**

4. **Incumbent Performance Stratification** (*Directly Observed on Audited Test Set*):
   - Point predictor error ($\text{Rel-L2}$) does **not** degrade monotonically with Reynolds number: $Re=13977$ achieves the lowest error ($0.0744$), outperforming both low-$Re$ ($6306$: $0.0947$) and high-$Re$ ($24204$: $0.0914$).
   - Point predictor error degrades **strictly monotonically with Angle of Attack**: Rel-L2 increases by $+118\%$ from $0.0535$ at $AoA=0^\circ$ to $0.1165$ at $AoA=20^\circ$.
   - Uncertainty calibration ($\text{SPS}$) suffers **severe degradation at high Reynolds number and high AoA**:
     - SPS score drops by $-11.22$ points from $Re=6306$ ($51.97$, $93.68\%$ coverage) to $Re=24204$ ($40.75$, $84.29\%$ coverage, with $u$-channel coverage dropping to $79.90\%$).
     - At the worst joint condition ($(24150, 20^\circ)$), SPS collapses to $33.07$ with $u$-coverage at $74.85\%$.

5. **Decision Gate Verdict: CONDITIONAL HOLD / GATE CLOSED FOR ARCHITECTURE RETRAINING**:
   - Re-training the model backbone under a "Leave-Reynolds-Out" protocol is **NOT warranted** because point predictor performance generalizes stably across Reynolds numbers without catastrophic failure.
   - However, UQ miscalibration is materially concentrated in the high-Reynolds, high-AoA regime with sparse Real coverage.
   - We specify the exact protocol for a controlled held-Reynolds-out experiment (Section 9), but enforce: **DO NOT RUN** unless explicitly authorized by the project leads.

---

## 2. What Was Actually Tested

| Audit Target | Scope | Verification Method | Pass Criteria |
| :--- | :--- | :--- | :--- |
| **All 182 HDF5 Files** | 82 Real + 100 Sim | Direct extraction from `archive.zip` via `h5py` in memory | Zero uninspected files; complete schema inspection |
| **Metadata Consistency** | `re`, `aoa` datasets vs filenames | Exact equality comparison (`nom_re == int_re`, `nom_aoa == int_aoa`) | Quantified diff distribution; no silent rounding |
| **Field Geometry & Arrays** | Shapes of $u, v, x, y, t$ | Structural inspection of numpy array shapes and dtypes | $u, v \in (T, 64, 128)$, $x, y \in (64, 128)$ |
| **Numerical Integrity** | Finite values & monotonicity | `np.isfinite` and `np.diff(t) > 0` across all files | Zero NaNs/Infs; strictly monotonic time |
| **Sampling Structure** | Timestep intervals $\Delta t$ | `median`, `min`, `max`, and `max_abs_dev` of `np.diff(t)` | Exact per-file statistics; irregularity quantification |
| **Incumbent Performance** | `sub7` deployed predictions | Matched evaluation on 80 development test windows | Stratification by nominal Re, actual Re, AoA, coverage |

---

## 3. Dataset Metadata Audit (Exhaustive)

All 182 files in `archive.zip` were read in full. Summary statistics by domain are presented below:

### Table 1: Domain-Level Summary
*(Source artifact: [`research/hypothesis02_audit/domain_summary.csv`](file:///d:/Project/NeurIPS/research/hypothesis02_audit/domain_summary.csv))*

| Metric / Property | Numerical Simulation (`train_sim`) | Experimental PIV (`train_real`) | Combined / Status |
| :--- | :---: | :---: | :---: |
| **Total Files Audited** | **100** | **82** | **182 files (100% complete)** |
| **Top-Level HDF5 Keys** | `aoa, p, re, t, u, v, x, y` (8 keys) | `aoa, re, t, u, v, x, y` (7 keys) | Pressure $p$ present in Sim only |
| **HDF5 Attributes (`attrs`)** | None (`[]`) | None (`[]`) | Scalar metadata stored as datasets |
| **Spatial Field Shape ($u, v$)** | `(T, 64, 128)` | `(T, 64, 128)` | **100% PASS** (182/182 files) |
| **Coordinate Shape ($x, y$)** | `(64, 128)` | `(64, 128)` | **100% PASS** (182/182 files) |
| **Finite Values Check** | All values finite (**True**) | All values finite (**True**) | **100% PASS** (Zero NaNs, zero Infs) |
| **Time Monotonicity Check** | Strictly monotonic (**True**) | Strictly monotonic (**True**) | **100% PASS** ($\Delta t > 0$ everywhere) |
| **Length Consistency ($T_u=T_v=T_t$)** | Strictly consistent (**True**) | Strictly consistent (**True**) | **100% PASS** (No truncated channels) |
| **Unique Nominal Re Groups** | 20 | 18 | 2 groups missing in Real ($15225, 27975$) |
| **Unique Actual Re Groups** | 20 | 18 | Offset in Real ($3750 \to 26761$) |
| **Unique Nominal AoA Values** | 5 ($0^\circ, 5^\circ, 10^\circ, 15^\circ, 20^\circ$) | 5 ($0^\circ, 5^\circ, 10^\circ, 15^\circ, 20^\circ$) | Orthogonal grid along AoA axis |
| **Trajectory Length Range ($T$)** | **[1000, 1000]** | **[282, 868]** | **Prior claim of universal $T=607$ is FALSE** |
| **Median Sampling Interval ($\Delta t$)** | **$0.020020\text{ s}$** ($20.0/999$) | **$0.020000\text{ s}$** ($50\text{ Hz}$) | **Prior claim of $\Delta t = 0.05\text{ s}$ is FALSE** |
| **Max Within-File $\Delta t$ Deviation** | $1.78 \times 10^{-15}\text{ s}$ | $1.43 \times 10^{-6}\text{ s}$ | High temporal sampling uniformity |

---

## 4. Nominal vs Actual Reynolds Numbers

### Table 2: Exact Reynolds Discrepancy by Nominal Operating Group
*(Source artifact: [`research/hypothesis02_audit/nominal_vs_actual_re.csv`](file:///d:/Project/NeurIPS/research/hypothesis02_audit/nominal_vs_actual_re.csv))*

| Nominal Re | Real Files | Internal Re (Real) | Discrepancy $\Delta Re$ | Sim Files | Internal Re (Sim) | Discrepancy $\Delta Re$ | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **3750** | 4 | 3750 | **0** | 5 | 3750 | 0 | Exact match |
| **5025** | 5 | 5028 | **+3** | 5 | 5025 | 0 | Systematic positive shift |
| **6300** | 5 | 6306 | **+6** | 5 | 6300 | 0 | Systematic positive shift |
| **7575** | 5 | 7585 | **+10** | 5 | 7575 | 0 | Systematic positive shift |
| **8850** | 5 | 8863 | **+13** | 5 | 8850 | 0 | Systematic positive shift |
| **10125** | 5 | 10142 | **+17** | 5 | 10125 | 0 | Systematic positive shift |
| **11400** | 5 | 11420 | **+20** | 5 | 11400 | 0 | Systematic positive shift |
| **12675** | 5 | 12698 | **+23** | 5 | 12675 | 0 | Systematic positive shift |
| **13950** | 5 | 13977 | **+27** | 5 | 13950 | 0 | Systematic positive shift |
| **15225** | 0 | — | — | 5 | 15225 | 0 | **Sim only** (Absent from Real) |
| **16500** | 5 | 16534 | **+34** | 5 | 16500 | 0 | Systematic positive shift |
| **17775** | 4 | 17812 | **+37** | 5 | 17775 | 0 | Systematic positive shift |
| **19050** | 5 | 19090 | **+40** | 5 | 19050 | 0 | Systematic positive shift |
| **20325** | 5 | 20369 | **+44** | 5 | 20325 | 0 | Systematic positive shift |
| **21600** | 5 | 21647 | **+47** | 5 | 21600 | 0 | Systematic positive shift |
| **22875** | 3 | 22926 | **+51** | 5 | 22875 | 0 | Systematic positive shift |
| **24150** | 4 | 24204 | **+54** | 5 | 24150 | 0 | Systematic positive shift |
| **25425** | 4 | 25482 | **+57** | 5 | 25425 | 0 | Systematic positive shift |
| **26700** | 3 | 26761 | **+61** | 5 | 26700 | 0 | Systematic positive shift |
| **27975** | 0 | — | — | 5 | 27975 | 0 | **Sim only** (Absent from Real) |

![Nominal vs Internal Reynolds Number Discrepancy](reynolds_nominal_vs_actual.png)

### Key Observations on Metadata Integrity:
1. **Fraction of Mismatches**:
   - `train_real`: **78/82 files (95.12%)** have `nominal_Re != internal_Re`.
   - `train_sim`: **0/100 files (0.00%)** have mismatches.
2. **Deterministic Scaling Law**:
   - Discrepancy is strictly positive and monotonically increasing with Reynolds number ($0 \le \Delta Re \le 61$).
   - A linear regression yields:
     $$\Delta Re \approx 0.002636 \times \text{nominal\_Re} - 9.94 \quad (R^2 = 0.9994)$$
   - This corresponds to an approximate $+0.26\%$ systematic calibration offset in the experimental velocity measurement facility.
3. **Absence of Intra-Condition Jitter**:
   - Within each nominal operating point (e.g. nominal $Re=10125$), **all trajectories regardless of AoA have the exact identical internal Reynolds number** ($10142$).
   - Therefore, while filenames do not equal internal physical parameters, **each nominal label is a 1-to-1 surrogate for a unique physical operating point**.
   - **Recommendation**: Models conditioning on Reynolds number MUST use internal HDF5 scalar metadata, never parsed filename strings.

---

## 5. Sim vs Real Parameter-Space Coverage

### Table 3: Nominal $20 \times 5$ Condition Matrix
*(Source artifact: [`research/hypothesis02_audit/condition_coverage.csv`](file:///d:/Project/NeurIPS/research/hypothesis02_audit/condition_coverage.csv))*

| Nominal Re | AoA 0° | AoA 5° | AoA 10° | AoA 15° | AoA 20° | Real Completeness |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **3750** | Real + Sim | Real + Sim | Real + Sim | <span style="color:red">Sim Only</span> | Real + Sim | 4 / 5 |
| **5025** | Real + Sim | Real + Sim | Real + Sim | Real + Sim | Real + Sim | **5 / 5** |
| **6300** | Real + Sim | Real + Sim | Real + Sim | Real + Sim | Real + Sim | **5 / 5** |
| **7575** | Real + Sim | Real + Sim | Real + Sim | Real + Sim | Real + Sim | **5 / 5** |
| **8850** | Real + Sim | Real + Sim | Real + Sim | Real + Sim | Real + Sim | **5 / 5** |
| **10125** | Real + Sim ($T=607$) | Real + Sim | Real + Sim | Real + Sim | Real + Sim | **5 / 5** |
| **11400** | Real + Sim | Real + Sim | Real + Sim | Real + Sim | Real + Sim | **5 / 5** |
| **12675** | Real + Sim | Real + Sim | Real + Sim | Real + Sim | Real + Sim | **5 / 5** |
| **13950** | Real + Sim | Real + Sim | Real + Sim | Real + Sim | Real + Sim | **5 / 5** |
| **15225** | <span style="color:red">Sim Only</span> | <span style="color:red">Sim Only</span> | <span style="color:red">Sim Only</span> | <span style="color:red">Sim Only</span> | <span style="color:red">Sim Only</span> | **0 / 5 (Full Re Absent)** |
| **16500** | Real + Sim | Real + Sim | Real + Sim | Real + Sim | Real + Sim | **5 / 5** |
| **17775** | <span style="color:red">Sim Only</span> | Real + Sim | Real + Sim | Real + Sim | Real + Sim | 4 / 5 |
| **19050** | Real + Sim | Real + Sim | Real + Sim | Real + Sim | Real + Sim | **5 / 5** |
| **20325** | Real + Sim | Real + Sim | Real + Sim | Real + Sim | Real + Sim | **5 / 5** |
| **21600** | Real + Sim | Real + Sim ($T=492$) | Real + Sim | Real + Sim | Real + Sim | **5 / 5** |
| **22875** | Real + Sim | <span style="color:red">Sim Only</span> | Real + Sim | Real + Sim | <span style="color:red">Sim Only</span> | 3 / 5 |
| **24150** | Real + Sim | <span style="color:red">Sim Only</span> | Real + Sim | Real + Sim | Real + Sim ($T=282$) | 4 / 5 |
| **25425** | Real + Sim | <span style="color:red">Sim Only</span> | Real + Sim | Real + Sim | Real + Sim | 4 / 5 |
| **26700** | Real + Sim | <span style="color:red">Sim Only</span> | Real + Sim | Real + Sim | <span style="color:red">Sim Only</span> | 3 / 5 |
| **27975** | <span style="color:red">Sim Only</span> | <span style="color:red">Sim Only</span> | <span style="color:red">Sim Only</span> | <span style="color:red">Sim Only</span> | <span style="color:red">Sim Only</span> | **0 / 5 (Full Re Absent)** |

![Parameter Space Coverage](parameter_space_coverage.png)

### Summary of Coverage:
- Total Sim conditions: **100** ($20 \text{ Re} \times 5 \text{ AoA}$).
- Total Real conditions: **82** ($18 \text{ Re}$ groups, with varying AoA completeness).
- Sim-supported conditions absent from observed Real training set: **exactly 18 conditions**.
  - **10 conditions** from 2 fully absent Reynolds bands: $Re=15225$ and $Re=27975$.
  - **8 conditions** from scattered configurations: $(3750, 15^\circ)$, $(17775, 0^\circ)$, $(22875, 5^\circ)$, $(22875, 20^\circ)$, $(24150, 5^\circ)$, $(25425, 5^\circ)$, $(26700, 5^\circ)$, $(26700, 20^\circ)$.
- **Critical Structural Pattern**:
  At higher Reynolds numbers ($Re \ge 22875$), $AoA = 5^\circ$ is missing across **four consecutive operating groups** ($22875, 24150, 25425, 26700$). This indicates an experimental limitation or missing test cell in the physical acquisition campaign.

---

## 6. Temporal Structure Audit

### Table 4: Trajectory Lengths and Sampling Spacing
*(Source artifact: [`research/hypothesis02_audit/per_file_metadata.csv`](file:///d:/Project/NeurIPS/research/hypothesis02_audit/per_file_metadata.csv))*

| Property | `train_sim` (100 files) | `train_real` (82 files) | Prior Hypothesis 2 Notebook Claim | Verdict |
| :--- | :---: | :---: | :---: | :---: |
| **Trajectory Lengths ($T$)** | Exactly $1000$ (100 files) | 79 files: $T=868$<br>1 file: $T=607$ (`10125_0.h5`)<br>1 file: $T=492$ (`21600_5.h5`)<br>1 file: $T=282$ (`24150_20.h5`) | "Strictly uniform $T=607$ across all 182 files" | **FALSE** |
| **Time Span ($t_{\text{start}} \to t_{\text{end}}$)** | $0.08\text{ s} \to 20.08\text{ s}$ | $0.08\text{ s} \to 17.42\text{ s}$ ($T=868$)<br>$0.08\text{ s} \to 12.20\text{ s}$ ($T=607$)<br>$0.08\text{ s} \to 9.90\text{ s}$ ($T=492$)<br>$0.08\text{ s} \to 5.70\text{ s}$ ($T=282$) | "$t \in [0.0, 30.3]\text{ s}$" | **FALSE** |
| **Median $\Delta t$** | $0.020020\text{ s}$ | $0.020000\text{ s}$ | "Constant $\Delta t = 0.05\text{ s}$" | **FALSE** |
| **Equivalent Frequency** | $49.95\text{ Hz}$ | $50.00\text{ Hz}$ | $20.00\text{ Hz}$ | **FALSE** |
| **Strict Monotonicity** | 100% PASS | 100% PASS | "Strictly monotonic" | **SUPPORTED** |
| **Channel Consistency** | 100% PASS ($u, v, t$ match) | 100% PASS ($u, v, t$ match) | "No missing temporal records" | **SUPPORTED** |

![Temporal Distribution](temporal_distribution.png)

### Methodological Autopsy of the Prior Error:
In `hypo_02_reynolds_aoa_distribution.ipynb` (Cell 1, lines 113–126), the code ran:
```python
for fpath in [real_files[0], real_files[-1], sim_files[0], sim_files[-1]]:
    ...
```
`real_files[0]` happened to be `train_real/train_real/10125_0.h5`. That single file has $T=607$. The notebook printed:
- `10125_0.h5 Real T_steps=607 mean_dt=0.02000`
- `8850_5.h5 Real T_steps=868 mean_dt=0.02000`
- `10125_0.h5 Sim T_steps=1000 mean_dt=0.02002`

Despite its own print output clearly showing $T=868$ and $T=1000$ and $\Delta t = 0.02\text{ s}$, the narrative summary hallucinated:
> *"Across all 182 files, temporal trajectories are strictly uniform: Exactly T=607 frames per trajectory. Constant time step dt=0.05 s (t in [0.0, 30.3] s)."*

This confirms that the prior notebook was an unreliable diagnostic report.

---

## 7. Incumbent Performance Stratification

Using the exact audited predictions from `sub7_cno_zeromean_head_adaptive_sps` on the 80 development test windows (14 Real trajectories across 3 Reynolds operating groups and all 5 AoAs), we stratified all official competition metrics:

### Table 5: Incumbent Metrics Stratified by Reynolds Group
*(Source artifact: [`research/hypothesis02_audit/incumbent_stratification_by_re.csv`](file:///d:/Project/NeurIPS/research/hypothesis02_audit/incumbent_stratification_by_re.csv))*

| Nominal Re | Actual Re | Trajectories | Windows | Rel-L2 (Field) | MVPE | TKE Error | Aggregate Coverage | Coverage $u$ | Coverage $v$ | Mean Band Width | SPS Score |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **6300** | **6306** | 5 | 30 | $0.0947 \pm 0.026$ | $0.0979 \pm 0.040$ | $0.5153 \pm 0.141$ | $93.68\%$ | $92.41\%$ | $96.16\%$ | $0.0165$ | **51.97** |
| **13950** | **13977** | 5 | 30 | **$0.0744 \pm 0.023$** | **$0.0713 \pm 0.033$** | $0.6000 \pm 0.114$ | $90.84\%$ | $88.82\%$ | $94.63\%$ | $0.0206$ | **48.44** |
| **24150** | **24204** | 4 | 20 | $0.0914 \pm 0.031$ | $0.0742 \pm 0.032$ | **$0.6495 \pm 0.151$** | **$84.29\%$** | **$79.90\%$** | $91.73\%$ | $0.0269$ | **40.75** |

### Table 6: Incumbent Metrics Stratified by Angle of Attack (AoA)
*(Source artifact: [`research/hypothesis02_audit/incumbent_stratification_by_aoa.csv`](file:///d:/Project/NeurIPS/research/hypothesis02_audit/incumbent_stratification_by_aoa.csv))*

| AoA | Trajectories | Windows | Rel-L2 (Field) | MVPE | TKE Error | Aggregate Coverage | Coverage $u$ | Coverage $v$ | Mean Band Width | SPS Score |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0°** | 3 | 18 | **$0.0535 \pm 0.010$** | **$0.0428 \pm 0.011$** | $0.5885 \pm 0.238$ | $93.69\%$ | $91.85\%$ | $96.90\%$ | $0.0166$ | **55.09** |
| **5°** | 2 | 12 | $0.0622 \pm 0.010$ | $0.0460 \pm 0.013$ | $0.5556 \pm 0.123$ | $95.03\%$ | $93.68\%$ | $97.57\%$ | $0.0162$ | **55.29** |
| **10°** | 3 | 18 | $0.0925 \pm 0.011$ | $0.0967 \pm 0.019$ | $0.5642 \pm 0.113$ | $89.65\%$ | $86.84\%$ | $94.41\%$ | $0.0224$ | **45.84** |
| **15°** | 3 | 18 | $0.1052 \pm 0.017$ | $0.0982 \pm 0.033$ | $0.6291 \pm 0.076$ | $87.44\%$ | $84.69\%$ | $92.67\%$ | $0.0234$ | **42.82** |
| **20°** | 3 | 14 | **$0.1165 \pm 0.015$** | **$0.1235 \pm 0.021$** | $0.5506 \pm 0.093$ | **$86.54\%$** | **$83.56\%$** | $91.13\%$ | $0.0239$ | **41.74** |

![Incumbent Performance Stratification](incumbent_performance_stratification.png)

### Table 7: Joint $(Re, AoA)$ Cell Heatmap
*(Source artifact: [`research/hypothesis02_audit/incumbent_stratification_by_re_aoa.csv`](file:///d:/Project/NeurIPS/research/hypothesis02_audit/incumbent_stratification_by_re_aoa.csv))*

| Actual Re | Nominal AoA 0° | Nominal AoA 5° | Nominal AoA 10° | Nominal AoA 15° | Nominal AoA 20° |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **6306** | Rel-L2: **$0.0668$**<br>SPS: **$59.79$** (Cov: $97.3\%$) | Rel-L2: **$0.0713$**<br>SPS: **$56.68$** (Cov: $96.4\%$) | Rel-L2: **$0.0911$**<br>SPS: **$51.58$** (Cov: $94.1\%$) | Rel-L2: **$0.1128$**<br>SPS: **$47.37$** (Cov: $92.2\%$) | Rel-L2: **$0.1312$**<br>SPS: **$45.00$** (Cov: $88.7\%$) |
| **13977** | Rel-L2: **$0.0453$**<br>SPS: **$55.57$** (Cov: $93.9\%$) | Rel-L2: **$0.0531$**<br>SPS: **$53.91$** (Cov: $93.7\%$) | Rel-L2: **$0.0835$**<br>SPS: **$46.58$** (Cov: $90.5\%$) | Rel-L2: **$0.0852$**<br>SPS: **$45.34$** (Cov: $90.0\%$) | Rel-L2: **$0.1047$**<br>SPS: **$41.34$** (Cov: $86.4\%$) |
| **24204** | Rel-L2: **$0.0485$**<br>SPS: **$49.87$** (Cov: $89.8\%$) | *[Absent from Real]* | Rel-L2: **$0.1027$**<br>SPS: **$39.52$** (Cov: $84.5\%$) | Rel-L2: **$0.1175$**<br>SPS: **$35.73$** (Cov: $80.1\%$) | Rel-L2: **$0.1076$**<br>SPS: **$33.07$** (Cov: $80.3\%$) |

![Incumbent Grid Heatmaps](incumbent_grid_heatmaps.png)

### Key Performance Findings:
1. **Field Error Does Not Degrade Systematically with Reynolds Number**:
   - The Rel-L2 error across Reynolds numbers follows a **U-shaped curve**: $0.0947$ at $Re=6306$, dropping to $0.0744$ at $Re=13977$, and rising mildly to $0.0914$ at $Re=24204$.
   - The incumbent point predictor generalizes across Reynolds regimes without catastrophic collapse.
2. **Angle of Attack is the Primary Driver of Field Error**:
   - Rel-L2 error exhibits strict monotonic growth with AoA ($0.0535 \to 0.1165$, $+118\%$).
   - MVPE error exhibits strict monotonic growth with AoA ($0.0428 \to 0.1235$, $+188\%$).
   - This aligns with the fluid physics of flow separation and massive vortex shedding at high incidence angles ($\ge 15^\circ$).
3. **SPS Uncertainty Calibration Degrades Severely at High Re and High AoA**:
   - SPS score drops by **$-11.22$ points** from $Re=6306$ ($51.97$) to $Re=24204$ ($40.75$).
   - At high Reynolds numbers, coverage drops from $93.68\%$ to $84.29\%$ (undercovering the $90\%$ target), despite the adaptive SPS bands expanding by $+63\%$ (mean width $0.0165 \to 0.0269$).
   - Undercoverage is concentrated almost entirely in the $u$-channel: at $Re=24204$, $u$-coverage falls to $79.90\%$ (and $74.85\%$ at $AoA=20^\circ$).
4. **Interaction with Real Training Coverage**:
   - The worst performance cell ($(24150, 20^\circ)$, SPS $33.07$) is in a Reynolds group where $AoA=5^\circ$ is completely absent from the training data, and the trajectory itself was shortened to $T=282$ frames (providing only 2 windows for evaluation).

---

## 8. Claim-by-Claim Verdict Table

Every scientific finding is formally classified below in accordance with scientific reporting standards:

| Scientific Claim | Prior H2 Claim | Audit Verdict | Epistemic Status | Supporting Evidence |
| :--- | :---: | :---: | :---: | :--- |
| **1. Sim nominal parameter grid is complete** | CONFIRMED ($20 \times 5 = 100$) | **SUPPORTED** | *Directly observed* | 100 HDF5 files form a complete orthogonal lattice across all 20 Re and 5 AoA. |
| **2. Real nominal support is a strict subset of Sim** | CONFIRMED (82 / 100) | **SUPPORTED** | *Directly observed* | Real covers 82 nominal conditions; all 82 exist in Sim. |
| **3. Filename Re equals actual internal Re** | CONFIRMED (Implicit) | **FALSE** | *Directly observed* | 78/82 Real files (95.12%) have $\Delta Re = +3$ to $+61$. Only $Re=3750$ matches. |
| **4. Filename AoA equals actual internal AoA** | CONFIRMED | **SUPPORTED** | *Directly observed* | 182/182 files (100%) have `nom_aoa == int_aoa`. |
| **5. Trajectories are uniform at $T=607$** | CONFIRMED | **FALSE** | *Directly observed* | Sim is $T=1000$ (100%); Real is $T=868$ (96.3%), with only 1 file having $T=607$. |
| **6. Sampling interval is uniform at $\Delta t = 0.05\text{ s}$** | CONFIRMED | **FALSE** | *Directly observed* | Real is $\Delta t = 0.020000\text{ s}$ (50 Hz); Sim is $\Delta t = 0.020020\text{ s}$. |
| **7. Absent Real conditions = Hidden test set** | CONFIRMED | **SPECULATION** | *Unresolved* | Missing training conditions cannot be inferred as hidden test targets without external ground truth. |
| **8. Real Reynolds discrepancy is random noise** | N/A | **FALSE** | *Directly observed* | Zero intra-condition jitter; linear affine drift $\Delta Re \approx 0.002636 \times Re - 9.94$. |
| **9. Reynolds number causes point error degradation** | Implied | **NOT SUPPORTED** | *Directly observed* | Point Rel-L2 is U-shaped ($0.0947 \to 0.0744 \to 0.0914$). High Re is not worse than low Re. |
| **10. Angle of attack causes point error degradation** | Unexamined | **SUPPORTED** | *Directly observed* | Rel-L2 grows monotonically from $0.0535$ to $0.1165$ ($+118\%$). |
| **11. High Re and high AoA cause SPS miscalibration** | Unexamined | **SUPPORTED** | *Directly observed* | SPS drops from $59.79$ to $33.07$; $u$-coverage collapses from $95.8\%$ to $74.8\%$. |
| **12. Controlled held-Re-out experiment warranted** | Recommended (GroupKFold) | **GATE CLOSED FOR POINT MODEL / GATED UQ PROPOSAL** | *Supported inference* | Point models do not need Re-out retraining; UQ calibration requires targeted channel scaling. |

---

## 9. Decision Gate: Evaluation & Controlled Experiment Proposal

### Gate Evaluation (Section F Rules):
The project rules specify that a held-Reynolds-out experiment may be opened **only if**:
1. *Performance varies materially and systematically with Re*: **Partial / Channel-specific**. Point Rel-L2 does NOT degrade systematically, but TKE increases by $+26\%$ and SPS collapses by $-11.22$ points.
2. *Low-coverage Real regimes have clearly worse performance*: **Supported**. The high-Re sparse coverage regime ($Re=24204$, missing $AoA=5^\circ$) shows the lowest SPS ($40.75$) and lowest coverage ($84.29\%$).
3. *SPS miscalibration is materially concentrated by Re/AoA*: **Strongly Supported**. Measured 26.7-point drop in SPS from low Re/low AoA to high Re/high AoA.
4. *Sim-supported parameter coverage plausibly explains incumbent weakness*: **Supported Inference**. Sim provides complete coverage of $Re=15225, 27975$ and all $AoA=5^\circ$ configurations.

### Gate Verdict:
- **FOR POINT ARCHITECTURES / BACKBONE RETRAINING**: **GATE CLOSED**.
  - Retraining the CNO backbone or fine-tuning point predictors with a "Leave-Reynolds-Out" split is **NOT warranted**. The point predictor interpolates smoothly across Reynolds numbers (Rel-L2 at $Re=24204$ is $0.0914$, better than $Re=6306$ at $0.0947$). Architectural effort spent on Reynolds extrapolation would be addressing a non-existent bottleneck.
- **FOR UQ CALIBRATION (ADAPTIVE SPS)**: **GATED PROPOSAL (DO NOT RUN WITHOUT EXPLICIT APPROVAL)**.
  - The true vulnerability identified by Hypothesis 2 is **UQ undercoverage at high Reynolds and high AoA** (specifically in the $u$-velocity wake dynamics).
  - Rather than retraining the neural operator, this defect is addressed by **regime-adaptive UQ scaling** (e.g. Reynolds/AoA-dependent $s_u$ expansion).

### Controlled Experiment Protocol (If Authorized):
If the team chooses to authorize an experiment on held-Reynolds generalization, the exact valid protocol is defined as:

1. **Held-Out Partition**:
   - Hold out the **entire nominal $Re=6300$ group** (actual $Re=6306$, 5 Real trajectories, 30 windows) OR **nominal $Re=24150$ group** (actual $Re=24204$, 4 Real trajectories, 20 windows).
   - Strict requirement: **Zero window leakage**. No trajectory from the held-out Reynolds operating point may contribute any history or target windows to training or validation.
2. **Controlled Pair**:
   - **Variant A (Incumbent Sim+Real)**: Sim pre-trained CNO backbone + fine-tuned on remaining Real data.
   - **Variant B (Matched Real-Only)**: Identical architecture, seeds (42, 43), learning rate schedule, and optimization budget, trained solely on the remaining Real data without Sim pre-training.
3. **Evaluation**:
   - Evaluated strictly on the held-out Real Reynolds group.
   - Metrics: Rel-L2, TKE, MVPE, SPS, channel-wise coverage.
4. **Current Status**: **BLOCKED / PROPOSED ONLY**. Do not execute without explicit authorization.

---

## 10. Answers to Core Scientific Questions

### What is the actual joint distribution of Reynolds number, AoA, trajectory length, and timestep spacing in train_sim and train_real?
- **Sim**: 100 trajectories, spanning 20 Re $\times$ 5 AoA. All trajectories have $T=1000$ frames, $t \in [0.08, 20.08]\text{ s}$, uniform $\Delta t = 0.020020\text{ s}$.
- **Real**: 82 trajectories, spanning 18 nominal Re operating points (actual internal Re $3750 \to 26761$) and 5 AoAs. 79 trajectories have $T=868$ frames ($t \in [0.08, 17.42]\text{ s}$), 1 has $T=607$, 1 has $T=492$, and 1 has $T=282$. Uniform sampling interval is $\Delta t = 0.020000\text{ s}$ ($50\text{ Hz}$).

### How much of the Sim parameter space is covered by Real?
- Real covers **82 out of 100 nominal conditions (82.0%)**.
- 18 Sim-supported conditions are absent from the observed Real training set: 10 from two completely missing Reynolds numbers ($15225, 27975$) and 8 from scattered configurations, heavily concentrated at high Re ($AoA=5^\circ$ missing for $Re \ge 22875$).

### Are filename Reynolds/AoA labels identical to the internal HDF5 metadata?
- **In Sim**: Yes, 100% identical.
- **In Real**: AoA is 100% identical. Reynolds number is **NOT identical in 95.12% of files** (78/82). It exhibits a systematic, monotonic positive offset ranging from $+3$ to $+61$. Filenames cannot be used as physical ground truth.

### Is there any evidence that incumbent model performance or uncertainty calibration degrades systematically with Reynolds number, AoA, or sparse Real coverage?
- **Point Field Error (Rel-L2)**: Does NOT degrade systematically with Reynolds number ($0.0947 \to 0.0744 \to 0.0914$). Degrades **strictly monotonically with AoA** ($0.0535 \to 0.1165$, $+118\%$).
- **Uncertainty Calibration (SPS)**: Degrades **severely with both Reynolds number and AoA**. SPS drops from $59.79$ (low Re, low AoA) to $33.07$ (high Re, high AoA). High Reynolds regimes with sparse Real coverage exhibit severe $u$-channel undercoverage ($79.90\%$).

### What new information did Hypothesis 2 provide that can change a modeling or validation decision?
1. **Validation Pipeline Correction**: Corrected the false belief that all trajectories have $T=607$ and $\Delta t = 0.05\text{ s}$. Real trajectories have $T=868$ and $\Delta t = 0.02\text{ s}$ ($50\text{ Hz}$). Any metric or physical integration assuming $20\text{ Hz}$ was erroneous.
2. **Metadata Hygiene**: Proved that any future model conditioning on physical Reynolds number MUST read the internal HDF5 scalar `re`, because filename parsing produces an error of up to $+61$.
3. **Strategic Modeling Decision**: Proved that the incumbent CNO point predictor does **not** suffer from a Reynolds generalization bottleneck. Attempting to build specialized Reynolds-extrapolation architectures would be a misallocation of effort.
4. **UQ Roadmap Guidance**: Pinpointed that the primary vulnerability of `sub7` is **undercoverage in the high-dynamic wake regime ($u$-velocity at high Re and high AoA)**. Future improvements should focus on regime-adaptive uncertainty bands rather than point operator re-engineering.

---

## 11. Artifact Paths and Reproducibility

All code, data tables, and plots have been generated and archived in the local repository:

### Core Data Artifacts:
- **Per-File Metadata Table (182 files)**: [`research/hypothesis02_audit/per_file_metadata.csv`](file:///d:/Project/NeurIPS/research/hypothesis02_audit/per_file_metadata.csv)
- **Domain Summary Table**: [`research/hypothesis02_audit/domain_summary.csv`](file:///d:/Project/NeurIPS/research/hypothesis02_audit/domain_summary.csv)
- **Nominal vs Actual Reynolds Table**: [`research/hypothesis02_audit/nominal_vs_actual_re.csv`](file:///d:/Project/NeurIPS/research/hypothesis02_audit/nominal_vs_actual_re.csv)
- **Condition Coverage Matrix (100 cells)**: [`research/hypothesis02_audit/condition_coverage.csv`](file:///d:/Project/NeurIPS/research/hypothesis02_audit/condition_coverage.csv)
- **Nearest Sim-Re Distance Table**: [`research/hypothesis02_audit/nearest_sim_re_distance.csv`](file:///d:/Project/NeurIPS/research/hypothesis02_audit/nearest_sim_re_distance.csv)

### Incumbent Performance Stratification Artifacts:
- **Per-Sample Metrics Table (80 test windows)**: [`research/hypothesis02_audit/incumbent_per_sample_metrics.csv`](file:///d:/Project/NeurIPS/research/hypothesis02_audit/incumbent_per_sample_metrics.csv)
- **Stratification by Reynolds Group**: [`research/hypothesis02_audit/incumbent_stratification_by_re.csv`](file:///d:/Project/NeurIPS/research/hypothesis02_audit/incumbent_stratification_by_re.csv)
- **Stratification by Angle of Attack**: [`research/hypothesis02_audit/incumbent_stratification_by_aoa.csv`](file:///d:/Project/NeurIPS/research/hypothesis02_audit/incumbent_stratification_by_aoa.csv)
- **Stratification by Joint (Re, AoA) Cell**: [`research/hypothesis02_audit/incumbent_stratification_by_re_aoa.csv`](file:///d:/Project/NeurIPS/research/hypothesis02_audit/incumbent_stratification_by_re_aoa.csv)

### High-Resolution Figures:
- **Parameter Space Coverage**: [`research/hypothesis02_audit/parameter_space_coverage.png`](file:///d:/Project/NeurIPS/research/hypothesis02_audit/parameter_space_coverage.png)
- **Nominal vs Actual Reynolds Number Discrepancy**: [`research/hypothesis02_audit/reynolds_nominal_vs_actual.png`](file:///d:/Project/NeurIPS/research/hypothesis02_audit/reynolds_nominal_vs_actual.png)
- **Temporal Distribution**: [`research/hypothesis02_audit/temporal_distribution.png`](file:///d:/Project/NeurIPS/research/hypothesis02_audit/temporal_distribution.png)
- **Incumbent Performance Stratification**: [`research/hypothesis02_audit/incumbent_performance_stratification.png`](file:///d:/Project/NeurIPS/research/hypothesis02_audit/incumbent_performance_stratification.png)
- **Incumbent Grid Heatmaps**: [`research/hypothesis02_audit/incumbent_grid_heatmaps.png`](file:///d:/Project/NeurIPS/research/hypothesis02_audit/incumbent_grid_heatmaps.png)

### Reproducibility Commands:
To reproduce all audit tables and figures from scratch:
```powershell
# 1. Run exhaustive HDF5 inspection (182 files in archive.zip)
python research/hypothesis02_audit/audit_data.py

# 2. Run incumbent stratification across Re and AoA
python research/hypothesis02_audit/stratify_incumbent.py

# 3. Generate parameter space and temporal plots
python research/hypothesis02_audit/analyze_parameter_space.py

# 4. Generate incumbent stratification plots and heatmaps
python research/hypothesis02_audit/plot_incumbent_stratification.py
```
