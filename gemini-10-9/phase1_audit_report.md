# RealPDE Track 1 — Phase 1 Comprehensive Audit Report (G0 & G1)
**Directory**: `gemini-10-9`  
**Date**: 2026-09-10  
**Incumbent Reference**: `sub7_cno_zeromean_head_adaptive_sps.zip` (Official Hidden Leaderboard Score: **79.158181**)  
**Authority Hierarchy**: `architecture-short.md` > Protocol v2 (`architecture-decision.md`) > Historical `walkthrough.md`.

---

## Executive Summary & Gate Decisions

| Gate / Component | Status / Result | Key Findings & Evidence |
| :--- | :---: | :--- |
| **G0: Measurement Integrity & Provenance** | **BLOCKED** | Live research inference matches packaged `sub7` bit-for-bit at batch 16 (`max_abs = 0.0`), but legacy batch-1 cached CNO predictions cannot be interchanged with deployed batch-16 outputs (`max_abs = 1.976e-4`). Provenance of surviving local checkpoints is incomplete: 7 local candidates exist and reproduce predictions, but run-bound training logs/schedulers are uncertified. |
| **2×2 Checkpoint Reuse Matrix** | **7 Exist, 1 Missing** | 7 of 8 target checkpoints exist in `research/exp_zero_mean_head/artifacts/checkpoints/`. Exactly **1 checkpoint is missing**: `Ordinary λ=0.30 seed 43`. Minimum new runs required: **1** if historical protocol is certified, up to **8** if a clean matched protocol is re-run. |
| **G1.A1: ZeroMean Algebra** | **SUPPORTED** | Algebraic property strictly holds before downstream masking: $\max \|\text{mean}_t(\mathbf{r}')\| = 2.421 \times 10^{-9} \le 10^{-6}$ and $\max \|\text{mean}_t(\text{base} + \mathbf{r}') - \text{mean}_t(\text{base})\| = 1.490 \times 10^{-7} \le 10^{-6}$. Downstream masking alters the mean by up to $0.01784$ at masked pixels. |
| **G1.A2: Persistent-Zero Guardrail** | **SUPPORTED OFFLINE WITH REGRESSIONS** | Improves aggregate dev RelL2 ($0.08655 \to 0.08623$), TKE ($0.58097 \to 0.58061$), and MVPE ($0.08212 \to 0.08201$). However, it causes regressions on $5.0\%$ of windows ($4/80$) for MVPE and $1.25\%$ ($1/80$) for TKE. $9.23\%$ of history-persistent-zero pixels become nonzero in the future. |
| **G1.A3: Constant vs Adaptive SPS** | **SUPPORTED** | Point metrics identical by construction. Adaptive SPS increases offline SPS score ($45.0858 \to 47.8390$, $+2.7532$ points) and coverage ($84.38\% \to 90.26\%$) with minimal GPU latency overhead ($0.0026\text{ s} \to 0.0180\text{ s}$ per 80 windows). |
| **G1.A4: SPS Calibration Structure** | **SUPPORTED AS STRUCTURED SIGNAL** | Severe channel asymmetry (u coverage $86.75\%$ vs v coverage $93.82\%$). Extreme undercoverage in high history-std regimes for u ($74.22\%$ in Q4 vs $94.74\%$ in Q1). Supports **ONE** controlled UQ candidate after G0 resolution. |
| **Overall Decision** | **STOP & RETAIN SUB7** | Retain immutable `sub7` incumbent. Do NOT proceed to G2 or G3 yet. Measurement/provenance block must be formally addressed first. |

---

## 1. G0 — Measurement Integrity & Provenance Audit

### 1.1 Source-of-Truth Artifact Table
All files audited directly in the local repository with SHA-256 checksums:

#### Submission ZIP Packages
| Submission Package | File Size (Bytes) | SHA-256 Checksum | Status / Reusability |
| :--- | :---: | :--- | :--- |
| `sub1_cno_sps.zip` | 29,653,891 | `51620e1a3256df389cf83a9310ff84660b3b20560c224365d04404ff3711bf9a` | Baseline reference (Sim+Real CNO, constant band) |
| `sub5_cno_head_adaptive_sps.zip` | 30,070,239 | `833cb41267ae6da2788fcb7ccac3e5379c9e76bf38a25d37ce9f79fda716dc57` | Intermediate milestone ($\lambda=0.10$, GPU adaptive SPS) |
| `sub6_cno_zeromean_head_sps.zip` | 30,070,284 | `5b4a6d597013dfecff5275a4179511c4cf1789e36e6ea4f4338de0e5d7afaaa3` | Point-predictor twin of sub7 (constant SPS band) |
| `sub7_cno_zeromean_head_adaptive_sps.zip` | 30,070,489 | `2e42b16da2eba7e794d90579003e31c9417d3e720f7babdb932bd376f810762a` | **Current Incumbent** (ZeroMean $\lambda=0.30$, GR, GPU Adaptive SPS) |

#### Model Checkpoints Lineage & Provenance
| Checkpoint Path | Architecture | Seed | $\lambda$ | Parameters | Selection Rule | Provenance Status |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| `sim_real_cno.pth` (in sub1/5/6/7) | Continuous Neural Operator (CNO3d) | unknown | — | 7,960,467 | best_val_loss at iter 5000 | Organizer pre-trained weights; immutable backbone |
| `sub7: cno_head_weights.pth` | Dual ZeroMean Residual Heads | 42, 43 | 0.30 | $2 \times 55,528$ | Validation proxy | Exact bit-level match to local `Model_S_ZeroMean_L0.30_seed42/43` |
| `Model_C_Ordinary_L0_seed42.pth` | OrdinaryResidualHead | 42 | 0.0 | 55,528 | Validation proxy (best ep 12) | Exists; full training provenance uncertified |
| `Model_C_Ordinary_L0_seed43.pth` | OrdinaryResidualHead | 43 | 0.0 | 55,528 | Validation proxy (best ep 9) | Exists; full training provenance uncertified |
| `Model_D_ZeroMean_L0_seed42.pth` | ZeroMeanResidualHead | 42 | 0.0 | 55,528 | Validation proxy (best ep 11) | Exists; full training provenance uncertified |
| `Model_D_ZeroMean_L0_seed43.pth` | ZeroMeanResidualHead | 43 | 0.0 | 55,528 | Validation proxy (best ep 9) | Exists; full training provenance uncertified |
| `Model_H_Ordinary_L0.30_seed42.pth` | OrdinaryResidualHead | 42 | 0.30 | 55,528 | Validation proxy (best ep 11) | Exists; full training provenance uncertified |
| `Model_H_Ordinary_L0.30_seed43.pth` | OrdinaryResidualHead | 43 | 0.30 | — | — | **MISSING** from local repository |
| `Model_S_ZeroMean_L0.30_seed42.pth` | ZeroMeanResidualHead | 42 | 0.30 | 55,528 | Validation proxy (best ep 10) | Exists; packaged into `sub6` and `sub7` |
| `Model_S_ZeroMean_L0.30_seed43.pth` | ZeroMeanResidualHead | 43 | 0.30 | 55,528 | Validation proxy (best ep 12) | Exists; packaged into `sub6` and `sub7` |
| `loss_screen: primary_lambda_0.3_seed43.pth` | OrdinaryResidualHead | 43 | 0.30 | 55,528 | Min validation field RelL2 | **Incompatible selection policy** (not proxy selected) |

### 1.2 Checkpoint Reuse Matrix for the Matched 2×2 Experiment
The 2×2 experiment requires:
$$\begin{array}{c|cc}
\text{Architecture} & \lambda = 0.0 & \lambda = 0.30 \\
\hline
\text{Ordinary} & \text{O0 (seeds 42, 43)} & \text{O3 (seeds 42, 43)} \\
\text{ZeroMean} & \text{Z0 (seeds 42, 43)} & \text{Z3 (seeds 42, 43)}
\end{array}$$

| Checkpoint | Architecture | $\lambda$ | Seed | Exists? | Compatible Protocol? | Reusable? | Exact Reason / Required Action |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **O0_s42** | Ordinary | 0.0 | 42 | Yes | Unresolved | Pending Audit | Checkpoint exists (`Model_C_Ordinary_L0_seed42.pth`, best ep 12); historical training log linkage uncertified. |
| **O0_s43** | Ordinary | 0.0 | 43 | Yes | Unresolved | Pending Audit | Checkpoint exists (`Model_C_Ordinary_L0_seed43.pth`, best ep 9); historical training log linkage uncertified. |
| **O3_s42** | Ordinary | 0.30 | 42 | Yes | Unresolved | Pending Audit | Checkpoint exists (`Model_H_Ordinary_L0.30_seed42.pth`, best ep 11); historical training log linkage uncertified. |
| **O3_s43** | Ordinary | 0.30 | 43 | **No** | **No** | **No** | **Missing from local cohort.** Older Kaggle loss-screen run exists but used a different selection rule (field RelL2 vs proxy). |
| **Z0_s42** | ZeroMean | 0.0 | 42 | Yes | Unresolved | Pending Audit | Checkpoint exists (`Model_D_ZeroMean_L0_seed42.pth`, best ep 11); historical training log linkage uncertified. |
| **Z0_s43** | ZeroMean | 0.0 | 43 | Yes | Unresolved | Pending Audit | Checkpoint exists (`Model_D_ZeroMean_L0_seed43.pth`, best ep 9); historical training log linkage uncertified. |
| **Z3_s42** | ZeroMean | 0.30 | 42 | Yes | Unresolved | Pending Audit | Checkpoint exists (`Model_S_ZeroMean_L0.30_seed42.pth`, best ep 10); packaged into `sub7`. Historical logs uncertified. |
| **Z3_s43** | ZeroMean | 0.30 | 43 | Yes | Unresolved | Pending Audit | Checkpoint exists (`Model_S_ZeroMean_L0.30_seed43.pth`, best ep 12); packaged into `sub7`. Historical logs uncertified. |

**Practical Question Answer**:
> *Do we really need eight new runs, or is the causal table missing only one or two controls such as Ordinary $\lambda=0.30$ seed 43?*

**Answer**:
- **7 of the 8 required model checkpoints already exist** in `research/exp_zero_mean_head/artifacts/checkpoints/`.
- Only **ONE model checkpoint is completely missing**: `Ordinary λ=0.30, seed 43` (`Model_H_Ordinary_L0.30_seed43.pth`).
- However, because the historical training logs and exact per-epoch random states for the surviving 7 checkpoints were not recorded in immutable run-bound logs, full protocol certification remains open.
- Therefore:
  - **Best case (if historical protocol is accepted/certified)**: **1 new training run** (`O3_s43`).
  - **Conservative case (if clean simultaneous re-execution is required)**: **8 new training runs**.
  - **Status**: No training runs are executed in Phase 1.

### 1.3 Sample Identity & Fold Integrity Audit
- **Window Count & Explicit Identification**:
  - Exactly 477 windows across 81 trajectories:
    - `train`: 339 windows (57 trajectories)
    - `validation`: 58 windows (10 trajectories)
    - `test`: 80 windows (14 trajectories)
  - Every sample is explicitly identified by `sample_id = file:start:20:20` (e.g. `13950_0.h5:160:20:20`).
  - `sample_id` uniqueness: **100% unique** (477/477).
- **Cross-Fold Trajectory and Frame Overlap**:
  - Maximum folds per trajectory: **1** (strict trajectory-level split; no trajectory crosses folds).
  - Cross-fold frame overlap: **0 windows** (zero leakage).
- **Raw Cache vs. Precomputed Target Verification**:
  - Verified bit-for-bit against raw cache `.npz` targets on all 80 test windows.
  - Max absolute difference between raw cache and precomputed target: **0.0**.
- **Audit of Historical Hypo 13/14 Re Labeling**:
  - Diagnostic notebooks hardcoded Re labels by positional slicing: rows 0–27 $\to$ 6306, rows 28–53 $\to$ 13977, rows 54–79 $\to$ 24204.
  - Manifest truth: rows 0–29 $\to$ 13977, rows 30–49 $\to$ 24204, rows 50–79 $\to$ 6306.
  - **Scope Determination**: **Scope is strictly Category A (diagnostic labels only)**.
    - Model training scripts (`train_and_ablate.py`, `loss_experiment.py`) assigned folds using `meta.Re` from HDF5 file metadata.
    - Test evaluation scripts aligned targets by row order matching `manifest.csv`.
    - The positional row error was confined purely to post-hoc plotting scripts in Hypo 13 and 14. Aggregate model training, validation, and evaluation were completely unaffected.

### 1.4 Research-Path vs. Packaged-Submission Parity Audit
Evaluated on all 80 test windows under float32 precision on GPU:

| Comparator | Max Absolute Diff | Max Relative Diff (Ref $\ge 10^{-3}$) | Metric Delta ($\Delta\text{RelL2}, \Delta\text{TKE}, \Delta\text{MVPE}$) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Legacy Cached Base / Features vs. Package (Batch 16)** | $1.9759 \times 10^{-4}$ | $7.13\%$ | $[+1.34 \times 10^{-7}, -1.32 \times 10^{-5}, +1.56 \times 10^{-6}]$ | **FAIL** (exceeds $10^{-6}$ tolerance) |
| **Fresh Live Research Path (Batch 16) vs. Package (Batch 16)** | $\mathbf{0.0}$ | $\mathbf{0.0}$ | $[\mathbf{0.0}, \mathbf{0.0}, \mathbf{0.0}]$ | **PASS** (exact parity) |
| **CNO Live Batch 1 vs. Live Batch 16** | $1.9346 \times 10^{-4}$ | $10.17\%$ | — | Diagnostic: CNO Conv3d/GroupNorm batch-dependent |

**Root Cause Analysis**:
- The discrepancy in the legacy cache arises because `feature_cache.npz` cached CNO base predictions generated at **batch size 1**, whereas packaged `sub7` executes in **batches of 16**.
- When the research path is executed live at the matching batch size of 16, **it matches packaged `sub7` bit-for-bit** ($0.0$ difference across all outputs).
- **Rule**: Do not use legacy batch-1 cached predictions as an exact substitute for deployed batch-16 outputs.

### 1.5 Scorer Audit & Formula Verification
Scorer implementation inspected at `realpde_t1_starting_kit_v9/realpde_t1_starting_kit_v9/scoring.py`:
1. **RelL2**:
   $$\text{RelL2} = \frac{1}{N}\sum_{i=1}^N \frac{\|\hat{\mathbf{u}}_i - \mathbf{u}_i\|_2}{\max(\|\mathbf{u}_i\|_2, 10^{-8})}, \quad \text{Score}_{\text{Rel}} = \frac{100}{1 + 0.5 \times \text{RelL2}}$$
2. **TKE (Turbulent Kinetic Energy)**:
   $$K(x, y) = \frac{1}{2}\left(\frac{1}{T}\sum_t (u_t - \bar{u})^2 + \frac{1}{T}\sum_t (v_t - \bar{v})^2\right)$$
   $$\text{TKE Error} = \frac{1}{N}\sum_{i=1}^N \frac{\|K_{\text{pred}} - K_{\text{target}}\|_2}{\max(\|K_{\text{target}}\|_2, 10^{-8})}, \quad \text{Score}_{\text{TKE}} = \frac{100}{1 + 0.5 \times \text{TKE Error}}$$
3. **MVPE (Mean Velocity Profile Error)**:
   - Evaluates temporal mean velocity at probes located at 4 streamwise stations ($x$) and up to 9 vertical locations ($y$).
   $$\text{MVPE Error} = \frac{1}{N}\sum_{i=1}^N \text{mean}_{\text{stations}}\left(\frac{\|\bar{\mathbf{u}}_{\text{pred}} - \bar{\mathbf{u}}_{\text{target}}\|_2}{\max(\|\bar{\mathbf{u}}_{\text{target}}\|_2, 10^{-8})}\right), \quad \text{Score}_{\text{MVPE}} = \frac{100}{1 + 0.5 \times \text{MVPE Error}}$$
4. **Time Score**:
   $$r = \frac{t_{\text{neural}}}{0.72896}, \quad \text{ST} = \frac{1}{1 + \sqrt{r}}, \quad \text{Score}_{\text{Time}} = 100 \times \text{ST}$$
5. **SPS (Sample Prediction Score / Uncertainty)**:
   - Scored strictly on non-zero target pixels ($target \ne 0.0$):
   $$\text{nil} = \frac{\text{upper} - \text{lower}}{\sigma_{\text{global}}}, \quad \sigma_{\text{global}} = 0.0563870259$$
   $$\text{Accuracy Factors: } a_{\text{Rel}} = \frac{0.5}{0.5 + \text{RelL2}}, \quad a_{\text{TKE}} = \frac{0.5}{0.5 + \text{TKE}}, \quad a_{\text{MVPE}} = \frac{0.5}{0.5 + \text{MVPE}}$$
   $$\text{SPS} = \frac{1}{N_{\text{scored}}}\sum_{\text{scored}} \mathbf{1}_{t \in [l, u]} \cdot e^{-\text{nil}} \cdot (0.5 a_{\text{Rel}} + 0.3 a_{\text{TKE}} + 0.2 a_{\text{MVPE}})$$
   $$\text{Score}_{\text{SPS}} = 100 \times \text{clamp}(\text{SPS}, 0, 1)$$
6. **Coverage**:
   $$\text{Coverage} = \frac{\sum_{\text{scored}} \mathbf{1}_{t \in [l, u]}}{N_{\text{scored}}}$$
7. **Final Aggregation Formula**:
   - **CRITICAL FINDING**: The starting kit scorer **DOES NOT publish the official final combination formula**.
   - Lines 4–8 and 324–325 of `scoring.py` explicitly state that the leaderboard combination is unpublished and local scoring stops at subscores.
   - Any local linear combination (e.g. $0.25 \text{Rel} + 0.25 \text{TKE} + 0.25 \text{MVPE} + 0.15 \text{SPS} + 0.10 \text{Time}$) is designated strictly as an:
     $$\textbf{Offline Development Composite Proxy}$$
     and must NOT be treated as an official or calibrated final score.

---

## 2. G1 — Audit of Currently Deployed Components

All G1 diagnostics were executed using the exact deployed `sub7` point-prediction tensor on the verified 80-window development test set.

### A1 — ZeroMean Algebra & Implementation Check
- **Hypothesis**: The ZeroMean projection guarantees $\text{mean}_t(\mathbf{r}') \equiv 0$ and preserves the base temporal mean before masking.
- **Comparator**:
  $$Y_{\text{base}} \quad \text{vs.} \quad Y_{\text{base}} + \mathbf{r}', \quad \mathbf{r}'_t = \mathbf{r}_t - \frac{1}{T}\sum_{\tau=1}^T \mathbf{r}_\tau$$
- **Numerical Findings**:
  - Max absolute correction temporal mean: $\max |\text{mean}_t(\mathbf{r}')| = \mathbf{2.421 \times 10^{-9}} \le 10^{-6}$.
  - Max absolute temporal mean shift: $\max |\text{mean}_t(Y_{\text{base}} + \mathbf{r}') - \text{mean}_t(Y_{\text{base}})| = \mathbf{1.490 \times 10^{-7}} \le 10^{-6}$.
  - Shift after persistent-zero masking: $\max |\text{mean}_t(Y_{\text{final}}) - \text{mean}_t(Y_{\text{base}})| = \mathbf{0.01784}$.
  - Mean at masked pixels: identically $\mathbf{0.0}$.
- **Epistemic Classification**: **SUPPORTED**. The algebraic mean-zero property is strictly satisfied before post-processing. Downstream spatial masking alters the final temporal mean at masked locations.

### A2 — Persistent-Zero Guardrail (OFF vs. ON)
- **Hypothesis**: Post-processing predictions with the history-derived persistent-zero mask ($M_{\text{active}} = 1 - \mathbf{1}[\text{all 20 history frames are exactly 0}]$) improves predictor accuracy.
- **Comparator**: Exact same deployed point predictor with guardrail OFF vs. ON.
- **Quantitative Findings**:
  - **Aggregate Errors**:
    - RelL2: OFF $= 0.0865529 \to$ ON $= 0.0862319$ ($\Delta = -0.0003210$, improved).
    - TKE: OFF $= 0.5809697 \to$ ON $= 0.5806146$ ($\Delta = -0.0003550$, improved).
    - MVPE: OFF $= 0.0821229 \to$ ON $= 0.0820055$ ($\Delta = -0.0001174$, improved).
  - **Window Regression Rates**:
    - RelL2 regressions: **0.0%** (0/80 windows).
    - TKE regressions: **1.25%** (1/80 windows regresses, file `13950_0.h5`).
    - MVPE regressions: **5.0%** (4/80 windows regress).
  - **Spatial Region Breakdown**:
    - Total pixels masked: $12,846 / 163,840$ sample-pixels (**7.84%** of domain).
    - Future target nonzero in mask: **9.23%** of masked pixels have nonzero targets in future frames!
    - Inside mask MSE: drops by an order of magnitude ($5.607 \times 10^{-6} \to 5.877 \times 10^{-7}$).
    - Outside mask MSE: strictly unchanged ($1.4098 \times 10^{-4} \to 1.4098 \times 10^{-4}$).
- **Trade-off & Assessment**: **SUPPORTED OFFLINE WITH QUANTIFIED REGRESSIONS**.
  - The guardrail provides a consistent, small aggregate gain across all three point metrics offline.
  - Trade-off: It causes regressions on $5\%$ of windows for MVPE and truncates legitimate flow when the wake transiently sweeps into previously quiet pixels.
  - Decision: **Retain in incumbent `sub7` based on offline evidence**, but do not claim independent hidden benefit as no matched mask-OFF submission exists.

### A3 — Constant vs. Adaptive SPS on Fixed Predictions
- **Hypothesis**: Input-conditioned uncertainty half-width improves interval scoring without altering point metrics.
- **Comparator**: sub6 constant bounds ($0.85 \times 0.010925$) vs. sub7 adaptive bounds ($\text{base\_hw} + 0.85 \cdot \sigma_{\text{hist}} \cdot \sqrt{h/10}$).
- **Quantitative Findings**:
  - Point metrics (RelL2, TKE, MVPE): **Strictly identical**.
  - SPS Score: **$45.0858 \to 47.8390$** ($+2.7532$ points offline).
  - Coverage: **$84.38\% \to 90.26\%$** ($+5.89\%$ offline).
  - Mean Width: $0.01857 \to 0.02065$ ($+11.2\%$).
  - Median Width: $0.01857 \to 0.01711$ ($-7.9\%$).
  - Channel Width: u width $= 0.02716$, v width $= 0.01415$.
  - GPU Latency (all 80 windows): Constant $= 2.58\text{ ms}$ vs. Adaptive $= 17.97\text{ ms}$ ($\approx 0.22\text{ ms / sample}$). Negligible runtime cost.
- **Epistemic Classification**: **SUPPORTED**. Strong offline evidence replicating the external hidden gain ($\text{SPS } 35.51 \to 37.58$).

### A4 — Residual SPS Calibration Structure
- **Hypothesis**: There is structured, systematic residual calibration error in the adaptive SPS policy that could justify a targeted UQ improvement.
- **Diagnostic Breakdown**:
  1. **Channel Asymmetry**:
     - u channel: Coverage $= \mathbf{86.75\%}$, $\text{MAE} = 0.00851$, SPS Contribution $= 41.17$.
     - v channel: Coverage $= \mathbf{93.82\%}$, $\text{MAE} = 0.00300$, SPS Contribution $= 54.60$.
     - *Finding*: u channel is substantially undercovered compared to v.
  2. **History Standard Deviation Quantiles** (edges derived strictly from 339 training trajectories):
     - u:Q1 (lowest std): Coverage $= 94.74\%$, $\text{MAE} = 0.00318$
     - u:Q2: Coverage $= 94.01\%$, $\text{MAE} = 0.00381$
     - u:Q3: Coverage $= 86.28\%$, $\text{MAE} = 0.00663$
     - u:Q4 (highest std): Coverage $= \mathbf{74.22\%}$, $\text{MAE} = 0.01933$
     - *Finding*: u-channel coverage collapses severely in high-fluctuation wake regions ($74.22\%$), while v coverage remains stable ($91.00\%$ in Q4).
  3. **Forecast Horizon ($h=1\dots 20$)**:
     - Coverage starts at $87.38\%$ at $h=1$, peaks around $91.15\%$ mid-horizon, and finishes at $89.14\%$ at $h=20$.
- **Epistemic Classification**: **SUPPORTED AS STRUCTURED SIGNAL**.
  - A clear, systematic calibration imbalance exists: **u-channel high-fluctuation regimes suffer severe undercoverage**.
  - This provides plausible justification for **ONE controlled UQ candidate** (e.g. channel-specific scaling or non-linear history-std widening) in Gate G2, once G0 conditions are formally satisfied.

---

## 3. Decision & Answers to Required Questions

### Question 1: Is sub7 research-path / package parity trustworthy?
**YES**, for live batch-16 inference under matching float32 precision.  
Parity between live research inference and packaged `sub7` is exact ($\max |\Delta| = 0.0$, raw metric delta $= 0.0$).  
However, legacy batch-1 cached predictions must NOT be substituted for deployed batch-16 outputs ($\max |\Delta| = 1.976 \times 10^{-4}$).

### Question 2: Should the persistent-zero guardrail remain in the incumbent based on current offline evidence?
**YES, RETAIN IN INCUMBENT**.  
The guardrail delivers consistent aggregate improvements offline across RelL2 ($-0.00032$), TKE ($-0.00035$), and MVPE ($-0.00012$). While it incurs regressions on $5\%$ of windows for MVPE and incorrectly masks $9.2\%$ of pixels that become transiently active, retaining it preserves the validated configuration of `sub7`.

### Question 3: Does adaptive SPS show a remaining structured calibration error worth one controlled UQ experiment?
**YES**.  
A4 reveals severe undercoverage in the u channel during high-fluctuation regimes ($74.22\%$ in Q4 vs $94.74\%$ in Q1), contrasting with well-calibrated v-channel coverage ($91.00\%$). This structured pattern justifies exploring **at most one** simple UQ refinement (e.g. channel-specific scaling or non-linear wake broadening).

### Question 4: Exactly which matched Ordinary / ZeroMean checkpoints are still missing?
- For the local proxy-selected cohort, exactly **ONE checkpoint is missing**: `Ordinary λ=0.30 seed 43` (`Model_H_Ordinary_L0.30_seed43.pth`).
- The other 7 required checkpoints exist in `research/exp_zero_mean_head/artifacts/checkpoints/` and reproduce saved test predictions bit-for-bit.
- However, full historical training provenance (run-bound configs, per-epoch random orderings, schedulers) remains uncertified for the surviving 7 checkpoints.
- Minimum number of new matched-learning runs required: **1** (if historical protocol is accepted) or **8** (if a fresh certified cohort is trained).

### Question 5: Based on evidence, what should the NEXT gate be?
$$\textbf{STOP AND RETAIN SUB7 FOR NOW (G0 BLOCKED)}$$
- G0 is formally blocked from certifying the 2×2 matched causal table because historical training-protocol equivalence cannot be proven from surviving metadata.
- Therefore, do NOT start G3 or retrain checkpoints yet.
- Once measurement and provenance conditions are closed:
  - The highest expected ROI path supported by hidden-validated evidence is **G2-UQ before G3** (exploring one simple UQ modification to address the A4 u-channel calibration deficit without altering point predictions).
  - No G2, G3, G4, or G5 work was initiated in this task.
