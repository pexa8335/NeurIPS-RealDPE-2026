# RealPDE Track 1 — Architecture Decision & Evidence Audit

**Directory**: `gemini-10-9`  
**Date**: 2026-09-10  
**Incumbent Reference**: `sub7_cno_zeromean_head_adaptive_sps.zip` (Official Hidden Leaderboard Score: **79.158181**)  
**Authority Hierarchy**: `architecture-short.md` > Protocol v2 (`architecture-decision.md`) > Historical `walkthrough.md`.

---

## Executive Gate Summary

| Gate / Decision | Status | Summary of Evidence & Impact |
| :--- | :---: | :--- |
| **G0-UQ (Uncertainty Track)** | **PASS** | `sub7` artifact and SHA-256 are verified; live research-vs-package parity passes at canonical batch size 16 (`max_abs = 2.98e-8`); sample identities and splits are verified; local scorer reproduces; G2-UQ is completely decoupled from historical model training provenance. |
| **G0-G3 (Architecture Track)** | **BLOCKED / Incomplete** | Surviving local checkpoints reproduce predictions bit-for-bit, but run-bound training logs/schedulers/orderings are uncertified. Local proxy `Ordinary λ=0.30 seed 43` is absent. Matched 2×2 causal claims remain blocked until provenance recovery. |
| **G1 (Incumbent Component Audit)** | **COMPLETED** | A1 (ZeroMean algebra) verified; A2 (Guardrail) retained with quantified local regressions; A3 (Adaptive SPS) reproduced; A4 (Calibration structure) identified structured under-coverage on high-variance $u$. |
| **G2-UQ (Development Screen)** | **EXPLORATORY SCREEN (NOT MET IN TESTED WIDENING FAMILY)** | Reclassified as an exploratory development screen (validation partition was inspected in preliminary sweeps before final candidate declaration). Simple widening of $u$ variance scaling increases coverage but decreases SPS ($-0.13$ to $-1.04$); widening is not supported. An exploratory narrowing observation ($s_u=0.70$) beat baseline by $+0.156$ on validation; formulated as a frozen candidate for secondary grouped confirmation. |
| **G4 (Secondary Grouped Confirmation)** | **`UQ CANDIDATE PASSES SECONDARY CONFIRMATION`** | Evaluated on the 80-window grouped development partition (14 trajectories, Re 6306/13977/24204). Aggregate SPS improved from `47.8390` to `47.9938` ($\Delta = +0.1548$). Improvement replicates across **14/14 trajectories**, **3/3 Reynolds groups**, and **5/5 AoA groups** ($100\%$ positive sign). Point metrics identical. Latency identical ($<0.05\text{ ms/window}$). Recommends single controlled exploratory `sub8` leaderboard submission. |
| **Final Incumbent Decision** | **`KEEP CURRENT SUB7 BOUNDS` (Pending User Direction on Sub8)** | Current repo incumbent `sub7` remains strictly immutable. Zero models retrained, no submissions launched in this task. |

---

## Explicit Contradictions & Historical Reconciliation

| Issue / Topic | Historical `walkthrough.md` Claim | Current Authority (`architecture-short.md` & latest audit) | Audited Reality & Provenance Reconciliation |
| :--- | :--- | :--- | :--- |
| **`sub5` Subscores** | Recorded with asterisks (`RelL2: 94.564*`, `TKE: 74.231*`, `MVPE: 93.537*`, `Time: 87.36*`, `SPS: 37.47*`, `Final: 79.100273`) | User-supplied milestone comparison (`+0.057908` final gain over `sub5`) | Subscores were approximate user notes; final composite is verified. Only `sub1`, `sub6`, `sub7` have raw Codabench records locally. |
| **`sub3` Milestone** | Conflicting local records ($78.68$ vs $78.931207$) | Excluded as comparative milestone | No raw Codabench JSON exists locally; excluded as an anchor. |
| **`sub7` Size & Capacity** | Stated 30,070,381 bytes and 55,144 params in early text | 30,070,489 bytes; 55,528 params per head | Exact audit: 30,070,489 bytes (SHA-256 `2e42b16da2...`), exactly 55,528 parameters per head (3 Conv2D layers, width 32, GroupNorm, GELU). |
| **Guardrail Regressions** | Claimed zero window regressions universally | Offline dev holdout shows regressions on 4/80 windows for MVPE and 1/80 for TKE | Beneficial on aggregate (RelL2 `-0.00032`, TKE `-0.00035`, MVPE `-0.00012`), but has quantified local trade-offs (`13950_0.h5` TKE `+5.32e-5`). |
| **ZeroMean Invariance** | Claimed permanent 0.00% MVPE degradation across full pipeline | Algebraic invariance $\frac{1}{T}\sum r'_t \equiv 0$ holds strictly before downstream masking | Post-processing `final_uv * active_mask` modifies temporal mean at masked pixels (max base-mean shift `0.01785`). |
| **Ordinary vs ZeroMean** | Claimed decisive proof that Ordinary overfits at $\lambda \ge 0.25$ | Superiority across equal ensembles is unproven | Confounded by ensembling: Ordinary was evaluated with 1 seed while ZeroMean had 2 seeds. |
| **Hypo 13/14 Claims** | Claimed "89% of error is phase" and "$Re^{0.678}$ scaling law" | Unexplained residual is not pure phase; $Re^{0.678}$ is an empirical curve-fit on 18 conditions | Scalar rescaling leaves ~89% of fluctuation error unexplained (spatial shape, wake localization, etc.). Hardcoded Re labels were also wrong on 78/80 test rows (diagnostic only). |
| **Persistent-Zero Interpretation** | Labeled $M_{\text{extra}}$ as "PIV shadow" and $M_{\text{both}}$ as "airfoil" | Physical cause unverified; designated neutrally as `persistent_zero_mask` | 9.23% of persistent-zero pixels become nonzero in future frames; cannot be assumed solid geometry or shadow. |
| **Final Score Combination** | Used local linear combination (~80.08 / 81.43) | Unpublished in local starting kit | Local formulas are strictly an **offline development composite proxy**, not an official calibrated final score. |
| **Engineering Tolerance** | Protocol v2 referenced "predeclared float32 engineering tolerances" without values | Explicitly locked: `atol=1e-6, rtol=1e-5`, raw metric tolerance `1e-6`, algebra tolerance `1e-6` | Resolved by predeclaring explicit Phase-1 tolerances before measurement. |

---

## 1. G0 Result: Measurement Integrity & Provenance

### 1.1 Artifact / Provenance Source of Truth
- **Immutable Submission ZIPs**:
  - `sub1_cno_sps.zip` (29,653,891 bytes, SHA-256 `51620e1a3256df389cf83a9310ff84660b3b20560c224365d04404ff3711bf9a`)
  - `sub5_cno_head_adaptive_sps.zip` (30,070,239 bytes, SHA-256 `833cb41267ae6da2788fcb7ccac3e5379c9e76bf38a25d37ce9f79fda716dc57`)
  - `sub6_cno_zeromean_head_sps.zip` (30,070,284 bytes, SHA-256 `5b4a6d597013dfecff5275a4179511c4cf1789e36e6ea4f4338de0e5d7afaaa3`)
  - `sub7_cno_zeromean_head_adaptive_sps.zip` (30,070,489 bytes, SHA-256 `2e42b16da2eba7e794d90579003e31c9417d3e720f7babdb932bd376f810762a`)
- **Backbone & Adapter Lineage**:
  - Backbone: `sim_real_cno.pth` (7,960,467 params, SHA-256 `12513f56e9...`, stored `iteration=5000`).
  - Deployed adapter: Dual `ZeroMeanResidualHead` (55,528 params each, seeds 42 & 43, $\lambda=0.30$). Tensors match local checkpoints `Model_S_ZeroMean_L0.30_seed42/43.pth` bit-for-bit.

### 1.2 Sample Identity & Overlap Findings
- All 477 windows verified by exact input/target float equality: Train 339 (57 trajectories), Validation 58 (10 trajectories), Development Test 80 (14 trajectories).
- Explicit IDs: `file:start:20:20`. Zero trajectory crosses folds; zero frame interval overlaps across folds.
- **Hypo 13/14 Re-labeling Issue**: Scope is strictly **A (diagnostic labels only)**. Hardcoded wrong Re labels in notebooks did not contaminate aggregate prediction/target alignment.

### 1.3 Research-vs-Package Parity Result
- **Fresh Live Research Path vs. Package (Batch 16, float32)**:
  - Max absolute difference: **`2.980232e-8`** ($\le 10^{-6}$, PASS)
  - Max relative difference: **`6.376534e-6`** ($\le 10^{-5}$, PASS)
  - Raw metric deltas (RelL2 / TKE / MVPE): `−1.86e-10` / `−2.61e-9` / `+3.49e-10` ($\le 10^{-6}$, PASS)
- **Legacy Batch-1 Cache vs. Package (Batch 16)**:
  - Max absolute difference: **`1.975894e-4`** (FAIL; not interchangeable due to CNO batch accumulation numerics).

### 1.4 Scorer Audit
- RelL2, TKE, MVPE, SPS formulas in [`scoring.py`](file:///d:/Project/NeurIPS/realpde_t1_starting_kit_v9/realpde_t1_starting_kit_v9/scoring.py) verified.
- Starting kit does NOT publish the official final combination formula; local combinations are strictly **offline development composite proxies**.

---

## 2. Checkpoint Reuse Matrix (Matched 2×2 Experiment)

| Cell | Seed | Candidate Checkpoint | Exists? / Best Ep | Compatible? | Reusable for Causal Claim? | Missing Information / Action |
| :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| **O0** | 42 | `Model_C_Ordinary_L0_seed42.pth` | Yes (Ep 12) | Unresolved | No (pending provenance) | Run config/order/scheduler linkage missing; replays exact |
| **O0** | 43 | `Model_C_Ordinary_L0_seed43.pth` | Yes (Ep 9) | Unresolved | No (pending provenance) | Run config/order/scheduler linkage missing; replays exact |
| **O3** | 42 | `Model_H_Ordinary_L0.30_seed42.pth` | Yes (Ep 11) | Unresolved | No (pending provenance) | Run config/order/scheduler linkage missing; replays exact |
| **O3** | 43 | `Model_H_Ordinary_L0.30_seed43.pth` | **No** | No | No | **Local proxy cell absent**; older field-selected run incompatible |
| **Z0** | 42 | `Model_D_ZeroMean_L0_seed42.pth` | Yes (Ep 11) | Unresolved | No (pending provenance) | Run config/order/scheduler linkage missing; replays exact |
| **Z0** | 43 | `Model_D_ZeroMean_L0_seed43.pth` | Yes (Ep 9) | Unresolved | No (pending provenance) | Run config/order/scheduler linkage missing; replays exact |
| **Z3** | 42 | `Model_S_ZeroMean_L0.30_seed42.pth` | Yes (Ep 10) | Unresolved | No (pending provenance) | Deployed in sub6/7; historical training linkage uncertified |
| **Z3** | 43 | `Model_S_ZeroMean_L0.30_seed43.pth` | Yes (Ep 12) | Unresolved | No (pending provenance) | Deployed in sub6/7; historical training linkage uncertified |

**Minimum number of new matched-learning runs required: N = unknown (confirmed lower bound 1, conservative upper bound 8).**  
No training was launched.

---

## 3. G1 Component Audit (Saved Predictions)

- **A1 (ZeroMean Algebra)**: Max physical correction mean: `1.49e-9`; max addition mean shift: `8.94e-8` (both $\le 10^{-6}$). Masked mean shift: `0.01785`. **SUPPORTED** as an algebraic property before masking.
- **A2 (Persistent-Zero Guardrail)**: RelL2 `0.08655 → 0.08623`, TKE `0.58097 → 0.58061`, MVPE `0.08212 → 0.08201`. 5% MVPE regressions; 9.23% history-zero pixels become nonzero in future. **SUPPORTED OFFLINE WITH REGRESSIONS (Retained in incumbent)**.
- **A3 (Constant vs. Adaptive SPS)**: SPS `45.0858 → 47.8390` (+2.7532 points), coverage `84.38% → 90.26%`, point metrics identical. **SUPPORTED**.
- **A4 (Residual Calibration Structure)**: Severe asymmetry ($u$ coverage 86.75% vs. $v$ 93.82%; Q4 drops to 74.22%). **SUPPORTED AS STRUCTURED SIGNAL**.

---

## 4. G2-UQ Exploratory Development Screen — u-channel history-std scaling

### 4.1 Evidentiary Reclassification & Limitations
1. **Exploratory Development Status**: The G2-UQ experiment is classified strictly as an **exploratory development UQ screen**, NOT a fully predeclared or confirmatory test. Execution records confirm that the validation partition (58 windows) was evaluated in preliminary sweeps across multiple slopes and Q4 boost factors before the final candidate subset was formalized.
2. **Development Holdout Caveat**: The 80-window development holdout is **not an independent confirmation set**, as it has been inspected repeatedly throughout earlier phases of the project.
3. **No Global Optimality Claim**: Neither $s_u = 0.85$ nor the incumbent `sub7` interval rule is claimed to be globally optimal. `s_u = 0.85` is established solely as the **best among the tested final widening variants**.
4. **Narrow Supported Finding**:
   > **Increasing the tested u-channel history-std scaling improved coverage but decreased SPS; therefore simple widening of this dimension is not supported as the next UQ improvement.**

---

### 4.2 Complete Preliminary Validation Sweep Data

All preliminary validation sweeps evaluated on the 58-window validation partition prior to candidate subset declaration are recorded below.  
Artifacts: [`gemini-10-9/g2_uq_preliminary_sweep.csv`](file:///d:/Project/NeurIPS/gemini-10-9/g2_uq_preliminary_sweep.csv), [`gemini-10-9/g2_uq_preliminary_sweep.json`](file:///d:/Project/NeurIPS/gemini-10-9/g2_uq_preliminary_sweep.json).

#### Table 4.2a: Full Preliminary Slope Sweep on $u$ (Validation Partition)
*Formula*: $hw(u) = \text{clip}(0.0075 + s_u \cdot \sigma_{\text{hist}}(u) \cdot \sqrt{h/10}, 0.0075, 0.040)$, with frozen $v$ and point predictions.

| Parameter $s_u$ | Variant Type | Val SPS ↑ | Val $\Delta$ SPS (vs 0.85) | Val Coverage % | Mean Width |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **0.70** | **Narrowing (Exploratory)** | **47.664464** | **+0.156323** | 88.4415% | 0.02009547 |
| **0.85** | **Incumbent Baseline (`sub7`)** | **47.508141** | **0.000000** | **89.4638%** | **0.02105612** |
| 0.95 | Widening (Evaluated as C1) | 47.376708 | -0.131434 | 90.0360% | 0.02164955 |
| 1.05 | Widening (Evaluated as C2) | 47.228805 | -0.279336 | 90.5342% | 0.02220908 |
| 1.15 | Widening | 47.071289 | -0.436852 | 90.9724% | 0.02273768 |
| 1.25 | Widening | 46.903917 | -0.604225 | 91.3493% | 0.02323782 |
| 1.35 | Widening | 46.730843 | -0.777298 | 91.6772% | 0.02371190 |
| 1.50 | Widening | 46.468784 | -1.039357 | 92.1019% | 0.02437918 |

#### Table 4.2b: Full Preliminary Selective Q4 Boost Sweep on $u$ (Validation Partition)
*Formula*: $s_u = 0.85 + \text{boost} \cdot \mathbf{1}_{\sigma_{\text{hist}}(u) > q_{75}}$, where $q_{75} = 0.00949892$ is fixed from training histories.

| Q4 Boost Value | Formula on $u$ | Val SPS ↑ | Val $\Delta$ SPS (vs 0.85) | Val Coverage % | Mean Width |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **0.0** | $s_u = 0.85$ (Baseline) | **47.508141** | 0.000000 | 89.4638% | 0.02105612 |
| 0.1 | $s_u = 0.85 + 0.10 \cdot \mathbf{1}_{Q4}$ | 47.444183 | -0.063958 | 89.7811% | 0.02142337 |
| **0.2** | $s_u = 0.85 + 0.20 \cdot \mathbf{1}_{Q4}$ (C3) | 47.377743 | -0.130398 | 90.0521% | 0.02175672 |
| 0.3 | $s_u = 0.85 + 0.30 \cdot \mathbf{1}_{Q4}$ | 47.309921 | -0.198220 | 90.2801% | 0.02205914 |
| 0.5 | $s_u = 0.85 + 0.50 \cdot \mathbf{1}_{Q4}$ | 47.176179 | -0.331962 | 90.6268% | 0.02258099 |

---

### 4.3 Analysis of Narrowing vs. Widening
1. **Response to Key Question**:
   > *Did any already-tested narrowing configuration, especially $s_u=0.70$, outperform the incumbent $s_u=0.85$ on validation SPS?*
   
   **YES.** The narrowing configuration **$s_u = 0.70$** achieved a validation SPS of **`47.664464`**, outperforming the incumbent $s_u = 0.85$ (`47.508141`) by **`+0.156323`** points. Mean interval width tightened from $0.021056$ to $0.020095$, while empirical coverage dropped moderately from $89.46\%$ to $88.44\%$.

2. **Monotonic Widening Degradation**:
   Across all tested widening configurations ($s_u \in [0.95, 1.50]$ and selective Q4 boosts $\in [0.1, 0.5]$), empirical coverage strictly increased, but validation SPS strictly degraded (up to $-1.039$ points). In the official scoring function:
   $$\text{element\_score} = \mathbf{1}_{y \in [lo, hi]} \cdot \exp\left(-\frac{\text{width}}{0.056387}\right) \cdot \left(0.5 a_{\text{RelL2}} + 0.3 a_{\text{TKE}} + 0.2 a_{\text{MVPE}}\right)$$
   Because the exponential width penalty applies globally across all covered pixels, widening intervals to capture outlier tail points incurs an exponential score penalty on the vast majority (~75–88%) of points already inside the interval. Conversely, tightening width reduces the exponential penalty on all covered points, which on validation outweighed the ~1% loss in tail coverage.

---

### 4.4 Final Candidates Diagnostic Comparison (80-Window Development Holdout)
Script: [`gemini-10-9/run_g2_uq_experiment.py`](file:///d:/Project/NeurIPS/gemini-10-9/run_g2_uq_experiment.py).  
Artifacts: [`gemini-10-9/g2_uq_comparison_summary.csv`](file:///d:/Project/NeurIPS/gemini-10-9/g2_uq_comparison_summary.csv), [`gemini-10-9/g2_uq_val_results.json`](file:///d:/Project/NeurIPS/gemini-10-9/g2_uq_val_results.json), [`gemini-10-9/g2_uq_test_results.json`](file:///d:/Project/NeurIPS/gemini-10-9/g2_uq_test_results.json).

| Configuration | Val SPS ↑ | Val $\Delta$ SPS | Val Cov % | Test SPS ↑ | Test $\Delta$ SPS | Test Cov % | Test Mean Width | Test Median Width | Test $u$ Q4 Cov % | GPU Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (`sub7`, $s_u=0.85$)** | **47.508141** | 0.000000 | 89.46% | **47.839039** | 0.000000 | 90.26% | 0.020653 | 0.017110 | 74.22% | 265.0 ms |
| **C1 (Mild Widening, $s_u=0.95$)** | 47.376708 | -0.131434 | 90.04% | 47.707489 | -0.131551 | 90.78% | 0.021195 | 0.017319 | 76.31% | 148.4 ms |
| **C2 (Moderate Widening, $s_u=1.05$)** | 47.228805 | -0.279336 | 90.53% | 47.559873 | -0.279166 | 91.23% | 0.021708 | 0.017522 | 78.05% | 142.0 ms |
| **C3 (Selective Q4 Boost, $+0.20$)** | 47.377743 | -0.130398 | 90.05% | 47.721870 | -0.117169 | 90.75% | 0.021246 | 0.017110 | 78.05% | 131.0 ms |

Point prediction metrics are bit-for-bit identical across all candidates:
$$\text{RelL2} = 0.08623190, \quad \text{TKE} = 0.58061481, \quad \text{MVPE} = 0.08200549$$

---

### 4.5 Formal Gate Decision & Progression to G4

1. **Simple $u$-Widening Branch: CLOSED (`NOT MET IN THE TESTED FAMILY`)**
   - Simple widening of $u$-channel variance scaling is conclusively rejected across all tested linear factors and selective Q4 boosts.
   - Widening searches and sweeps terminated.

2. **Exploratory Narrowing Finding ($s_u=0.70$): PROTOCOL DISCIPLINE APPLIED**
   - The $+0.156$ validation SPS improvement observed for $s_u = 0.70$ was exploratory only.
   - Validation tuning ceased immediately (no finer slope search).
   - Formulated **one frozen narrowing candidate** ($s_u = 0.70$) for evaluation on a secondary grouped confirmation partition.

---

## 5. G4 — Secondary Grouped Development Confirmation: Frozen Narrowing Candidate ($s_u = 0.70$)

`frozen candidate (s_u=0.70) → partition exposure audit → frozen comparator → multi-level stability → UQ CANDIDATE PASSES SECONDARY CONFIRMATION`

### 5.1 Data-Use Ledger & Partition Exposure Audit Before Scoring

Before evaluating candidate performance on confirmation data, an exhaustive project data-use ledger was constructed across all available partitions:

| Partition / Cohort | Size & Trajectories | Reynolds Groups | Historical Project Exposure Audit | Confirmation Status & Label |
| :--- | :---: | :---: | :--- | :--- |
| **Train** | 339 windows (57 trajectories) | 7 Re groups | **Trained point models**: Backbone CNO pretraining (Sim+Real); residual heads (seeds 42/43) in `exp_zero_mean_head`, `sub5`, `sub6`, `sub7`. Quartile threshold fitting ($q_{75}$). | **Disqualified** for confirmation (in-sample training partition). |
| **Validation** | 58 windows (10 trajectories) | 2 Re groups (10142, 20369) | **Monitored validation loss** during head training; **architecture & proxy selection** ($\lambda=0.30$); **preliminary UQ screen** where $s_u=0.70$ was discovered. | **Disqualified** for confirming $s_u=0.70$ (candidate was discovered on this partition). |
| **Development Test** | 80 windows (14 trajectories) | 3 Re groups (6306, 13977, 24204) | **Never used in training backprop or model selection**. Inspected for guardrail ON/OFF (G1.A2), calibration diagnostics (G1.A3/A4), and diagnostic check of widening candidates (C1–C3). **NEVER evaluated for $s_u = 0.70$**. | **Selected as least-exposed grouped partition**. Designated as **`secondary grouped development confirmation`** (not an independent/pristine test). |

#### Frozen Partition Manifest
The confirmation partition was frozen strictly as the **80 development test windows** across 14 trajectories and 3 Reynolds numbers not present in validation:
- **Trajectories (14)**: `13950_0.h5`, `13950_5.h5`, `13950_10.h5`, `13950_15.h5`, `13950_20.h5`, `24150_0.h5`, `24150_10.h5`, `24150_15.h5`, `24150_20.h5`, `6300_0.h5`, `6300_5.h5`, `6300_10.h5`, `6300_15.h5`, `6300_20.h5`.
- **Reynolds Numbers**: Re 6306 (30 windows), Re 13977 (30 windows), Re 24204 (20 windows).
- **Angles of Attack**: 0°, 5°, 10°, 15°, 20°.
- **Window Format**: Explicit IDs `file:start:20:20`. Zero overlap with train or validation.

---

### 5.2 Frozen Comparator Specification

The point prediction tensor and all non-candidate bounds components were strictly frozen:
- **Frozen Point Predictor**: Incumbent `sub7` deployed pipeline (Frozen CNO + Dual ZeroMean Residual Heads $\lambda=0.30$ seeds 42 & 43 + Persistent-Zero Guardrail).
- **Frozen $v$-Channel Formula**: $hw(v) = \text{clip}(0.0032 + 0.85 \cdot \sigma_{\text{hist}}(v) \cdot \sqrt{h/10}, \min=0.0032, \max=0.040)$ (bit-for-bit identical).
- **Frozen Bounds Architecture**: Horizon scaling $\sqrt{h/10}$, base width $[0.0075, 0.0032]$, maximum clip $0.040$, float32 precision, canonical batch size 16.
- **Comparison Only**:
  - **Baseline**: $hw(u) = \text{clip}(0.0075 + \mathbf{0.85} \cdot \sigma_{\text{hist}}(u) \cdot \sqrt{h/10}, 0.0075, 0.040)$
  - **Frozen Narrowing Candidate**: $hw(u) = \text{clip}(0.0075 + \mathbf{0.70} \cdot \sigma_{\text{hist}}(u) \cdot \sqrt{h/10}, 0.0075, 0.040)$

#### Point Metrics Verification
Point predictions are bit-for-bit identical by explicit calculation:
$$\text{RelL2} = 0.08623190, \quad \text{TKE} = 0.58061481, \quad \text{MVPE} = 0.08200549$$
Deltas across all three point metrics: $\Delta \equiv 0.00000000$.

---

### 5.3 Confirmation Aggregate Results

Script: [`gemini-10-9/run_secondary_confirmation.py`](file:///d:/Project/NeurIPS/gemini-10-9/run_secondary_confirmation.py).  
Artifacts: [`gemini-10-9/secondary_confirmation_results.json`](file:///d:/Project/NeurIPS/gemini-10-9/secondary_confirmation_results.json), [`gemini-10-9/confirmation_per_trajectory.csv`](file:///d:/Project/NeurIPS/gemini-10-9/confirmation_per_trajectory.csv), [`gemini-10-9/confirmation_per_reynolds.csv`](file:///d:/Project/NeurIPS/gemini-10-9/confirmation_per_reynolds.csv), [`gemini-10-9/confirmation_per_aoa.csv`](file:///d:/Project/NeurIPS/gemini-10-9/confirmation_per_aoa.csv).

| Metric | Baseline (`sub7`, $s_u=0.85$) | Frozen Candidate ($s_u=0.70$) | Delta (Candidate − Baseline) |
| :--- | :---: | :---: | :---: |
| **SPS Score ↑** | **47.839039** | **47.993838** | **+0.154799** |
| **Empirical Coverage %** | 90.2639% | 89.3352% | -0.9288% |
| **Mean Interval Width** | 0.020653 | 0.019779 | -0.000873 |
| **Median Interval Width** | 0.017110 | 0.016785 | -0.000326 |
| **Mean $u$-Channel Width** | 0.027159 | 0.025413 | -0.001746 |
| **Mean $v$-Channel Width** | 0.014146 | 0.014146 | 0.000000 |
| **$u$-Channel Coverage %** | 86.7540% | 84.9085% | -1.8455% |
| **$v$-Channel Coverage %** | 93.8199% | 93.8199% | 0.000000 |
| **GPU Latency (Vectorized)** | 3.63 ms / 80 windows | 3.63 ms / 80 windows | 0.00 ms (<0.05 ms/window) |

#### Coverage Across Forecast Horizons ($h=1 \dots 20$)
- $h=1$: Baseline $87.38\% \to$ Candidate $86.70\%$
- $h=5$: Baseline $89.94\% \to$ Candidate $89.03\%$
- $h=10$: Baseline $90.78\% \to$ Candidate $89.84\%$
- $h=15$: Baseline $91.11\% \to$ Candidate $90.16\%$
- $h=20$: Baseline $89.14\% \to$ Candidate $88.05\%$

#### $u$-Channel Coverage by Training History-Std Quartile
- **Q1** ($\le 0.001547$): Baseline $94.74\% \to$ Candidate $94.46\%$ ($-0.28\%$)
- **Q2** ($0.001547 - 0.003416$): Baseline $94.01\% \to$ Candidate $93.34\%$ ($-0.67\%$)
- **Q3** ($0.003416 - 0.009499$): Baseline $86.28\% \to$ Candidate $84.25\%$ ($-2.03\%$)
- **Q4** ($> 0.009499$): Baseline $74.22\% \to$ Candidate $70.33\%$ ($-3.89\%$)

---

### 5.4 Natural Grouping Stability Audit

Windows within trajectories share physical flow dynamics; evaluating stability at the independent grouping level (trajectories, Reynolds numbers, AoA) is required to rule out localized anomalies.

#### Table 5.4a: Per-Trajectory Stability (14 Independent Trajectories)

| Trajectory File | Re | AoA | Windows | Base SPS | Cand SPS | $\Delta$ SPS | Sign | Base Cov % | Cand Cov % | Base Mean Width | Cand Mean Width |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `13950_0.h5` | 13977 | 0° | 6 | 55.5749 | 55.7241 | **+0.1492** | **+** | 93.93% | 93.32% | 0.01559 | 0.01504 |
| `13950_5.h5` | 13977 | 5° | 6 | 53.9096 | 54.0897 | **+0.1800** | **+** | 93.65% | 92.85% | 0.01674 | 0.01602 |
| `13950_10.h5` | 13977 | 10° | 6 | 46.5779 | 46.7988 | **+0.2208** | **+** | 90.49% | 89.48% | 0.02209 | 0.02103 |
| `13950_15.h5` | 13977 | 15° | 6 | 45.3361 | 45.5925 | **+0.2564** | **+** | 90.04% | 88.94% | 0.02246 | 0.02128 |
| `13950_20.h5` | 13977 | 20° | 6 | 41.3421 | 41.4942 | **+0.1521** | **+** | 86.39% | 85.06% | 0.02608 | 0.02478 |
| `24150_0.h5` | 24204 | 0° | 6 | 49.8693 | 49.9716 | **+0.1023** | **+** | 89.82% | 88.77% | 0.01867 | 0.01781 |
| `24150_10.h5` | 24204 | 10° | 6 | 39.5246 | 39.6620 | **+0.1373** | **+** | 84.48% | 83.28% | 0.02913 | 0.02793 |
| `24150_15.h5` | 24204 | 15° | 6 | 35.7305 | 35.8068 | **+0.0763** | **+** | 80.06% | 78.47% | 0.03020 | 0.02873 |
| `24150_20.h5` | 24204 | 20° | 2 | 33.0729 | 33.1328 | **+0.0599** | **+** | 80.32% | 78.82% | 0.03517 | 0.03376 |
| `6300_0.h5` | 6306 | 0° | 6 | 59.7939 | 60.0114 | **+0.2176** | **+** | 97.30% | 96.96% | 0.01566 | 0.01525 |
| `6300_5.h5` | 6306 | 5° | 6 | 56.6789 | 56.9057 | **+0.2268** | **+** | 96.41% | 96.01% | 0.01558 | 0.01513 |
| `6300_10.h5` | 6306 | 10° | 6 | 51.5782 | 51.7464 | **+0.1682** | **+** | 94.09% | 93.53% | 0.01584 | 0.01530 |
| `6300_15.h5` | 6306 | 15° | 6 | 47.3688 | 47.4999 | **+0.1311** | **+** | 92.20% | 91.35% | 0.01754 | 0.01685 |
| `6300_20.h5` | 6306 | 20° | 6 | 44.9962 | 45.0308 | **+0.0346** | **+** | 88.74% | 87.75% | 0.01805 | 0.01733 |

**Result**: Exactly **14 out of 14 trajectories** exhibit positive SPS gains ($\Delta \text{SPS} > 0$). **100% positive sign rate**.

#### Table 5.4b: Per-Reynolds Group Stability (3 Held-Out Re Groups)

| Reynolds Number | Trajectories | Windows | Base SPS | Cand SPS | $\Delta$ SPS | Sign | Base Cov % | Cand Cov % | Base Mean Width | Cand Mean Width |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Re 6306** | 5 | 30 | 51.9715 | 52.1255 | **+0.1540** | **+** | 93.68% | 93.04% | 0.01654 | 0.01597 |
| **Re 13977** | 5 | 30 | 48.4395 | 48.6311 | **+0.1916** | **+** | 90.84% | 89.87% | 0.02059 | 0.01963 |
| **Re 24204** | 4 | 20 | 40.7538 | 40.8546 | **+0.1008** | **+** | 84.29% | 82.99% | 0.02692 | 0.02572 |

**Result**: All **3 out of 3 Reynolds groups** show positive SPS improvement.

#### Table 5.4c: Per-AoA Group Stability (5 Angles of Attack)

| Angle of Attack | Trajectories | Windows | Base SPS | Cand SPS | $\Delta$ SPS | Sign | Base Cov % | Cand Cov % | Base Mean Width | Cand Mean Width |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **AoA 0°** | 3 | 18 | 55.0923 | 55.2487 | **+0.1565** | **+** | 93.69% | 93.03% | 0.01664 | 0.01603 |
| **AoA 5°** | 2 | 12 | 55.2913 | 55.4947 | **+0.2034** | **+** | 95.03% | 94.43% | 0.01616 | 0.01557 |
| **AoA 10°** | 3 | 18 | 45.8392 | 46.0145 | **+0.1753** | **+** | 89.65% | 88.72% | 0.02235 | 0.02142 |
| **AoA 15°** | 3 | 18 | 42.8197 | 42.9744 | **+0.1547** | **+** | 87.44% | 86.26% | 0.02340 | 0.02229 |
| **AoA 20°** | 3 | 14 | 41.7387 | 41.8274 | **+0.0886** | **+** | 86.54% | 85.33% | 0.02394 | 0.02287 |

**Result**: All **5 out of 5 AoA groups** show positive SPS improvement.

---

### 5.5 Formal Decision Rule Evaluation & Protocol Outcome

#### Decision Rule Check:
1. **Higher Aggregate SPS**: Candidate SPS `47.9938` > Baseline `47.8390` ($\Delta = +0.154799$). (**PASSED**)
2. **Gain Not Confined to Local Outlier**: 14/14 trajectories, 3/3 Re groups, and 5/5 AoA groups are positive ($100\%$ positive sign). (**PASSED**)
3. **No Pathological Coverage Collapse**: Empirical coverage shifts from $90.26\%$ to $89.34\%$ ($-0.9288\%$), maintaining healthy high coverage across all horizons. (**PASSED**)
4. **Point Metrics Bit-for-Bit Identical**: $\text{RelL2} = 0.08623190, \text{TKE} = 0.58061481, \text{MVPE} = 0.08200549$ ($\Delta = 0.00000000$). (**PASSED**)
5. **Runtime Impact Negligible**: Latency overhead is $0.00\text{ ms}$ (identical tensor graph, identical GPU arithmetic operations). (**PASSED**)

#### Gate Designation:
# **`UQ CANDIDATE PASSES SECONDARY CONFIRMATION`**

#### Standard Evidentiary Statement:
> **"A fixed reduction of the u-channel history-std slope from 0.85 to 0.70, selected during exploratory development, reproduced an SPS improvement on the secondary grouped confirmation partition while leaving point predictions fixed."**

*Strict Caveat*: This evaluation is conducted on the secondary grouped development confirmation partition (which has prior project exposure for guardrail/calibration diagnostics); it is *not* an untouched blind test. $s_u = 0.70$ is not claimed to be globally optimal.

#### Next Action Recommendation:
Recommend packaging the frozen candidate ($s_u=0.70$) into a single controlled exploratory submission:  
**`sub8_cno_zeromean_head_adaptive_sps_narrow`**  
*(Official packaging, submission, and test evaluation withheld pending explicit user instruction).* Current repo incumbent remains `sub7`.

