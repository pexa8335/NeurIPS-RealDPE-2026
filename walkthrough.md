# RealPDE Track 1 — Research Record & Technical Walkthrough (sub1 → sub7)

**Document Purpose**: This document serves as the authoritative, scientifically audited research record for the RealPDE Track 1 Sim2Real fluid forecasting pipeline. All statements are strictly categorized into **Facts**, **Empirical Evidence**, **Interpretation**, and **Not Established**.

---

## 1. Provenance & Master Leaderboard Table

### A. Evaluator Records (Official Hidden Leaderboard)
The following records represent official submissions evaluated on the Codabench hidden test server:

| Submission | Pipeline Configuration | `rel_l2_score` | `tke_score` | `mvpe_score` | `time_score` | `sps_score` | `final_score` (Reported) | Gain vs Baseline |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`sub1`** | **Incumbent Baseline**: Sim+Real CNO point predictor + constant SPS band ($0.85 \times \text{RMS}$) | 94.504578 | 73.717525 | 93.484537 | **87.472436** | 35.149137 | **78.421963** | — (Anchor) |
| **`sub5`** | CNO + 55k Residual Head ($\lambda=0.10$, seeds 42 & 43) + **GPU Adaptive SPS** | 94.564* | 74.231* | 93.537* | 87.36* | 37.47* | **79.100273** | +0.678310 |
| **`sub6`** | CNO + **Zero-Mean Head** ($\lambda=0.30$, seeds 42 & 43) + **Persistent-Zero Guardrail** + Constant SPS | 94.551736 | **74.823387** | 93.487926 | **87.356676** | 35.513169 | **78.639483** | +0.217520 |
| **`sub7` (CURRENT BEST)** | CNO + **Zero-Mean Head** ($\lambda=0.30$, seeds 42 & 43) + **Persistent-Zero Guardrail** + **GPU Adaptive SPS** | 94.551736 | **74.823387** | 93.487926 | 87.277381 | **37.579004** | **`79.158181`** | $\mathbf{+0.736218}$ |

\* *Provenance Note on `sub5`: approximate user-recorded subscores; final composite score (79.100273) verified.*

> [!NOTE]
> **Sub3 Provenance Note**: An earlier intermediate submission (`sub3`, testing residual head with constant bounds) had conflicting local records ($78.68$ vs $78.931207$). Because no raw Codabench JSON record exists locally to adjudicate `sub3`, it is omitted as a comparative milestone. The primary verified reference points are **`sub1`** (baseline), **`sub5`** (adaptive SPS milestone), **`sub6`** (clean point predictor test), and **`sub7`** (current best).

### B. Mathematical Delta Decomposition
1. **The Counterfactual on Hidden Test (`sub6` → `sub7`)**:
   - `sub6` and `sub7` share the **exact identical point predictor**:
     $$\text{RelL2} = 94.551736, \quad \text{TKE} = 74.823387, \quad \text{MVPE} = 93.487926$$
   - The only change was replacing the constant uncertainty band with the input-conditioned adaptive band:
     $$\Delta \text{SPS} = 37.579004 - 35.513169 = \mathbf{+2.065835}$$
     $$\Delta \text{Time} = 87.277381 - 87.356676 = -0.079295$$
     $$\Delta \text{Final} = 79.158181 - 78.639483 = \mathbf{+0.518698}$$
   - **Evidence**: This is strong hidden-set evidence that replacing the constant band with the adaptive band improved SPS and final score while leaving point-prediction metrics unchanged.

2. **The Progression from `sub5` → `sub7`**:
   - Sub7 exceeds Sub5 by **$+0.057908$ final points**.
   - Comparing point predictors between `sub5` and `sub7`:
     - TKE score gained substantially: $74.231 \to 74.823$ ($\mathbf{+0.592}$ points).
     - RelL2 score slightly decreased: $94.564 \to 94.552$ ($-0.012$ points).
     - MVPE score slightly decreased: $93.537 \to 93.488$ ($-0.049$ points).
   - **Interpretation**: Zero-Mean Head at $\lambda=0.30$ is a **TKE-specialized improvement with small trade-offs** across other point metrics, not a uniform improvement across all four objectives.

---

## 2. Scientific Audit: Facts vs. Overclaims

| Area | Stated in Draft / Hypotheses | Audited Reality & Strict Finding | Epistemic Status |
| :--- | :--- | :--- | :---: |
| **Baseline Error Diagnosis** | "RelL2 error ≈ 8.7%, TKE error ≈ 64.9%, MVPE ≈ 8.2%" | These percentages ($0.0874, 0.6491, 0.0821$) are **offline development holdout errors**, NOT hidden test errors. Using the supplied local `score_error` mapping ($score = \frac{100}{1+e/2}$), hidden `sub1` scores invert to implied errors $e_{\text{Rel}} \approx 0.116$, $e_{\text{TKE}} \approx 0.713$, $e_{\text{MVPE}} \approx 0.139$. These are local mathematical inversions, not officially reported Codabench raw errors. | **Correction Applied** |
| **Hypo 13 (TKE Rescaling)** | "89% of TKE error is structural spatial phase error" | Optimal scalar amplitude rescaling reduced TKE relative error from $0.6491 \to 0.5795$ (~$11\%$ improvement). The remaining ~$89\%$ discrepancy is simply **unexplained by scalar amplitude correction**; it may stem from spatial phase, wake localization, dispersion, shape mismatch, or temporal decorrelation. | **Correction Applied** |
| **Hypo 14 (Reynolds Fit)** | "Strict Reynolds Scaling Law: $\text{Wake\_RMS} \sim Re^{0.678}$" | This is an **empirical power-law fit** on the 18 available Real condition groups ($R^2=0.9296$). Furthermore, `sub5` and `sub7` compute bands using history sample standard deviation $\sigma_{\text{hist}}(x, y)$, NOT the power-law exponent. `sub7` validates input-derived local variance, not the universality of $Re^{0.678}$. | **Correction Applied** |
| **Hypo 12 (Persistent Zeros)** | "$M_{\text{extra}}$ is PIV shadow, $M_{\text{both}}$ is true airfoil" | $M_{\text{extra}} = M_{\text{hist}} \setminus M_{\text{sim}}$ represents **persistent exact-zero Real pixels outside the nominal Sim mask**. Its physical cause is unverified (could be support structures, FOV limits, optical occlusion, or preprocessing sentinels). Variable renamed to `persistent_zero_mask`. | **Correction Applied** |
| **Zero Guardrail Safety** | "Zero window regressions universally" | In an initial 40-window exploratory audit, no RelL2 regressions were observed. However, on the 80-window dev holdout, applying the guardrail degraded MVPE on $5\%$ of windows ($4/80$), though aggregate score improved. | **Correction Applied** |
| **Zero-Mean Invariance** | "Permanently guarantees 0.00% MVPE degradation across pipeline" | Algebraic invariance $\frac{1}{T}\sum \mathbf{r}'_t \equiv 0$ holds **strictly for the additive residual correction before downstream masking**. When post-processed by `final_uv * active_mask`, the temporal mean is altered on masked pixels. | **Correction Applied** |
| **Ordinary vs. Zero-Mean** | "Decisive counterfactual proof that Ordinary overfits at $\lambda \ge 0.25$" | Ordinary was evaluated with 1 seed (seed 42), while Zero-Mean was evaluated as a 2-seed ensemble (seeds 42 & 43). This confounds architecture with ensembling. The deployed 2-seed ZeroMean ensemble outperforms the tested 1-seed Ordinary model, but architectural superiority across equal ensembles is unproven. | **Correction Applied** |
| **Pareto Knee at $\lambda=0.30$** | "True Pareto optimum reached at $\lambda=0.30$" | The dev curve flattened substantially between $\lambda=0.25$ and $\lambda=0.30$ ($+0.010$ proxy gain). $\lambda=0.30$ was selected as an **engineering stopping point**, not a mathematically proven global optimum. | **Correction Applied** |
| **Local Final Score** | "Calibrated Final Score: 81.017 / 81.430" | Starting-kit scorer does not publish the official final combination formula. Local weighted sums are strictly **Offline Development Composite Proxies**, not official calibrated final scores. | **Correction Applied** |

---

## 3. The Validated Technical Pipeline (sub7)

### A. Architecture Specification
1. **Frozen CNO Backbone**:
   - Organizer's pre-trained Continuous Neural Operator checkpoint (`sim_real_cno.pth`, $32.1\text{ MB}$).
   - Forward pass in batches of 16 directly on GPU.
2. **Zero-Mean Residual Head (`ZeroMeanResidualHead`)**:
   - $55,528$ trainable parameters per head (3 Conv2D layers, width 32, GroupNorm, GELU).
   - Zero-initialized final layer ensures identity transformation at initialization.
   - Dual-seed ensemble: Seed 42 (best epoch 10) + Seed 43 (best epoch 12), trained with $\lambda_{\text{TKE}} = 0.30$.
   - **Mean-Centering Operation**:
     ```python
     out = self.net(x).reshape(-1, 20, 2, 32, 64).permute(0, 1, 3, 4, 2)
     return out - out.mean(dim=1, keepdim=True)
     ```
   - Features ($120$ channels): Concatenation of normalized input history $X_{0:19}$, CNO base forecast, and damped stationary prior $\mathbf{u}_{\text{prior}}(t) = \bar{\mathbf{u}} + 0.9^t (\mathbf{u}_{19} - \bar{\mathbf{u}})$.
3. **Persistent-Zero Guardrail**:
   - Derived strictly from the 20 observation frames:
     ```python
     persistent_zero_mask = torch.all(uv_in == 0.0, dim=(1, 4))  # persistent exact-zero history
     active_mask = (~persistent_zero_mask)[:, None, :, :, None].float()
     final_uv = (base_uv + corr * residual_std) * active_mask
     ```
4. **GPU-Vectorized Adaptive Spatio-Temporal SPS Band**:
   - Computed without Python loops or CPU transfers:
     ```python
     std_uv = torch.std(uv_in, dim=1, keepdim=True)  # (N, 1, 32, 64, 2)
     hw_raw = base_hw + 0.85 * std_uv * h_factors    # base_hw: [0.0075, 0.0032]
     hw_uv = torch.clamp(hw_raw, min=base_hw, max=0.040)
     lower = prediction - hw
     upper = prediction + hw
     ```

### B. Hardware Latency & Resource Footprint
Measured on NVIDIA GeForce RTX 3060 (Local Benchmark):
- **Median Inference Latency**: $146.74\text{ ms / sample}$ (CNO baseline: $141.99\text{ ms / sample}$).
- **Overhead**: $+4.75\text{ ms / sample}$ ($~3.3\%$).
- **Codabench Server Result**: Time Score = **$87.277381$** (retains full GPU throughput parity with baseline CNO $87.472$).
- **Peak GPU VRAM**: Peak allocated VRAM was $1.61\text{ GB}$ in the local benchmark.

---

## 4. Development Holdout (80 Windows) vs. Hidden Leaderboard

### A. Nature of the 80-Window Split
The 80-window holdout (covering Re 6306, 13977, 24204 across 14 trajectories) was used iteratively across research phases to select model types, tune $\lambda$, choose $0.30$ over $0.25$, and evaluate guardrails.
- **Classification**: It is an **offline development holdout (dev holdout)**, NOT an uncompromised test set.
- **Optimism Bias**: Offline composite proxies on this split scored $\sim 81.0\text{--}81.4$, whereas the official hidden leaderboard scored $\sim 78.6\text{--}79.2$.

### B. Detailed Offline Ablations (Separated by Objective)

#### Table 1: Point Predictor Ablation (80 Windows Dev Split)
*Evaluates point prediction accuracy before uncertainty band generation.*

| Configuration | Dev RelL2 Err ↓ | Dev TKE Err ↓ | Dev MVPE Err ↓ | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **CNO Baseline** | 0.08741 | 0.64915 | 0.08212 | Organizer baseline checkpoint |
| **CNO + Guardrail** | 0.08715 | 0.64896 | 0.08200 | Clamps persistent zero pixels |
| **Ordinary Head ($\lambda=0.20$, 3 seeds)** | 0.08614 | 0.59630 | 0.08153 | Standard residual adapter |
| **Zero-Mean Head ($\lambda=0.20$, 3 seeds)** | 0.08616 | 0.59772 | 0.08212 | Invariant residual adapter |
| **Ordinary Head ($\lambda=0.30$, 1 seed)** | 0.08722 | 0.58688 | 0.08151 | Higher TKE loss penalty |
| **Zero-Mean Head ($\lambda=0.30$, 2 seeds)** | 0.08649 | 0.58084 | 0.08212 | Balanced TKE vs mean retention |
| **Zero-Mean ($\lambda=0.30$) + GR** | **0.08623** | **0.58061** | **0.08200** | Point predictor deployed in `sub6` and `sub7` |

#### Table 2: Uncertainty / SPS Ablation (80 Windows Dev Split)
*Evaluates interval scoring parameter (SPS) across band policies.*

| Uncertainty Policy | Underlying Point Predictor | Dev SPS Score ↑ | Causal Mechanism |
| :--- | :--- | :---: | :--- |
| **Constant Band ($0.85 \times \text{RMS}$)** | CNO Baseline | 44.3725 | Static heuristic baseline |
| **Constant Band ($0.85 \times \text{RMS}$)** | Zero-Mean + GR | 45.0858 | *Interval rule is constant-width, but SPS changes because prediction center shifts residual coverage.* |
| **GPU Adaptive Band ($\alpha \cdot \sigma_{\text{hist}}$)** | Zero-Mean + GR | **47.8390** | Input-conditioned local standard deviation dynamically scales half-width. |

#### Table 3: Deployed Pipeline Configurations (Composite Development Proxy)
*Shows aggregate configuration benchmarks.*

| Deployed Configuration | Dev RelL2 Err ↓ | Dev TKE Err ↓ | Dev MVPE Err ↓ | Dev SPS Score ↑ | Dev Time Proxy | Dev Composite Proxy ↑ | Hidden Official Final ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CNO Baseline (`sub1`)** | 0.08741 | 0.64915 | 0.08212 | 44.3725 | 69.1009 | 80.4068 | 78.421963 |
| **ZeroMean + GR + Constant SPS (`sub6`)** | **0.08623** | **0.58061** | **0.08200** | 45.0858 | 68.9687 | 81.0169 | 78.639483 |
| **ZeroMean + GR + Adaptive SPS (`sub7`)** | **0.08623** | **0.58061** | **0.08200** | **47.8390** | 68.9687 | **81.4299** | **79.158181** |

---

## 5. Artifact Verification & Deployment Package (`sub7`)

- **ZIP Location**: [`CCCCCC/submissions/sub7_cno_zeromean_head_adaptive_sps.zip`](file:///d:/Project/NeurIPS/CCCCCC/submissions/sub7_cno_zeromean_head_adaptive_sps.zip)
- **Archive Size**: $28.68\text{ MB}$ ($30,070,489$ bytes, 63 files at zip root).
- **Exact SHA-256 Checksum**:
  `2e42b16da2eba7e794d90579003e31c9417d3e720f7babdb932bd376f810762a`
- **Starting-Kit Clean-Room Validation**:
  - Module import with simulated eval-container blocker (`scipy`, `h5py`, `matplotlib`, `pandas` blocked via `sys.meta_path`): **PASSED**.
  - Bounds contract: `lower <= prediction <= upper` holds everywhere on finite floats: **PASSED**.
  - Persistent zero enforcement: Predictions on exact-zero history pixels evaluate strictly to `0.0`: **PASSED**.

---

## 6. Project Status & Next Steps (Handoff Guidelines)

1. **Incumbent Locked**: `sub7` is established as the project's official state-of-the-art benchmark (**`79.158181`**).
2. **Validated Capabilities**:
   - **The deployed ZeroMean + TKE-targeted residual pipeline transferred a substantial TKE improvement to hidden data** ($73.72 \to 74.82$).
   - **Temporal mean centering algebraically preserves the base forecast-window temporal mean before downstream guardrail masking.**
   - GPU-vectorized adaptive SPS reliably expands uncertainty scores on hidden test data ($35.15 \to 37.58$).
3. **Open Research Questions for Potential `sub8`**:
   - **Fair Architecture Ablation**: Train Ordinary Head seed 43 under the exact same $\lambda=0.30$ protocol to establish an equal 2-seed vs 2-seed comparison.
   - **Boundary Localization Diagnostics**: Analyze where SPS coverage errors concentrate (near wake vs freestream) to evaluate whether bounds width should vary conditionally on space.
4. **Low Priority Under Current Evidence and Compute Budget**:
   - Rigid causal transport on top of CNO and global scalar scaling are well falsified within the tested model families.
   - Unconstrained large replacement backbones are deprioritized as unpromising under our strict compute budget compared to targeted adapter refinements.
