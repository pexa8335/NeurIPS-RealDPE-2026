# Residual Predictability & Oracle Decomposition Analysis

**Date**: 2026-09-10  
**Context**: CNO Fluctuation Bottleneck & Predictability Ceiling Investigation  
**Dataset**: 80 Development Test Windows (Re 6306, 13977, 24204; 14 Trajectories)  
**Incumbent Reference**: CNO Point Predictions vs Ground Truth Fluctuation ($y' = y - \bar{y}$)

---

## 1. Executive Summary & Core Scientific Answers

| Question | Empirical Finding | Strategic Implication |
| :--- | :--- | :--- |
| **1. Is CNO failing due to spatial alignment / advection shift?** | **NO.** Rigid spatial shift search ($(\Delta x, \Delta y) \in [-4, 4]^2$) recovers only **0.68%** of fluctuation squared error. | **Do NOT build spatial warp/advection networks.** Spatial misplacement is negligible. |
| **2. Is CNO failing due to temporal phase shift / lag?** | **NO.** Temporal phase lag search ($\tau \in [-5, 5]$ frames) recovers only **0.22%** of fluctuation squared error. CNO's temporal projection onto the dominant physical POD shedding mode has $R^2 = \mathbf{88.0\%}$. | **Do NOT build phase-shifting networks.** CNO already captures the dominant vortex shedding frequency and phase. |
| **3. Is CNO failing due to localized energy damping?** | **YES.** CNO predicts only **49.8%** of the true fluctuation energy. A local spatial amplitude map $A(x,y)$ recovers **25.48%** of fluctuation squared error and slashes TKE RelL2 error from `0.6491` to `0.4897` (**-24.6%**). | The primary deterministic modeling ceiling lies in **localized wake-energy modulation**, not phase or advection. |
| **4. What is the theoretical coherent structure ceiling?** | Ground truth fluctuations are highly coherent: Top-1 POD mode captures **46.1%**, Top-3 capture **82.5%**, and Top-5 capture **93.9%** of total energy. Perfect top-3 mode reconstruction recovers **70.69%** of error (TKE error `0.1686`). | The gap is real coherent physics, not white observation noise. |
| **5. Can fine future fluctuations be predicted linearly from 20-frame history?** | **NO.** Causal linear ridge regression from 20-frame history PCA to future POD modes fails on held-out test (MSE increases by 57%). | Instantaneous turbulent details rapidly decorrelate. A deterministic model cannot predict fine-scale trajectory realization at $t+20$. |

---

## 2. Fluctuation vs Mean Error Breakdown

Total error of CNO across the 80 development test windows:
$$\frac{1}{T}\sum_{t=1}^{20} \| \hat{y}_t - y_t \|^2 = \| \bar{\hat{y}} - \bar{y} \|^2 + \frac{1}{T}\sum_{t=1}^{20} \| \hat{y}'_t - y'_t \|^2$$

- **Total MSE**: `0.00013227` (100.0%)
- **Temporal Mean MSE**: `0.00004158` (**31.43%**)
- **Fluctuation MSE**: `0.00009069` (**68.57%**)
- **Fluctuation Energy Ratio (CNO / Target)**: `0.00007307 / 0.00014667` = **0.4982**
- **Baseline CNO TKE RelL2 Error**: `0.649145`

*Finding*: CNO already fits the temporal mean very well (RelL2 score ~94.55, MVPE ~93.49). Over two-thirds (68.57%) of the remaining squared error resides strictly in the temporal fluctuation field. Furthermore, CNO severely underestimates fluctuation energy by roughly a factor of 2.

---

## 3. The Master Oracle Decomposition Table

Each transformation is evaluated as an oracle on the 80 test windows to determine the **maximum recoverable error** under that specific physical mechanism:

| # | Transformation Family | Degrees of Freedom | Recovered Fluctuation MSE % | Remaining Fluctuation MSE | Resulting TKE RelL2 Error | Physical / Modeling Interpretation |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **0** | **Baseline CNO** | 0 | **0.00%** | `0.00009069` | **`0.649145`** | Unmodified CNO point predictions |
| **1** | **Global Amplitude Oracle** | 1 scalar $\alpha$ / window | **2.36%** | `0.00008855` | `0.735010` | Uniform scalar inflation overshoots outside wake; degrades TKE error |
| **2** | **Temporal Phase Shift Oracle** | 1 discrete lag $\tau \in [-5, 5]$ / window | **0.22%** | `0.00009049` | `0.649145` | Phase shift explains virtually zero error; shedding phase is already captured |
| **3** | **Spatial Alignment Oracle** | 2 discrete shifts $(\Delta x, \Delta y) \in [-4, 4]^2$ | **0.68%** | `0.00009007` | `0.650011` | Rigid spatial advection shift explains $<0.7\%$ error |
| **4** | **Local Spatial Amplitude Map** | $32 \times 64 \times 2 = 4,096$ spatial weights / window | **25.48%** | `0.00006758` | **`0.489723`** | **Major bottleneck**: spatially non-uniform energy damping in the wake shear layer |
| **5** | **Temporal Lag + Local Spatial Amplitude** | 1 lag + 4,096 weights / window | **25.93%** | `0.00006717` | `0.488753` | Adding temporal lag to spatial amplitude adds only **+0.45%** |
| **6** | **Spatial Shift + Local Spatial Amplitude** | 2 shifts + 4,096 weights / window | **26.33%** | `0.00006681` | `0.487773` | Adding spatial shift to spatial amplitude adds only **+0.85%** |
| **7** | **Joint Full Alignment (Lag + Shift + Amplitude)** | 3 search params + 4,096 weights | **26.38%** | `0.00006676` | `0.486462` | Full space-time alignment ceiling is capped at **26.4%** |
| **8** | **Theoretical Coherent Structure Ceiling** | Top-3 POD modes of future target | **70.69%** | `0.00002658` | **`0.168640`** | Theoretical ceiling if all 3 dominant coherent structures were known |
| **9** | **Causal Linear History Predictor** | 32 PCA $\to$ 16 POD linear projection | **-57.03%** (Overfit/Fail) | `0.00014241` | `0.962985` | Linear history-to-future projection cannot track instantaneous turbulent realization |

---

## 4. Proper Orthogonal Decomposition (POD) Insights

Performing singular value decomposition (SVD/POD) on the true future fluctuation matrices $Y_{\text{fluc}} \in \mathbb{R}^{20 \times 4096}$ reveals:
1. **Dominant Energy Concentration**:
   - **Mode 1**: Captures **46.1%** of total fluctuation energy.
   - **Modes 1–3**: Capture **82.5%** of total fluctuation energy.
   - **Modes 1–5**: Capture **93.9%** of total fluctuation energy.
2. **CNO's Coherent Mode Fidelity**:
   - Projecting CNO's predicted fluctuation onto the true dominant spatial POD mode yields an average temporal correlation $R^2$ of **88.0%**.
   - This proves that CNO is **not** producing random noise or wildly wrong frequencies; it correctly identifies the dominant vortex shedding mode shape and temporal frequency.
   - However, CNO damps the amplitude of this mode (underpredicts variance by ~50%).

---

## 5. Strategic Takeaways: Where Does a Real Breakthrough Lie?

1. **What to Abandon**:
   - **Do not pursue spatial warp / advection alignment**: Recovers $<0.7\%$.
   - **Do not pursue phase alignment**: Recovers $<0.25\%$.
   - **Do not pursue scalar global amplitude boosting**: Overshoots calm fluid regions, increasing TKE error from 0.649 to 0.735.

2. **Where the Recoverable Deterministic Energy Resides**:
   - **Localized wake-envelope modulation**: Recovers **25.5%** of fluctuation squared error and reduces TKE error to `0.489`.
   - The current ZeroMean residual head ($\lambda = 0.30$) operates precisely in this family (a localized convolutional residual adder), but its 55k-parameter capacity only partially captured this potential.

3. **The Predictability Ceiling**:
   - The remaining **~70–75%** of fluctuation error is not reachable by simple coordinate transforms or linear history models.
   - Instantaneous fine-scale turbulent eddies at $t=20$ are weakly correlated with $t=-20$.
   - Therefore, pushing the overall score significantly past ~80 requires:
     (a) Non-linear wake-localized energy injection (generative / spectral regularized modeling of the active wake), OR
     (b) Recognizing the fundamental aleatoric uncertainty and maximizing the SPS scoring geometry (which is why SPS narrowing to 0.70 gave +0.15 free points).
