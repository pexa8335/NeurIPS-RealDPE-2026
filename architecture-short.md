# RealPDE Track 1 — Architecture & Strategy Summary

## Current incumbent

**Official best submission:** `sub7_cno_zeromean_head_adaptive_sps.zip`

**Official hidden leaderboard score:** **79.158181**

Pipeline:

$$
\boxed{
\text{Frozen CNO}
+
\text{2-seed ZeroMean residual head }(\lambda_{\mathrm{TKE}}=0.30)
+
\text{persistent-zero postprocessing}
+
\text{GPU adaptive SPS}
}
$$

Official hidden metrics:

* RelL2: 94.551736
* TKE: 74.823387
* MVPE: 93.487926
* SPS: 37.579004
* Time: 87.277381

Gain:

* vs `sub1` baseline 78.421963: **+0.736218**
* vs previous best `sub5` 79.100273: **+0.057908**

---

## What is established

### Hidden-established

**1. The deployed TKE-specialized point-prediction pipeline generalizes positively.**

Compared with the original CNO baseline, the deployed `sub6/sub7` point predictor improved hidden TKE:

$$
73.7175 \rightarrow 74.8234
$$

while RelL2 and MVPE remained comparable.

This validates the **deployed combination** of residual adaptation, ZeroMean projection, TKE-targeted training and postprocessing. Hidden results do not isolate the individual causal contribution of each component.

**2. Adaptive SPS is cleanly hidden-validated.**

`sub6` and `sub7` use identical point predictions. The uncertainty rule is the relevant change:

$$
SPS:\ 35.5132 \rightarrow 37.5790
$$

$$
Final:\ 78.6395 \rightarrow 79.1582
$$

with identical RelL2, TKE and MVPE.

Therefore input-conditioned adaptive bounds have strong hidden-set evidence.

**3. ZeroMean projection has an exact algebraic property.**

For residual correction

$$
r'_t=r_t-\frac1T\sum_{\tau=1}^{T}r_\tau,
$$

we have

$$
\frac1T\sum_t r'_t=0.
$$

Therefore the additive residual head cannot change the base forecast-window temporal mean **before downstream postprocessing**.

A subsequent static spatial mask can still change the final mean at masked locations.

---

## What is supported offline

**Small residual heads improve frozen CNO.**

Offline matched experiments showed modest improvements in field and TKE error relative to the frozen CNO.

**The tested rigid transport prior provides no incremental benefit on top of CNO.**

Transport improved a weaker Real-history model, but the tested CNO residual head performed better without the transport prior.

This does not establish that every possible transport architecture is useless.

**Global scalar amplitude correction is insufficient.**

Even target-informed scalar rescaling explains only a small fraction of fluctuation error. Matching total fluctuation energy can worsen field-level fluctuation error.

The remaining mismatch may involve spatial localization, phase, shape, temporal decorrelation or other factors; its mechanism is not established.

**Persistent-zero masking provides a small development-set gain.**

Define

$$
M_{\mathrm{zero}}(x,y)
=
\mathbf 1[\text{u and v are exactly zero at this pixel throughout the 20-frame history}],
$$

$$
M_{\mathrm{active}}=1-M_{\mathrm{zero}},
$$

and apply

$$
\hat{\mathbf u}_{final}
=
\hat{\mathbf u}_{raw}\odot M_{\mathrm{active}}.
$$

The physical cause of these persistent-zero pixels is not established, and the guardrail's independent hidden contribution has not been isolated.

---

## What is not established

Do not treat the following as facts:

* ZeroMean is universally superior to an equally ensembled Ordinary residual head.
* The guardrail independently improves hidden leaderboard score.
* Persistent-zero pixels outside the Sim mask are necessarily optical shadows or physical solid geometry.
* Remaining CNO TKE error is specifically phase error.
* The empirical relation

$$
WakeRMS \approx C Re^{0.678}
$$

is a universal physical law.

* Hidden SPS improvement is specifically caused by high-Re cases.
* \(\lambda_{\mathrm{TKE}}=0.30\) is the globally optimal value.
* Arbitrary noise injection or all possible transport-based approaches are universally ineffective.

---

## Strategy

Keep the current backbone strategy:

$$
\boxed{\text{Frozen CNO + lightweight residual adaptation}}
$$

Prioritize:

1. better fluctuation/TKE prediction without sacrificing field accuracy;
2. improved uncertainty calibration;
3. clean ablations that isolate proposed additions.

Under the current evidence and compute budget, treat large rigid-transport architectures, blanket amplitude scaling and arbitrary noise injection as **low priority**, not universally falsified.

For a rigorous ZeroMean-vs-Ordinary architectural claim, the missing control is a seed-matched/ensemble-matched Ordinary \(\lambda=0.30\) comparison.

For leaderboard progress, do not run that control unless it also informs a plausible `sub8` improvement.
