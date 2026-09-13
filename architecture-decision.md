# RealPDE Track 1: architecture decision and evidence audit

## 2026-09-10: Hypothesis 01 recheck — grid / observed zero-mask / geometry

Executed a read-only full-archive audit: 82 Real +100 Sim HDF5, actual Re/AoA metadata, all frames, protocol fixed before measurement. [Detailed report](hypothesis/hypo_01_audit.md), [per-file evidence](research/hypothesis01_audit/per_file.csv), [verification](research/hypothesis01_audit/verification.json).

* **Grid:** all `(64,128)` stored coordinate grids and UV spatial shapes verified. Sim–Real max coordinate differences x=1.0832e-05, y=1.9936e-05. Original1e-5 hypothesis fails; historical extension1e-4 passes. Report both; no post-hoc tolerance relaxation. This supports approximately shared stored grids, not exact invariance or unseen physical-motion claims.
* **Masks:** distinguish exact zero, temporal constancy and approximate zero at predeclared1e-12/1e-10/1e-8. Full-trajectory persistent intersection is static by construction and does not establish instantaneous mask stationarity. Full/LCC mask comparisons across actual Re at fixed AoA and history20→future20 activation rates are recorded for all files; do not conflate zero-mask variation with physical geometry changes.
* **Measured scope:** all82 Real files have changing instantaneous exact-zero sets; none of100 Sim files do. Across fixed-AoA/Re pairs, Sim exact/1e-10 masks match, while Real masks vary. On1693 Real windows at stride2, 25,045/269,572 history-zero sample-pixels (9.2907%) become nonzero in future. Five Sim-case precision checks show7–15 false constant pixels from float32 std; float64 agrees with direct temporal constancy.
* **Geometry:** no explicit mask-like top-level dataset names found. Connectivity/PCA and Real–Sim overlap do not independently identify solid/airfoil/optical shadow. In nominal10125 examples (Real actual10142; Sim10125), largest-component IoU at AoA10/15 is0 at1e-10; largest Real component is not a reliable default geometry correspondence.
* **Decision:** retain sub7 and its input-history `persistent_zero_mask` with the Phase1 measured trade-offs. Hypo1 does not justify masking all losses/point metrics, adding geometry conditioning or asserting a shadow mechanism. G0 training-provenance blocker is unchanged; no subsequent gate started. Original notebook and incumbent ZIP remain hash-identical.

---

## 2026-09-10: Phase-1 audit — G0/G1 only

**G0: BLOCKED for a certified matched-learning table. G1 diagnostics completed. Sub7 preserved.** Fresh research/package parity is trustworthy at identical batch/precision, but legacy caches are not numerically interchangeable with batch-16 deployment. Seven local cohort checkpoints exist and reproduce their saved predictions exactly; their full historical training-protocol equivalence cannot all be certified from surviving records. No training, tuning, Kaggle job, submission, architecture change or incumbent repackaging was performed.

Authority: `architecture-short.md` > Protocol v2/latest audit here > historical `walkthrough.md`. The reported hidden final **79.158181** remains external, user-supplied evidence; this audit does not independently verify Codabench. All new evidence is under [research/phase1_audit](research/phase1_audit). Run scripts there only for diagnosis; never run the legacy training entrypoints as an audit.

### 1. G0 result

#### Artifact/provenance source of truth

| Immutable submission ZIP | Bytes | SHA-256 |
|---|---:|---|
| `sub1_cno_sps.zip` | 29,653,891 | `51620e1a3256df389cf83a9310ff84660b3b20560c224365d04404ff3711bf9a` |
| `sub5_cno_head_adaptive_sps.zip` | 30,070,239 | `833cb41267ae6da2788fcb7ccac3e5379c9e76bf38a25d37ce9f79fda716dc57` |
| `sub6_cno_zeromean_head_sps.zip` | 30,070,284 | `5b4a6d597013dfecff5275a4179511c4cf1789e36e6ea4f4338de0e5d7afaaa3` |
| `sub7_cno_zeromean_head_adaptive_sps.zip` | 30,070,489 | `2e42b16da2eba7e794d90579003e31c9417d3e720f7babdb932bd376f810762a` |

[Submission inventory](research/phase1_audit/submissions.json) records hashes of every ZIP member, including source and checkpoint bytes. Sub7's before/after ZIP hash is unchanged. Its exact extraction used for inference is `research/phase1_audit/exact_sub7/`; original extraction and ZIP were not edited.

| Artifact family / actual source | Established architecture, seed, lambda | Normalization / features | Training and selection evidence | Reuse scope |
|---|---|---|---|---|
| `CCCCCC/submissions/sub1_cno_sps/sim_real_cno.pth`; identical CNO bytes in sub5/6/7 | CNO, 7,960,467 parameters; seed/lambda unknown | Official UV input/output constants from submission; zero third channel; 20→20, 32×64 | iteration=best_iteration=5000; best_val_loss stored. Original split, optimizer, epochs, scheduler, selection rule and training source hash unknown. Nearby `cno_baseline/config.json` describes a separate proposed run and is not evidence for this organizer checkpoint | yes as immutable frozen incumbent; not evidence of fold-pure backbone training |
| sub5 `cno_head_weights.pth` | Ordinary, seeds 42/43, lambda=.1 | 120-channel history+CNO+damped-prior, train-fitted mean/std/residual RMS | tensors and normalizers exactly match `kaggle_output/tke_test/loss_screen/primary_lambda_0.1_seed42/43.pth`; archived config and training logs available | yes for its original deployment/protocol; not local proxy-selected cohort |
| sub6/sub7 `cno_head_weights.pth` | ZeroMean, seeds 42/43, lambda=.30; best epochs 10/12 | same feature family, normalizers exactly match local S checkpoints | state tensors exactly match `Model_S_ZeroMean_L0.30_seed42/43.pth`; historical complete training config/source binding unknown | yes for inference and G1; causal reuse not yet certified |
| local `research/exp_zero_mean_head/artifacts/checkpoints/*.pth` | Ordinary and ZeroMean; 55,528 parameters each; individual seeds/lambdas/best epochs recorded | per-checkpoint numeric normalizers and tensor shapes recorded; seven target cohort predictions replay exactly at batch 12 | current source describes 12 epochs, batch 12, AdamW lr=.001/wd=.0001, no scheduler, seeded NumPy permutation, zero final convolution initialization, validation offline development composite proxy selection including epoch 0. It executes only O.25/O.30 seed42 now; cannot establish every earlier run's actual ordering/scheduler/selection from that mutable source | preserve and reuse predictions; historical protocol linkage unresolved |
| older `kaggle_output/tke_test/loss_screen/primary_lambda_*.pth` | Ordinary; lambdas 0/.1/.3/1, seeds 42/43 | same feature family; numeric normalizers stored separately | embedded config + archived source/logs: 12 epochs, batch12, AdamW .001/.0001, no scheduler, sorted files and `default_rng(seed).permutation(train)`; **minimum validation field RelL2 including epoch0**, not proxy selection | reusable under older native protocol; not a matched replacement for local O3 |

The [additional archived architecture-screen inventory](research/phase1_audit/architecture_screen_inventory.json) records eight earlier residual heads, including Ordinary `champion_zero_transport_seed42/43.pth` (lambda0 field-only source). Their native validation-field selection is not the local proxy-selected policy; history-mean/transport variants additionally change the base or features. They are preserved, not silently mixed into the 2×2 table.

The exhaustive [51-checkpoint provenance table](research/phase1_audit/artifact_provenance.csv) contains paths, hashes, architecture/seed/lambda, normalizers, optimizer, epoch policy, best epoch, data order, source-hash availability and reuse limitations. [Raw metadata](research/phase1_audit/checkpoints.json) preserves the exact stored fields; [packaged tensor matches](research/phase1_audit/packaged_head_matches.json) establishes bundle lineage. Current source fingerprints are audit-time hashes, **not** training-time attestations. Missing information is unknown, not inferred from a neighboring config or filename alone.

#### Sample identity and overlap

All **477 windows** were verified by exact input/target equality against the trajectory cache and its `t_starts`, actual `re` and `aoa`. Train: **339 windows/57 trajectories**; validation: **58/10**; development test: **80/14**. Test Re=(6306, 13977, 24204); validation Re=(10142, 20369). No trajectory crosses folds and no raw-frame interval overlaps across folds. Each window records `[start,start+20)` history and `[start+20,start+40)` future in the [explicit-ID manifest](research/phase1_audit/sample_manifest.csv). IDs are `file:start:20:20`, not positional labels. Filename nominal Re differs from measured Re; use the latter.

Legacy NPZ files **do not store IDs**, so the historical storage contract was positional. This audit did not assume alignment: it checked cache contents row by row, joined old/new manifests on ID, verified both generations of saved targets exactly, and replayed all seven local target-cohort checkpoints **bit-for-bit** against saved UV predictions at original head batch12. New deployed/OFF prediction files embed IDs, and the sidecar binds legacy arrays to verified identities. This certifies the audited alignment, not every unrelated project tensor. Cached data were checked; all original HDF5 files were not re-read in this phase. Backbone exposure to held conditions remains unknown.

**Hypo 13/14 scope = A, diagnostic labels only.** 78/80 hard-coded Re labels are wrong. Actual old-array order: rows 0–29 Re13977; 30–49 Re24204; 50–79 Re6306. Labels are assigned after prediction/target loading and are not input to training or inference. Corrected [Hypo13](research/phase1_audit/hypo13_corrected_per_Re.csv), [Hypo14](research/phase1_audit/hypo14_corrected_per_Re.csv) and per-window identity tables supersede those diagnostic groupings; original notebooks are preserved. No evidence of contamination of aggregate prediction/target alignment was found.

#### Research versus exact packaged inference

**Protocol conflict:** v2 refers to a predefined engineering tolerance but gives no numerical float32 parity tolerance. Before measurement, [protocol.md](research/phase1_audit/protocol.md) locked `atol=1e-6, rtol=1e-5`; raw metric absolute tolerance `1e-6`; algebra mean tolerance `1e-6`. Relative maxima exclude reference magnitudes below `1e-3`. These are an explicit Phase-1 engineering addendum, not a tolerance claimed to have existed in v2.

All 80 windows, float32, RTX2050, PyTorch2.11.0+cu128:

| Comparator | Max absolute output difference | Max relative (reference ≥1e-3) | RelL2 / TKE / MVPE aggregate raw difference | Outcome |
|---|---:|---:|---|---|
| Cached research base/features + research Z3 heads, versus package batch16 | 1.975894e-4 | .0713040 | +1.34110e-7 / −1.31726e-5 / +1.55717e-6 | FAIL; not interchangeable |
| Fresh independent research head implementation and live base, batch16, versus exact package | 2.980232e-8 | 6.376534e-6 | −1.86265e-10 / −2.60770e-9 / +3.49246e-10 | PASS, unchanged tolerance |
| Fresh CNO batch1 versus cached CNO | 0 | 0 | base-only identity | exact |
| Fresh CNO batch1 versus batch16 | 1.934618e-4 | .1017228 | not a scorer comparison | batch-dependent numerics localized |

[Investigation](research/phase1_audit/parity_investigation.json) identifies inference batch as sufficient to reproduce the cache discrepancy. cuDNN TF32 is enabled in this environment; its causal contribution was not separately toggled or claimed. Same-batch research/package agreement is strong, but **do not use old cached predictions as exact deployed batch16 outputs**. Legacy `benchmark_submission.py` also hardcodes lambda=.25; it is not the deployed .30 reference. The audit used the verified .30 checkpoints explicitly, without editing that historical script.

#### Scorer audit

Three local scorer copies are byte-identical, SHA-256 `a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39`. Reference: `realpde_t1_starting_kit_v9/realpde_t1_starting_kit_v9/scoring.py`.

* **RelL2:** per-window flattened measured-channel `||p−y||₂ / max(||y||₂,1e-8)`, then mean over windows.
* **TKE:** forecast-window population variance map `K=.5*(mean_t((u−mean_t u)²)+mean_t((v−mean_t v)²))`; spatial relative L2 with the same denominator floor, then window mean.
* **MVPE:** temporal-mean UV at probe rows `[8,10,...,24]` and columns `[13,21,29,37]` for this grid/default subsampling; relative vector L2 per column, mean across columns, then windows.
* **SPS:** only target elements `y!=0` enter interval average/coverage. For each window error `e_j`, accuracy factor `a_j=.5/(.5+e_j)`. Element contribution is `inside * exp(−width/.0563870259) * (.5*a_RelL2+.3*a_TKE+.2*a_MVPE)`; pool over scored elements; score clips aggregate to [0,1] and multiplies by100. Accuracy factors use the full field. Bounds are inclusive. Zero exclusion is a scorer rule, not proof of a physical region label.
* **Coverage:** number of covered nonzero-target elements divided by all nonzero-target elements. There is no prescribed 90%/95% target.
* **Error subscore:** `100/(1+.5*max(error,0))` for finite errors; nonfinite→0.
* **Time subscore:** `100/(1+sqrt(t_neural/.72896))` for positive finite time (default r_min=1), otherwise0. Audit interval timing is not official end-to-end time.
* **Final aggregation: unpublished in the kit.** The historical weighted final expression is an **offline development composite proxy**, not official/calibrated final score. SPS internal weights do not reveal final-score weights.

Saved scorer recomputation matches 15/16 local ablation rows to rounding. The sole mismatch is **ZeroMean lambda=.20**: CSV still reports two-seed TKE .599143863 while saved ensemble UV is the three-seed .597724140. Source's seed44 update replaces the ensemble then `continue`s without refreshing its summary row. That ensemble also has four channels from repeated `as3` padding; UV matches the three-seed mean exactly and this scorer reads UV only. This stale-summary/shape issue does **not** affect deployed lambda=.30. Original artifacts are preserved; [scorer_recomputation.csv](research/phase1_audit/scorer_recomputation.csv) and [scope record](research/phase1_audit/stale_summary_scope.json) document it.

### 2. Checkpoint reuse matrix

The intended local cohort is the proxy-selected family corresponding to deployed Z3. “Reusable” below means **certified for the fully matched causal table**, not usable for inference. Checkpoint prefix directory is `research/exp_zero_mean_head/artifacts/checkpoints/`.

| Cell | Seed | Checkpoint | Exists / best epoch | Protocol compatible? | Reusable? | Missing information / training still required? |
|---|---:|---|---|---|---|---|
| O0 | 42 | `Model_C_Ordinary_L0_seed42.pth` | yes; epoch 12 | unresolved | no, pending provenance | recover historical run config/order/scheduler/selection linkage; do not retrain by default |
| O0 | 43 | `Model_C_Ordinary_L0_seed43.pth` | yes; epoch 9 | unresolved | no, pending provenance | recover historical run config/order/scheduler/selection linkage; do not retrain by default |
| O3 | 42 | `Model_H_Ordinary_L0.30_seed42.pth` | yes; epoch 11 | unresolved | no, pending provenance | recover historical run config/order/scheduler/selection linkage; do not retrain by default |
| O3 | 43 | `Model_H_Ordinary_L0.30_seed43.pth` | no | no | no | one missing control under local proxy-selected cohort; older field-selected O3 exists but is incompatible |
| Z0 | 42 | `Model_D_ZeroMean_L0_seed42.pth` | yes; epoch 11 | unresolved | no, pending provenance | recover historical run config/order/scheduler/selection linkage; do not retrain by default |
| Z0 | 43 | `Model_D_ZeroMean_L0_seed43.pth` | yes; epoch 9 | unresolved | no, pending provenance | recover historical run config/order/scheduler/selection linkage; do not retrain by default |
| Z3 | 42 | `Model_S_ZeroMean_L0.30_seed42.pth` | yes; epoch 10 | unresolved | no, pending provenance | recover historical run config/order/scheduler/selection linkage; do not retrain by default |
| Z3 | 43 | `Model_S_ZeroMean_L0.30_seed43.pth` | yes; epoch 12 | unresolved | no, pending provenance | recover historical run config/order/scheduler/selection linkage; do not retrain by default |

All seven existing cohort checkpoints remain valid inference artifacts and replay exactly. Their capacity and numeric normalizers match; checkpoint metadata establish seed/lambda/best epoch and stored validation values. Missing run-bound configuration prevents asserting matching scheduler, sample ordering, total epoch policy and checkpoint-selection execution across the earlier runs. This is a recoverable provenance limitation, **not evidence that they were trained differently**, and not a recommendation to replace seven usable files.

`primary_lambda_0.3_seed43.pth` is a real existing Ordinary checkpoint, but uses minimum-field validation selection. `Model_F_ZeroMean_L0.3_seed42/43.pth` also exists and stores epoch0/val_scores rather than the deployed S-family proxy record; it is not silently substituted for trained Z3. The archived Ordinary O0/O3 checkpoints can be reused within their native field-selected protocol; mixing them into the local proxy-selected comparison would reintroduce the selection confound.

**One local cell is definitely absent: O3 seed43.** If the seven surviving runs' compatible historical records can be recovered, only that one new run is needed. If recovery fails, any future training count depends on a declared common protocol and which native cohorts it admits. There is no defensible exact sufficient count yet; do not turn uncertainty into an eight-run launch.

**Minimum number of new matched-learning runs required: N = unknown; confirmed lower bound 1 for the intended local cohort (best case 1, conservative upper bound 8).** No runs executed. Resolving the historical config/source/log linkage precedes any retraining decision.

### 3. G1 component audit

All results below use the **same exact packaged batch16 point tensor** and the verified 80-window development identities, except the explicitly labeled cached-path algebra comparison. The development holdout has been inspected repeatedly; these are descriptive offline results, not fresh confirmation.

**A1 — hypothesis → comparator → result → uncertainty → supported.** ZeroMean correction should preserve the base temporal mean before masking. On all samples/pixels/UV channels, cached-path physical correction mean maximum is **2.98023e-9**, additive temporal-mean shift **1.19209e-7**, both below1e-6. Fresh deployed-feature research reconstruction gives **1.49012e-9 / 8.94070e-8** respectively. After final persistent-zero masking, maximum base-mean shift is **.01785381**; masked output means are exactly zero. A static mask preserves zero-mean correction algebra but changes the final base mean at masked pixels. This is an implementation/algebra check, not physical conservation or independent hidden TKE causality.

**A2 — hypothesis → exact guardrail OFF/ON → result → trade-off → supported offline with regressions.** OFF=`base+unmasked centered correction`; ON applies the same input-defined static mask to the full predictor. All other inference operations and weights are identical. Reconstructed ON is bit-identical to package output.

| Raw error, lower better | OFF | ON | Mean paired ON−OFF | Window regressions |
|---|---:|---:|---:|---:|
| RelL2 | .086552896 | .086231917 | −.000321003 | 0/80 |
| TKE | .580969691 | .580614626 | −.000354976 | 1/80 |
| MVPE | .082122944 | .082005486 | −.000117449 | 4/80 |

There are **12,846 / 163,840 sample-pixels (7.840576%)** in `persistent_zero_mask`; **9.232446%** of those become nonzero in at least one future frame/channel. Mask-region UV MSE drops **5.60660e-6→5.87712e-7**; outside-region MSE is exactly unchanged at **1.409785e-4**. Every trajectory improves RelL2/MVPE on average; `13950_0.h5` regresses TKE by5.3167e-5. Regression rates are unchanged when requiring delta>1e-7. [Per-window](research/phase1_audit/mask_per_window.csv), [per-trajectory](research/phase1_audit/mask_per_trajectory.csv), and [per-horizon](research/phase1_audit/mask_per_horizon.csv) tables retain full deltas. Horizon TKE/MVPE are cumulative-prefix diagnostics; h=1 TKE is undefined and left blank, not reported as a successful zero. Single-frame RelL2 deltas are also included.

Historical saved head predictions already mask the **correction** before adding it to the base. That older “OFF” comparator differs from removal of the final mask on the full deployed predictor, explaining the slightly different historical OFF RelL2≈.08649. Both definitions are recorded; no physical labels are assigned. **Keep the guardrail in the incumbent on this offline evidence**, with transient future-nonzero and MVPE/TKE regressions acknowledged. No matched hidden mask-OFF result exists.

**A3 — hypothesis → constant sub6 vs adaptive sub7 on fixed predictions → result → width/runtime cost → supported.** Constant half-width=.85×.010925=.00928625. Adaptive half-width=`clip([.0075,.0032]+.85*std_history(ddof=1)*sqrt(h/10), min=[.0075,.0032], max=.040)`. Reconstructed adaptive lower/upper are bit-identical to package bounds.

| Policy | SPS score | Coverage | Mean width | Median width | Mean u/v width | GPU interval median [range], all80 windows |
|---|---:|---:|---:|---:|---|---|
| Constant | 45.085828 | 84.3750% | 0.01857250 | 0.01857250 | .01857250 / .01857250 | 0.335 ms [0.326, 0.740] |
| Adaptive | 47.839039 | 90.2639% | 0.02065258 | 0.01711030 | .02715893 / .01414623 | 2.632 ms [2.527, 2.726] |

RelL2/TKE/MVPE are identical by construction and explicit recomputation. Interval timing includes allocation/calculation only, one warmup and five synchronized repeats, excludes transfer/model/scoring. Mean width increases; median decreases because width redistributes. Every horizon width is in [intervals.json](research/phase1_audit/intervals.json). Audit width summaries use float64 accumulation (long float32 channel reductions were inaccurate and were corrected within new audit outputs only). The reported hidden sub6→sub7 SPS35.513169→37.579004 remains separate external evidence, without an invented Reynolds mechanism.

**A4 — hypothesis → fixed adaptive-rule residual group diagnostics → result → uncertainty → supported as a structured signal, not a tuned solution.** Coverage on scored elements is **u86.7540%, v93.8199%**; MAE **.00851273/.00300095**. History-std quartile edges were computed from **339 training histories only**, separately per channel with sample std(ddof1), before grouping dev data; no thresholds were fitted on the development holdout. u coverage by training-defined quartile is **94.7424%,94.0070%,86.2827%,74.2220%**; v is **98.8149%,96.6600%,90.9037%,91.0030%**. Overall horizon coverage goes **87.3768% at h1**, peaks around91.15% mid/late horizon, and ends **89.1401% at h20**. MAE increases from.00440771 to.00733431. Full [channel/horizon/std/region/Re/AoA/trajectory table](research/phase1_audit/calibration.csv) reports scored counts, coverage, MAE, width and contribution to SPS using full-window accuracy factors. Regions were fixed a priori as persistent-zero mask and complement. Re/AoA are descriptive groupings only.

This is a simple remaining channel/std-dependent calibration pattern that plausibly warrants **one** controlled UQ experiment after G0 closure. Coverage differences alone do not establish the optimal width, coverage target, or a guaranteed SPS gain. No modification, parameter range, bound tuning or candidate was designed here.

### 4. Decision (next gate not started)

1. **Is research/package parity trustworthy?** Yes for freshly recomputed lambda=.30 inference with matched batch16/float32. No for substituting legacy batch1 CNO caches as exact deployment outputs. Preserve the distinction and fixed tolerances.
2. **Keep persistent-zero guardrail?** Yes, retain immutable sub7: small aggregate offline gain with quantified regressions; independent hidden benefit remains unknown.
3. **One controlled UQ experiment warranted?** A4 provides a structured descriptive signal; yes in principle after G0 closure, without selecting/tuning a modification now.
4. **Which matched checkpoints are missing?** Local O3 seed43 is absent. Seven local candidates exist and reproduce exactly; historical full-protocol compatibility remains unverified. Older O3 seed43 is not the same selection protocol. Exact sufficient training count is blocked by provenance, not missing prediction logs alone.
5. **Next gate?** **Stop and retain sub7 for now because G0 is BLOCKED.** Recover/attest the existing cohort's historical protocol and lock batch handling first. Once those measurement/provenance conditions close, the evidence favors **G2-UQ before G3**, consistent with v2. No G2/G3/G4/G5 work was started.

Reproduction: `python research/phase1_audit/audit.py`, then `investigate_parity.py`, `provenance.py`, `supplement.py`, `replay_saved.py`, `finalize_tables.py`, `verify.py` from that directory's scripts (run from repository root). These are inference/analysis only. Diagnostic extraction does not repackage the incumbent. The report deliberately retains the missing-tolerance, stale benchmark, stale lambda=.20 summary, legacy positional IDs and incomplete training-provenance conflicts instead of silently reconciling them.

---

Initial audit: 2026-09-05. Follow-up experiments: 2026-09-06. This document contains all reports, recommendations, and experiment decisions. The original audit appears below the follow-up. Existing submissions, checkpoints, raw data, and the notebook are preserved; new experimental code and outputs are stored under `research/score_followup/`.

## 2026-09-09: đọc walkthrough sub1–sub7 và cập nhật bằng chứng

**Mốc hiện tại là `sub7_cno_zeromean_head_adaptive_sps.zip`, final 79.158181 do người dùng cung cấp.** Các đoạn ngày 06/09 bên dưới nói sub1 còn là incumbent hoặc chưa có submission mới là trạng thái lịch sử. Tôi đã đọc toàn bộ walkthrough, đối chiếu code/checkpoint và các output chính; chưa truy vấn độc lập bảng điểm Codabench, chưa chạy lại toàn bộ notebook và không khởi chạy thí nghiệm mới trong lượt này.

### Kế hoạch kiểm định tuần tự — protocol v2, chưa thực thi

**Mục tiêu:** tìm thay đổi có đủ bằng chứng để đáng thử thay `sub7`, đồng thời chỉ chạy causal ablation khi kết quả của nó có khả năng thay đổi quyết định kiến trúc.

Hai câu hỏi phải luôn tách riêng:

1. **Competition question:** cấu hình nào có khả năng tăng official leaderboard score?
2. **Scientific question:** thành phần nào thực sự gây ra thay đổi quan sát được?

Không yêu cầu mọi experiment phải trả lời cả hai câu hỏi.

---

## G0 — Khóa nền đo lường, provenance và khả năng reuse

Không train model ở bước này.

### A0. Artifact, prediction và metric có đang nói về cùng một đối tượng?

Khóa:

* `sub7` ZIP + SHA-256 hiện tại, không ghi đè.
* CNO checkpoint.
* ZeroMean head checkpoints seed 42/43.
* normalization, channel order, input/output shape và forecast horizon.
* sample identity: trajectory, Re, AoA, frame start/end, fold.
* scorer version dùng cho local evaluation.

Mọi prediction và target phải join bằng `sample_id`, không suy identity từ row position.

Kiểm tra:

1. frame overlap giữa train/validation/evaluation;
2. trajectory/Re không bị trộn fold ngoài thiết kế;
3. research inference và packaged `sub7` cho cùng output trên cùng input;
4. scorer recomputation tái tạo các local raw metrics;
5. mọi local composite chỉ được gọi là **development proxy**, không phải official final score;
6. các bảng Hypo 13/14 bị gán sai Re phải sửa provenance, nhưng lỗi per-Re này không được phép chặn model experiment nếu aggregate prediction không bị ảnh hưởng.

### Checkpoint reuse audit

Trước khi train mới, lập bảng:

| Checkpoint | Seed | Architecture |   λ | Split | Optimizer | Epoch policy | Input/features | Reusable? |
| ---------- | ---: | ------------ | --: | ----- | --------- | ------------ | -------------- | --------- |
| O0         |   42 | Ordinary     |   0 | ...   | ...       | ...          | ...            | yes/no    |
| O0         |   43 | Ordinary     |   0 | ...   | ...       | ...          | ...            | yes/no    |
| O3         |   42 | Ordinary     | .30 | ...   | ...       | ...          | ...            | yes/no    |
| O3         |   43 | Ordinary     | .30 | ...   | ...       | ...          | ...            | yes/no    |
| Z0         |   42 | ZeroMean     |   0 | ...   | ...       | ...          | ...            | yes/no    |
| Z0         |   43 | ZeroMean     |   0 | ...   | ...       | ...          | ...            | yes/no    |
| Z3         |   42 | ZeroMean     | .30 | ...   | ...       | ...          | ...            | yes/no    |
| Z3         |   43 | ZeroMean     | .30 | ...   | ...       | ...          | ...            | yes/no    |

**Reuse là mặc định.** Chỉ train lại checkpoint nếu không chứng minh được protocol compatibility.

Không được train lại tám run chỉ để có một bảng đẹp nếu artifact tương đương đã tồn tại.

### Gate G0

Pass khi:

* sample identity/tensor convention đúng;
* packaged-vs-research inference parity được giải thích trong tolerance kỹ thuật định trước;
* scorer/local metrics tái lập;
* checkpoint reuse status được xác định.

Nếu target alignment hoặc scorer còn sai, dừng model experiments.

---

## G1 — Audit các thành phần hiện có bằng saved predictions

Không train.

### A1. ZeroMean projection có thực sự giữ temporal mean trước postprocessing?

So:

$$
Y_{\mathrm{base}}
$$

và

$$
Y_{\mathrm{base}}+r',
\qquad
r'_t=r_t-\frac1T\sum_\tau r_\tau.
$$

Kiểm tra từng sample/pixel/channel:

$$
\max
\left|
\operatorname{mean}_t(Y_{\mathrm{base}}+r')
-
\operatorname{mean}_t(Y_{\mathrm{base}})
\right|.
$$

Đây là implementation/algebra check, không phải statistical hypothesis.

Nếu không khớp trong tolerance G0 → bug.

---

### A2. Persistent-zero guardrail có thực sự có ích cho predictor hiện tại?

Giữ nguyên:

* CNO;
* ZeroMean head ensemble;
* uncertainty policy;
* precision.

Chỉ đổi:

$$
\text{mask OFF}
\quad vs \quad
\text{mask ON}.
$$

Báo:

* aggregate RelL2/TKE/MVPE;
* delta theo trajectory;
* delta theo horizon;
* regression rate;
* fraction history-persistent-zero nhưng future target nonzero;
* error contribution chỉ trong vùng mask và ngoài mask.

Không gán persistent-zero thành airfoil, shadow hoặc solid vật lý.

**Output mong muốn:** quyết định guardrail là:

* beneficial;
* neutral;
* hoặc trade-off.

Không cần hidden claim nếu chưa có leaderboard ablation mask OFF.

---

### A3. Adaptive SPS tái lập đúng khi point prediction được cố định?

Giữ prediction tensor hoàn toàn cố định.

So:

1. constant bounds của `sub6`;
2. adaptive bounds của `sub7`.

Đo:

* SPS;
* coverage;
* mean/median interval width;
* width theo channel;
* width theo horizon;
* runtime.

Point metrics phải identical.

Hidden `sub6 → sub7` được giữ như external evidence riêng; local rerun chỉ kiểm tra code mechanism.

---

### A4. Adaptive SPS còn systematic calibration error để khai thác không?

Không thay predictor.

Phân tích SPS/coverage/width theo:

* channel \(u/v\);
* forecast horizon;
* quantile của history std;
* spatial region nếu region được định nghĩa a priori;
* Re/AoA chỉ như descriptive grouping, không mặc định là causal variable.

Bin thresholds phải chọn từ training/validation partition, không từ development holdout đã xem nhiều lần.

Không đặt coverage target 90%/95% nếu scorer không yêu cầu.

### Gate G1

Sau A1–A4 phải trả lời được hai câu:

1. guardrail có đáng giữ không?
2. adaptive SPS còn pattern residual rõ để thử một bounds modification nhỏ không?

Nếu A4 không có structured signal → đóng nhánh SPS, không sweep.

Nếu A4 có structured signal → đi **G2-UQ trước G3 architecture**, vì đây hiện là branch có hidden evidence/ROI tốt nhất.

---

## G2 — UQ candidate: chỉ khi A4 có structured signal

Không đổi point predictor.

### A5. Một thay đổi bounds đơn giản có tăng validation SPS không?

Chỉ chọn **một dimension** dựa trên A4, ví dụ:

* horizon scaling;
* channel scaling;
* history-std nonlinear scaling;
* hoặc một spatial factor được định nghĩa trước.

Không chỉnh nhiều dimension cùng lúc.

Trước khi chạy, khóa:

* candidate formula;
* parameter range;
* training/validation partition;
* maximum **3 candidates + baseline**;
* primary endpoint = validation SPS;
* secondary endpoints = coverage, width, latency;
* tie → baseline hoặc candidate rẻ hơn.

Candidate inference chỉ được dùng thông tin có sẵn từ input.

Không dùng 80-window dev holdout để fit rồi gọi kết quả đó confirmation.

### Gate G2-UQ

Chọn tối đa **một** UQ candidate nếu:

* validation SPS tốt hơn baseline vượt numerical/runtime noise;
* width không tăng vô lý;
* latency được đo;
* không cần thêm unverified inference metadata.

Nếu không có candidate rõ → giữ SPS của sub7.

Không mở sweep lớn hơn để “cứu” hypothesis.

---

## G3 — Matched ZeroMean × TKE-loss causal audit

Mục đích chính là sửa confound hiện tại.

Chỉ train checkpoint còn thiếu sau G0 reuse audit.

### A6. ZeroMean centering có làm TKE-targeted learning hữu ích hơn Ordinary không?

Thiết kế:

| Architecture | λ = 0          | λ = 0.30       |
| ------------ | -------------- | -------------- |
| Ordinary     | O0 seeds 42/43 | O3 seeds 42/43 |
| ZeroMean     | Z0 seeds 42/43 | Z3 seeds 42/43 |

Cùng:

* split;
* sample order;
* seed;
* initialization rule;
* head capacity;
* features;
* normalization;
* optimizer;
* batch;
* epochs;
* scheduler;
* checkpoint policy.

**Head-alone là primary comparison.**
Mask OFF.
Same simple interval policy.

### Hai loại comparison phải tách riêng

#### Scientific comparison — paired seed

$$
Z3_{42}-O3_{42},
\qquad
Z3_{43}-O3_{43}.
$$

Tương tự cho O0/Z0.

Không dùng ensemble để chứng minh architectural effect.

#### Deployment comparison

Sau paired analysis mới tạo:

$$
ensemble(Z3_{42},Z3_{43})
$$

vs

$$
ensemble(O3_{42},O3_{43}).
$$

Đây trả lời câu hỏi deployment, không thay thế causal comparison.

### Predeclared contrasts

$$
Z0-O0:
\text{ effect of centering without TKE loss}
$$

$$
O3-O0:
\text{ effect of TKE loss in Ordinary}
$$

$$
Z3-Z0:
\text{ effect of TKE loss in ZeroMean}
$$

$$
Z3-O3:
\text{ matched architecture comparison at }\lambda=.30
$$

Với TKE error:

$$
I=
(E_{Z3}-E_{Z0})
-
(E_{O3}-E_{O0}).
$$

Nếu \(I<0\), TKE loss cải thiện TKE nhiều hơn trong ZeroMean trong setting này.

Primary endpoint:

* TKE error.

Mandatory trade-offs:

* RelL2;
* MVPE;
* horizon errors;
* trajectory-level deltas.

Báo từng seed trước, ensemble sau.

Nếu hai seeds ngược dấu hoặc gain tương đương numerical/seed noise → **inconclusive**.

Không mở lambda sweep .35/.40 để cứu hypothesis.

### Gate G3

Kết quả phải cho phép một trong ba conclusion:

1. ZeroMean interaction được supported trong setting này;
2. ZeroMean chỉ cung cấp invariance nhưng không tạo TKE advantage;
3. chưa phân xử.

Chọn tối đa **một** point-predictor candidate hoặc giữ predictor sub7.

Không tuyên bố generalization tại đây.

---

## G4 — Secondary grouped confirmation

Chỉ chạy nếu G2 hoặc G3 tạo candidate có khả năng thay sub7.

Khóa candidate/hyperparameters trước khi mở confirmation result.

Confirmation này chỉ được gọi:

> **secondary grouped confirmation for head/UQ fitting**

không gọi là pristine unseen test của backbone.

### Nếu mục tiêu là scientific interaction claim

Chạy lại full:

$$
2\ architectures
\times
2\ lambdas
\times
2\ seeds
=
8\ runs
$$

trên secondary grouped split.

### Nếu mục tiêu chỉ là chọn submission

Chạy:

* candidate;
* matched incumbent/control;

mỗi bên 2 seeds.

Tối đa 4 runs.

Phải quyết định mục tiêu **trước khi chạy**.

Không được đổi λ, architecture, bounds formula hoặc checkpoint policy sau khi nhìn confirmation.

Nếu protocol bị sửa, partition đó trở thành development và không còn được gọi confirmation.

### Gate G4

Candidate được coi là submission-worthy nếu:

* gain/trade-off lặp lại qua seeds và grouped conditions;
* không có metric collapse chưa được chấp nhận trước;
* runtime cost được đo;
* deployment information hợp lệ.

Nếu không → giữ sub7.

---

## G5 — Packaging và official evaluation

Scientific evidence và competition decision là hai thứ khác nhau.

### Packaging

Đóng gói exact candidate:

* ZIP immutable;
* SHA-256;
* checkpoint hashes;
* source hash;
* API/shape test;
* finite outputs;
* deterministic repeated inference;
* exact bounds contract.

Benchmark:

* cùng hardware;
* same batch;
* warm-up cố định;
* 5 repeated end-to-end measurements;
* report median + range.

Local runtime không thay official Time Score.

### Official submission decision

Không dùng fabricated local final-score formula để tuyên bố candidate chắc chắn thắng.

Nếu candidate:

* có clear point/SPS gain với negligible trade-off → submit;
* có trade-off → có thể gửi **một exploratory submission** nếu trade-off đã được khai báo trước;
* unstable → giữ sub7.

Sau official result:

* record exact ZIP/hash;
* record raw subscores/final;
* chỉ lúc đó mới thay incumbent.

Official leaderboard đã được dùng nhiều lần sẽ trở thành feedback tuning; không gọi mỗi submission mới là một independent scientific confirmation.

---

## Ngân sách và thứ tự thực thi

Thứ tự:

$$
\boxed{
G0
\rightarrow
G1
\rightarrow
G2_{\mathrm{UQ}}\ \text{nếu có signal}
\rightarrow
G3_{\mathrm{matched}}
\rightarrow
G4_{\mathrm{confirmation}}
\rightarrow
G5_{\mathrm{submission}}
}
$$

### Batch đầu tiên

Chỉ chạy:

* G0;
* G1;
* checkpoint reuse audit.

**Không train gì trước khi hoàn thành ba việc này.**

### Development training budget

G3 chỉ train checkpoint còn thiếu.

* Best case: chỉ **1–2 runs**.
* Worst case nếu không checkpoint nào reusable: **8 runs**.

Không mặc định dùng hết 8.

### UQ budget

Tối đa:

* baseline;
* 3 candidate configurations.

Không train backbone/head.

### Confirmation budget

Chỉ mở khi có candidate:

* competition-only: tối đa 4 runs;
* full scientific confirmation: tối đa 8 runs.

Không train lại CNO trong protocol này.

---

## Hard stopping rules

Dừng branch nếu:

* gain không vượt measurement/seed noise;
* seeds cho dấu trái ngược;
* validation không support;
* candidate chỉ đẹp trên development holdout đã tune nhiều lần;
* runtime penalty không được bù bằng metric gain;
* hypothesis phải thay nhiều component cùng lúc mới “thắng”.

Không mở trong vòng này:

* large transport backbone;
* Reynolds exponent conditioning;
* phase-specific architecture;
* optical-shadow/airfoil geometry model;
* arbitrary stochastic noise generator;
* CNO retraining.

Các hướng này chỉ được mở lại nếu một diagnostic mới tạo ra evidence có khả năng thay đổi architecture decision.

---

## Required experiment record

Mỗi gate phải kết thúc bằng:

`hypothesis → exact protocol/hash → result → uncertainty/trade-off → supported / not met / inconclusive → next decision`

Artifact chi tiết có thể nằm trong `research/`, nhưng decision và evidence level phải được cập nhật ngay trong architecture record.

### Final principle

Không chạy experiment chỉ vì nó “thú vị”.

Mỗi experiment mới phải trả lời:

> **Nếu kết quả A xảy ra, architecture/submission decision thay đổi thế nào? Nếu kết quả B xảy ra, ta dừng hướng nào?**

Nếu cả hai outcome đều không thay đổi quyết định, không chạy experiment đó.


### Nhóm đã làm gì

Pipeline giữ CNO đóng băng, bổ sung residual head nhận history, dự báo CNO và stationary prior. Head mới trừ trung bình correction theo 20 frame tương lai, để điều chỉnh dao động trong khi giữ trung bình thời gian của CNO trước mask. Hai head seed 42/43, lambda TKE = 0.30, được ensemble. Prediction sau đó được đặt zero tại pixel có toàn bộ history UV bằng zero. Khoảng SPS phụ thuộc độ lệch chuẩn history và horizon, được tính vector hóa trên GPU.

- Sub1: CNO và SPS cố định, final 78.421963.
- Sub3: thêm residual head thông thường; **điểm đang có mâu thuẫn nguồn**: walkthrough ghi khoảng 78.68, phản biện mới ghi 78.931207 (SPS khoảng 37.451, time khoảng 85.789). Chưa tìm thấy raw Codabench record local để phân xử; không dùng sub3 làm mốc tính gain.
- Sub4 → sub5: adaptive SPS ban đầu tốn thời gian CPU, sau đó được vector hóa trên GPU; sub5 đạt 79.100273 theo walkthrough.
- Sub6: zero-mean head lambda 0.30, guardrail zero, SPS cố định.
- Sub7: giữ point predictor của sub6 và thêm adaptive SPS.

| Submission | RelL2 score | TKE score | MVPE score | Time score | SPS score | Final được báo |
|---|---:|---:|---:|---:|---:|---:|
| sub1 | 94.504578 | 73.717525 | 93.484537 | 87.472436 | 35.149137 | 78.421963 |
| sub6 | 94.551736 | 74.823387 | 93.487926 | 87.356676 | 35.513169 | 78.639483 |
| sub7 | 94.551736 | 74.823387 | 93.487926 | 87.277381 | 37.579004 | 79.158181 |

Sub7 hơn sub1 **0.736218 final**, TKE +1.105862 và SPS +2.429867; hơn sub5 0.057908 final. So với sub6, ba điểm point prediction giữ nguyên, SPS tăng 2.065835, time giảm 0.079295 và final tăng 0.518698. Đây là tiến bộ thực nghiệm được báo, không phải “mọi hypothesis đều thất bại”; nhưng chưa cô lập đóng góp của từng thay đổi từ sub1 đến sub6.

### Artifact đã xác nhận

[ZIP sub7](CCCCCC/submissions/sub7_cno_zeromean_head_adaptive_sps.zip) tồn tại, có 63 entry, SHA-256 `2e42b16da2eba7e794d90579003e31c9417d3e720f7babdb932bd376f810762a` khớp walkthrough. Kích thước thực tế 30,070,489 byte, khác 30,070,381 trong văn bản. Code và checkpoint trong ZIP khớp bản giải nén đã đọc.

Sub5/6/7 dùng cùng checkpoint CNO. Sub6 và sub7 dùng cùng hai head; diff code xác nhận chỉ thay phần SPS và các khai báo liên quan. Head đóng gói khớp checkpoint thí nghiệm zero-mean lambda 0.30 seed 42/43, best epoch 10/12. Mỗi head có **55,528 tham số**, không phải 55,144.

Nguồn: [submission sub7](CCCCCC/submissions/sub7_cno_zeromean_head_adaptive_sps/submission.py), [model](research/exp_zero_mean_head/models.py), [training](research/exp_zero_mean_head/train_and_ablate.py), [evaluation](research/exp_zero_mean_head/evaluate_suite.py), [ablation](research/exp_zero_mean_head/artifacts/ablation_summary.csv), [phase 2](research/exp_zero_mean_head/artifacts/phase2_evaluation_summary.csv).

### Những diễn giải cần sửa hoặc giữ ở mức giả thuyết

1. **Công thức final local không tái tạo điểm được báo.** Trọng số `0.25 RelL2 + 0.25 TKE + 0.25 MVPE + 0.15 SPS + 0.10 time` cho sub7 **80.08035095**, không phải 79.158181; cho sub1 79.44627415, không phải 78.421963. Starting-kit scorer đang có không công bố phép gộp cuối. Các “final” local dùng công thức này phải gọi là **proxy**; chưa thể dùng chúng để giải thích chính xác chênh lệch hidden score hoặc quy đổi SPS gain thành final. Không ghép local error với hidden subscore thành cùng phép đo.

2. **Zero-mean bảo toàn mean trước mask, không bảo đảm toàn pipeline.** Với `p = (CNO + correction) * fluid_mask`, mean cuối là `mean(CNO) * fluid_mask`. Bảng phase 2 ghi MVPE xấu hơn baseline ở 4/80 cửa sổ sau guardrail dù aggregate cải thiện. Đây là ràng buộc đại số của head, chưa phải định luật bảo toàn vật lý. History toàn zero cũng chưa tự chứng minh pixel là vật rắn hay bóng quang học.

3. **Ablation lambda lớn chưa cô lập tác dụng centering.** Ở lambda 0.25/0.30, ordinary head có một seed nhưng zero-mean là ensemble hai seed. Không thể quy hết gain khoảng 0.073–0.082 proxy cho zero-mean. Lượt mới còn chọn epoch bằng validation proxy, khác loss screen cũ chọn theo field với guardrail nghiêm ngặt. Kết quả mới không phủ nhận kết quả theo giao thức cũ, cũng chưa chứng minh lambda 0.30 tối ưu hoặc đã đạt plateau.

4. **“89% lỗi là spatial phase” vượt quá phép đo.** Hypo 13 cho thấy rescale một hệ số của bản đồ TKE chỉ giảm trung bình khoảng 11.05% relative error. Phần còn lại là phần chưa được giải thích bởi họ biến đổi scalar đó; có thể gồm sai vị trí, hình dạng và amplitude không đồng đều. Không có phép phân rã chứng minh 88.95% là phase error. Correlation bản đồ khoảng 0.743 là quan sát, chưa xác định cơ chế duy nhất.

5. **Nhãn Reynolds trong hypo 13/14 bị gán sai.** Notebook gán hàng 0–27 là Re 6306, 28–53 là 13977, 54–79 là 24204. [Manifest](research/score_followup/loss_screen/kaggle_results/loss_screen/primary_manifest.csv) của 80 hàng test thực tế là 0–29: **13977**, 30–49: **24204**, 50–79: **6306**. Bảng theo Re cần tính lại trước khi sử dụng. Sai nhãn này không tự làm sai thống kê aggregate trên 80 hàng, không nằm trong point predictor sub7 và không phủ nhận điểm submission được báo.

6. **RMS–Re là quan hệ thực nghiệm, chưa phải strict scaling law.** Hypo 14 fit thống kê 81 condition gộp thành 18 mức Re; đại lượng là RMS trung bình không gian toàn grid, không riêng wake trên 14 trajectory. Exponent 0.678 và R² 0.9296 là output fit. RMS 0.01379 tại Re 27975 là ngoại suy, không phải quan sát hidden target. AoA chưa được kiểm soát trong lập luận nhân quả. Giải thích SPS hidden thấp do Re cao vẫn là giả thuyết.

7. **SPS không độc lập hoàn toàn với point prediction.** [Scorer local](CCCCCC/realpde_t1_starting_kit_v9/scoring.py) dùng coverage, penalty độ rộng dạng exponential và hệ số từ RelL2/TKE/MVPE, không phải pinball loss. Có thể đổi interval khi giữ prediction cố định như sub6 → sub7, nhưng không suy ra độc lập thống kê. Code vector hóa interval trên GPU vẫn có vòng lặp batch Python và chuyển CPU/GPU; “zero CPU loops/roundtrips” là diễn đạt quá mức.

Notebook đối chiếu: [hypo 12](hypothesis/hypo_12_optical_piv_shadow_and_scoring_bias.ipynb), [hypo 13](hypothesis/hypo_13_phase_shift_vs_amplitude_tke_decomposition.ipynb), [hypo 14](hypothesis/hypo_14_reynolds_extrapolation_and_zero_mean_invariance.ipynb). Đây là kiểm tra source và output đã lưu, chưa phải tái lập toàn bộ 14 hypothesis.

### Cập nhật cách hiểu và nhận định cũ

Insight có ích là **ràng buộc correction theo mean giúp thiết kế head nhắm vào dao động; adaptive interval và tốc độ thực thi bổ sung cải thiện submission**. Pipeline đã có kết quả được báo; các giải thích vật lý cần giữ đúng mức bằng chứng.

Tôi rút lại nhận định “chưa có gì submit được” nếu áp dụng cho trạng thái hiện tại: nhóm đã tạo sub3–sub7 và cung cấp điểm mới. Không bắt buộc retrain toàn bộ train mới được submit; head sub7 khớp checkpoint từ split thí nghiệm. Retrain là lựa chọn cần đánh giá, không phải điều kiện hợp lệ mặc định. Nhận định “CNO đã chứa transport nên transport không giúp” cũng chỉ là khả năng giải thích, chưa phải cơ chế đã được chứng minh.

Lượt này chỉ cập nhật báo cáo: chưa sửa notebook, đổi model, chạy Kaggle hay submit thêm. Zero-mean head đã được nhóm thực hiện, không được đề xuất lại như thí nghiệm chưa làm.

### Bổ sung sau hai bản phản biện: Facts / Evidence / Interpretation / Not established

Đã đọc cả hai bản phản biện do người dùng gửi sau walkthrough. Giữ lịch sử protocol và kết quả ngày 06/09; các kết quả sub6/sub7 là bằng chứng được bổ sung về sau, không phải điều đã biết khi chọn hướng nghiên cứu. Không tiếp nhận các đánh giá chủ quan như 8/10 hay 6/10 làm kết quả kiểm định.

| Phân loại | Nội dung được giữ trong research record |
|---|---|
| **Facts** | Artifact sub7 tồn tại và hash khớp; code sub6/sub7 giữ cùng point predictor; zero-mean correction có mean bằng zero trước postprocessing. Điểm sub6/sub7 là số người dùng cung cấp, chưa phải raw evaluator record được truy xuất độc lập trong lượt audit. |
| **Evidence** | Matched transport screen hỗ trợ stationary-prior head hơn transport-prior head trong cấu hình đã thử. Loss screen cũ tìm thấy trade-off nhưng không qua quy tắc lựa chọn đã chốt. So sánh sub6 → sub7 được báo giữ nguyên ba point scores, SPS tăng 35.513169 → 37.579004 và final tăng 78.639483 → 79.158181. |
| **Interpretation** | Residual correction nhắm vào TKE và adaptive input-conditioned bounds là hướng có giá trị thực dụng. Sub6 → sub7 là đối chứng sạch nhất hiện có cho thay đổi bounds trong pipeline này; kết luận không cần viện dẫn một cơ chế Reynolds cụ thể. |
| **Not established** | Hidden gain riêng của zero-mean so với ordinary head trong đối chứng cùng seed/ensemble; hidden gain riêng của mask; physical origin của persistent-zero; 89% phase error; tính phổ quát của exponent 0.678; lambda tối ưu; lợi ích nhân quả riêng của Sim; công thức final chính thức. |

**Phạm vi của các bằng chứng:** Kaggle trong các lượt screen là môi trường GPU chạy thí nghiệm offline, không phải hidden evaluation của competition. 80 cửa sổ được gọi là `test` trong artifact là development holdout cho head; chưa chứng minh backbone organizer chưa từng thấy các condition đó. Hidden aggregate hỗ trợ hiệu quả submission nhưng không tự chứng minh cơ chế, tính độc lập của từng thành phần, hoặc mức ổn định qua nhiều lần đánh giá. Không có một thứ hạng bằng chứng duy nhất cho mọi câu hỏi: leaderboard trả lời hiệu quả submission; matched ablation trả lời tác dụng của thay đổi được kiểm soát.

**Các thuật ngữ phải giữ chính xác:** `M_extra = M_hist \\ M_sim` là persistent-zero Real pixels ngoài nominal Sim mask; `M_both = M_hist ∩ M_sim` là phần giao hai mask. Không gọi chúng lần lượt là PIV shadow và true airfoil khi chưa có bằng chứng vật lý. Biến `solid` trong submission thực tế chỉ phát hiện persistent exact-zero history. Nếu refactor code nghiên cứu sau này, tên `persistent_zero_mask` mô tả đúng hơn; lượt này không sửa artifact đã được chấm.

**Không gộp các phép kiểm tra khác nhau:** kết quả không có regression trong audit 40 cửa sổ không phải bảo đảm cho mọi cửa sổ hay mọi predictor. Bảng 80 cửa sổ của zero-mean + guardrail có 5% MVPE regression. Chênh lệch cũng cần ghi rõ comparator trước khi quy tác dụng riêng cho guardrail. Tương tự, ordinary head tệ hơn ở lambda cao chưa đủ chứng minh overfitting nếu chưa có bằng chứng train–validation gap.

**Không đồng nhất hai amplitude oracle:** oracle trên fluctuation MSE và oracle rescale bản đồ TKE tối ưu các đại lượng khác nhau. Các tỷ lệ lỗi chưa giải thích của chúng không phải tỷ lệ phase error và không cộng/gộp trực tiếp được. Việc adaptive SPS dùng history std, không dùng exponent Reynolds, cũng có nghĩa thành công sub7 không kiểm định exponent 0.678 trên hidden.

**Leaderboard provenance còn thiếu:** cần raw record gắn submission ID, ZIP/hash, thời điểm, từng subscore và final trước khi công bố Master Leaderboard như bảng đã xác minh. Hiện sub3 có hai số xung đột như ghi ở trên; tìm kiếm local trong Markdown/JSON/CSV chưa thấy record cho 78.931207. Không tự thay bằng số mới như một correction đã được chứng minh. Sub7 được coi là mốc tốt nhất trong các kết quả được cung cấp, không phải tuyên bố đã kiểm tra mọi submission của dự án.

**Quy tắc trình bày kết quả:** dùng “offline development composite proxy” cho phép gộp local. Không tính “SPS collapse cost” bằng cách lấy SPS dev trừ SPS hidden rồi nhân trọng số giả định: vừa khác phân bố vừa chưa xác minh công thức. Dùng sub6 → sub7 cho đối chứng bounds. Gain sub7 so với sub1 là +0.736218, so với sub5 là +0.057908; không gọi đó là bằng chứng các thành phần cộng độc lập. So sánh point predictor mới với sub5 cần báo trade-off từng metric, không gọi là thắng toàn diện. Quy tắc Pareto nghiêm ngặt của loss screen cũ là mục tiêu lựa chọn lúc đó, không phải mục tiêu competition tối ưu đã được chứng minh.

Lượt tiếp nhận phản biện này cập nhật tài liệu và provenance; chưa chạy thêm thí nghiệm, sửa notebook hoặc submit. Việc audit raw leaderboard vẫn chưa hoàn tất, không được ghi như đã hoàn tất.

## 2026-09-06: scored submission and architecture experiments (lịch sử; trạng thái mới ở trên)

**The 78.421963 result belongs to `CCCCCC/submissions/sub1_cno_sps.zip`, as confirmed by the user. Keep it as the incumbent. Eight architecture models have now actually trained on Kaggle: transport helps the small Real-history model, but the frozen CNO benefits more from the matched head without transport. The next incumbent candidate is the small residual head, not a larger transport architecture.** Kaggle authentication from the notebook's second cell succeeded. The private experiment [RPDE Architecture Screen V1](https://www.kaggle.com/code/phancanh/rpde-architecture-screen-v1) completed as version 1, kernel ID 133291362. No leaderboard submission has been made by this experiment.

**Cập nhật mới nhất:** lượt kiểm tra loss TKE cũng đã COMPLETE và được kiểm tra lại sau tải xuống. Tổng cộng hiện có **16 lượt huấn luyện trên Kaggle: 8 lượt architecture screen + 8 lượt loss screen**. Loss TKE tạo được cải thiện năng lượng nhỏ, nhưng chưa có cấu hình đạt đồng thời điều kiện bảo toàn field/MVPE đã chốt. Chưa có điểm leaderboard mới; ZIP 78.421963 vẫn là incumbent. Kết quả chi tiết của lượt mới nằm ngay dưới đây.

### Kiểm định tiếp theo: phân rã residual và đối chứng loss TKE

Theo yêu cầu tiếp tục kiểm định thay vì coi diễn giải là sự thật, các lựa chọn thiết kế dưới đây được ghi rõ trước khi huấn luyện. Một cấu hình hay ngưỡng chọn model là quyết định thực nghiệm, không phải bằng chứng về cơ chế dữ liệu. Trạng thái Kaggle của lượt architecture screen đã được truy vấn lại và xác nhận **COMPLETE**.

**Phép kiểm tra mới đã hoàn tất trên dự đoán lưu từ lượt trước:** phân rã MSE CNO thành lỗi trung bình cửa sổ và lỗi dao động cho thấy phần dao động chiếm **68.57% tổng squared error**, còn phần trung bình chiếm **31.43%**, trên cùng 80 cửa sổ test. Đây là tỷ lệ của tổng năng lượng sai số, không phải trung bình các tỷ lệ của từng cửa sổ. Đẳng thức phân rã được kiểm tra bằng số học float64. Như vậy, giả thuyết residual tập trung nhiều hơn ở dao động có bằng chứng trực tiếp trong tập này, nhưng phần lỗi trung bình vẫn đáng kể.

Increment RelL2 của CNO là **1.02869**; head stationary-prior cho **1.01420** và **1.01022** ở hai seed. Một dự báo có increments bằng zero cho giá trị 1 theo định nghĩa này, nên field RelL2 tốt không có nghĩa CNO dự báo biến thiên từng bước tốt hơn đối chứng zero-increment. Đây là metric chẩn đoán riêng, không thay thế field score và cũng chưa xác định dao động nào bất khả dự báo. Kết quả thay đổi theo Reynolds: increment error CNO lần lượt là 1.13874, 0.98903 và 0.92312 ở Re 6306, 13977 và 24204; không thể suy rộng giá trị aggregate >1 thành thất bại ở mọi nhóm. Tỷ lệ fluctuation/total squared error tương ứng là 57.04%, 65.78% và 71.14%, đều lớn hơn một nửa.

Tỷ trọng năng lượng phổ thời gian ở dải từ một phần tư Nyquist trở lên, sau chuẩn hóa tổng năng lượng trong từng cửa sổ, là **14.95% ở CNO so với 15.49% ở target**. Trong khi đó tỷ lệ tổng TKE CNO/target trung bình chỉ khoảng 0.514. Những con số này không hỗ trợ việc đơn giản đồng nhất residual với “CNO thiếu riêng tần số cao”: model có thể có tỷ trọng phổ gần đúng nhưng sai năng lượng tuyệt đối, vị trí hoặc pha. Đây là quan sát về phổ gộp trong 20 frame, không xác nhận tính đúng đắn của phổ tại từng pixel.

Chẩn đoán còn cho thấy Real-history transport head cải thiện field/TKE nhưng không cải thiện mọi đại lượng thời gian: temporal PSD-shape L1 và sai số ACF lag 1 tăng so với stationary-prior head ở cả hai seed. Do đó không nên suy từ TKE tốt hơn thành “temporal structure tốt hơn” trên mọi phương diện. PSD ở đây tính trên 20 frame nên chỉ là kiểm tra phổ ngắn, không đủ để kết luận về các tần số vật lý được phân giải tốt trên quỹ đạo dài.

Evidence: [script phân rã](research/score_followup/diagnose_saved.py), [định nghĩa diagnostics](research/score_followup/temporal_diagnostics.py), [kết quả tổng hợp](research/score_followup/temporal_summary.json), [kết quả từng cửa sổ](research/score_followup/temporal_decomposition.csv), [lỗi từng horizon](research/score_followup/temporal_horizons.csv).

**Giao thức loss được chốt trước lượt Kaggle mới:** giữ nguyên CNO đóng băng, head 55,528 tham số và stationary prior, hai seed 42/43, 12 epoch, batch size 12, AdamW và learning rate 0.001. Chỉ thay `lambda` trong `[0, 0.1, 0.3, 1]`. Loss field giữ nguyên normalized residual MSE của lượt trước; loss TKE là trung bình relative L2 của bản đồ TKE trong đơn vị vật lý, tính qua 20 frame; mẫu số norm target-map có floor `1e-8`. Không thêm loss phổ, increment hay transport.

Epoch được chọn theo validation field RelL2, bao gồm epoch zero. Sau khi cả tám run hoàn tất fitting, lambda được chọn theo TKE validation trung bình hai seed, với điều kiện field RelL2 và MVPE validation không lớn hơn đối chứng lambda zero; chỉ cho phép sai số số học tuyệt đối `1e-7`. Nếu hòa, chọn lambda nhỏ hơn. Lambda được chốt trước khi đánh giá test. Grid này là phạm vi kiểm tra hữu hạn, không phải khẳng định đã tìm được hệ số tối ưu.

Split chính giữ test Re `{6306,13977,24204}` và validation Re `{10142,20369}` để so sánh trực tiếp với lượt trước. Nếu một lambda dương được chọn, chạy thêm bốn lượt từ khởi tạo mới: lambda zero và lambda đã chọn, mỗi cấu hình hai seed, với test Re `{8863,16534,25482}`, validation Re `{11420,19090}`. Lambda từ split chính giữ nguyên khi đánh giá split thứ hai. Nếu lambda zero được chọn, dừng nhánh xác nhận vì chưa có ứng viên loss TKE đạt quy tắc đã chốt. Điều kiện dừng này không bác bỏ mọi loss TKE hay mọi cách chọn trọng số.

Các test Reynolds group của hai split không trùng nhau. Tuy vậy, dữ liệu đã được thăm dò trước đó và checkpoint organizer có thể đã thấy tất cả nhóm Real; đây là kiểm tra bổ sung cho head fitting, không phải một benchmark hoàn toàn chưa từng được quan sát hay bằng chứng fold-pure của Sim pretraining. Tất cả kết quả phải báo riêng field, TKE, MVPE, SPS cố định, mean/fluctuation MSE, increment error, ACF và lỗi theo horizon. Nếu chỉ TKE tốt hơn, kết luận chỉ giới hạn ở khớp năng lượng.

Lượt mới dùng [loss_experiment.py](research/score_followup/loss_experiment.py), đóng gói bằng [prepare_loss.py](research/score_followup/prepare_loss.py) vào kernel private [RPDE TKE Loss Test V1](https://www.kaggle.com/code/phancanh/rpde-tke-loss-test-v1). Version 1, kernel ID 133306638, đã **COMPLETE**: 8 lượt huấn luyện trên Tesla T4, 279.78 giây script wall time, cùng 477 cửa sổ của 81 trajectory. Ngân sách tối đa là 3,600 giây, guard nội bộ 3,300 giây; thực tế không cần dùng hết ngân sách hay chạy nhánh xác nhận. Giao thức được ghi trước dispatch. [check_loss.py](research/score_followup/check_loss.py) đã kiểm tra loss tương đương scorer, gradient hữu hạn kể cả khi dự đoán bằng target, tính bất biến TKE khi đảo thời gian, khả năng phát hiện thay đổi đó của increment error, đẳng thức phân rã MSE, quy tắc chọn lambda, và các nhóm dữ liệu.

### Kết quả loss screen: cải thiện TKE có đánh đổi

### Trạng thái artifact có thể submit

Hiện có hai ZIP có cấu trúc submission hoàn chỉnh trong `CCCCCC/submissions/`:

- `sub1_cno_sps.zip`: **có thể submit**, đã được người dùng xác nhận và có kết quả `final_score = 78.421963`.
- `sub2_fno_sps.zip`: **cấu trúc có thể submit** theo wrapper hiện có, nhưng trong chuỗi kiểm định này chưa có điểm leaderboard mới được xác nhận.

Các file `*.pth` trong `research/score_followup/kaggle_results/architecture_screen/` và `.../loss_screen/` là checkpoint của head trong các fold thử nghiệm. Chúng **chưa phải submission**: chưa có `submission.py` hoàn chỉnh để xây feature/prior cho mọi batch test, chưa train final theo toàn bộ dữ liệu train, và các checkpoint gắn với split/seed/lambda cụ thể. Không được đổi tên `.pth` thành ZIP rồi gửi vì sẽ thiếu preprocessing, base CNO, interval wrapper và kiểm tra API.

Ứng viên có bằng chứng tốt nhất hiện tại là **CNO + stationary residual head, lambda 0**, nhưng chỉ đạt lợi ích nhỏ trên ba Reynolds group đã giữ lại và CNO organizer có thể đã thấy các điều kiện đó. Trước khi đưa lên leaderboard, cần một lượt riêng: chốt checkpoint/seed hoặc ensemble theo quy tắc định trước, train final head trên toàn `train_real`, tạo wrapper giữ nguyên metadata API và khoảng SPS của `sub1`, chạy smoke test local trên nhiều batch, đo thời gian end-to-end, sau đó mới cân nhắc submission. Lượt đó là một hành động bên ngoài khác với việc kiểm định Kaggle private đã làm; hiện chưa submit artifact mới và chưa nên coi `lambda 0` là ứng viên leaderboard đã được chứng minh.

### Phân loại hypothesis: không phải tất cả đều thất bại

Các kết quả nên đọc theo ba mức thay vì nhãn “thành công/thất bại” duy nhất:

| Hypothesis | Kết luận hiện tại | Bằng chứng và giới hạn |
|---|---|---|
| Real có chuyển động không gian có thể khai thác | **Được hỗ trợ một phần** | Shift/transport dùng lịch sử giúp Real-history model: RelL2 `0.109182 → 0.103101`, TKE `0.887771 → 0.829145`, ở cả hai seed và cả ba Re test groups. Đây là kết quả cơ chế trong model yếu. |
| Transport là bottleneck còn thiếu của CNO | **Không được hỗ trợ** | CNO residual head có transport kém hơn head matched không transport ở field/TKE. Transport chưa tạo payoff trên champion; điều này không chứng minh mọi dạng transport đều vô ích. |
| Một scalar amplitude boost sẽ sửa TKE | **Bị bác bỏ trong phạm vi grid đã thử** | Leave-Re-out luôn chọn amplitude `1.0` dưới guard field; oracle scalar khớp tổng năng lượng còn làm fluctuation MSE tăng ở 80/80 cửa sổ. Không bác bỏ correction phụ thuộc location/horizon. |
| Phần lớn residual nằm ở fluctuation thay vì mean | **Được hỗ trợ, chưa giải thích hết** | Fluctuation chiếm `68.57%` pooled squared error của CNO, mean chiếm `31.43%`. Tỷ lệ này thay đổi theo Re (`57.04–71.14%`), nên không phải hằng số phổ quát. |
| Residual chủ yếu là high-frequency | **Chưa được chứng minh** | High-frequency PSD fraction CNO `14.95%`, target `15.49%`; tổng TKE vẫn thấp. Sai số có thể là amplitude tuyệt đối, vị trí, phase hoặc phổ theo pixel. |
| Một residual head nhỏ có thể cải thiện CNO | **Được hỗ trợ ở mức nhỏ** | Head 55,528 tham số, không transport, giảm RelL2 `0.087406 → 0.086572`, TKE `0.649071 → 0.635797`, tăng local SPS khoảng `0.236`. Đây chưa phải leaderboard proof vì checkpoint organizer có thể đã thấy held groups. |
| Thêm TKE loss sẽ tạo cải thiện đồng thời | **Chưa đạt** | `lambda=0.1` giảm test TKE `0.635797 → 0.622199`, nhưng MVPE tăng `0.37%`; validation rule chọn `lambda=0`. Đây là trade-off có tín hiệu, không phải “mọi thứ fail”. |
| Tối ưu interval sẽ giải quyết phần điểm thấp | **Chỉ cải thiện nhỏ** | Constant-width calibration tăng local SPS khoảng `0.622` điểm; channel/horizon fit không tăng aggregate. Không đủ lý do thay incumbent bands. |

Vì vậy, thứ đã thất bại là một số **can thiệp đơn giản**: warp trực tiếp cho CNO, tăng variance toàn cục, và tối ưu TKE mà không kiểm soát trade-off MVPE. Thứ đã rút ra được là thông tin định hướng: transport có thông tin dự báo nhưng CNO đã khai thác phần tương tự; residual còn lại mang tính dao động và spatial/temporal alignment; một adapter nhỏ cải thiện được incumbent ở mức khiêm tốn; TKE loss có thể thay đổi đúng metric nhưng chưa tạo Pareto win.

Kết quả âm tính này có giá trị vì nó loại bớt các hướng dễ làm nhưng dễ tự lừa: không thêm noise để ép TKE, không áp dụng variance boost mù, không xây transport model lớn chỉ vì correlation shift, và không chọn lambda bằng test. Bước tiếp theo cần kiểm tra **correction dao động có ràng buộc mean bằng zero**, hoặc loss dựa trên increment/ACF với cùng giao thức validation. Đây là các hypothesis mới, chưa được coi là đúng trước khi chạy.

#### Làm rõ: kết quả có thật sự tệ hơn không?

Không. Cách tóm tắt “lambda TKE thất bại” dễ gây hiểu nhầm. Đúng hơn là:

1. **Không có lỗi tính toán đã phát hiện.** Sau khi tải kết quả, script tính lại 9 bộ dự đoán; chênh lệch RelL2/TKE/MVPE lớn nhất là `1.79e-7`, SPS khớp trong `1e-4`, source hash và checkpoint hash khớp, tất cả run đều có đủ log. `check_loss.py` cũng kiểm tra gradient hữu hạn, TKE bằng zero khi dự đoán bằng target, và TKE bất biến khi đảo thứ tự frame. Đây không phải bằng chứng code đúng tuyệt đối, nhưng không có dấu hiệu lỗi triển khai trong các kiểm tra đã định.

2. **Lambda 0.1 không tệ hơn trên mọi metric.** Trên test, so với lambda 0, nó giữ RelL2 gần như ngang nhau (`0.086575` so với `0.086572`), giảm TKE error `0.635797 → 0.622199`, giảm increment error `1.012206 → 1.010923`, giảm PSD-shape L1 `0.144218 → 0.139699`, giảm ACF error `0.021885 → 0.021313`, và tăng SPS `44.6044 → 44.6980`. Đánh đổi là MVPE error tăng `0.082100 → 0.082407`. Vì vậy đây là một **trade-off có lợi cho temporal diagnostics**, không phải một run tệ hơn toàn diện.

3. **Lambda 0 được chọn vì quy tắc validation đã chốt trước đó, không vì test xấu.** Quy tắc yêu cầu field RelL2 và MVPE không được xấu hơn lambda 0 trong validation, rồi trong tập còn lại chọn TKE thấp nhất. Lambda 0.1 giảm TKE validation `0.653040 → 0.641275`, nhưng MVPE validation tăng `0.068983 → 0.069123`; do đó bị loại. Đây là một lựa chọn bảo thủ để tránh nhìn test rồi chọn model, không phải kết luận rằng lambda 0.1 vô dụng. Lambda 0.3 còn giảm TKE validation hơn nữa nhưng field/MVPE xấu hơn; lambda 1 được chọn epoch zero.

4. **Test này không kiểm định được “breakthrough”.** Nó chỉ cho thấy loss TKE có thể dịch chuyển phân bổ lỗi theo hướng temporal với cái giá nhỏ ở MVPE. Hai seed và ba Reynolds group là pilot; không có khoảng tin cậy cấp điều kiện hay điểm final leaderboard. Hơn nữa CNO checkpoint là Sim+Real organizer checkpoint, nên không thể dùng kết quả này làm bằng chứng sạch về Sim-to-Real generalization.

Do đó các hypothesis hiện có nên được gắn nhãn như sau: global amplitude boost **không đủ**; transport là cơ chế hữu ích cho Real-history model yếu nhưng **chưa có payoff thêm trên CNO**; residual head quanh CNO có **lợi ích nhỏ và lặp lại**; loss TKE có **tín hiệu tích cực về TKE/temporal diagnostics nhưng chưa có Pareto win theo rule đã đặt**. Không hypothesis nào trong số này bị bác bỏ hoàn toàn chỉ bởi một metric xấu hơn.

Điểm có thể cải thiện trong quy trình của mình là đã dùng một tiêu chí chọn hợp nhất khá chặt (`MVPE không được tăng dù chỉ một lượng nhỏ`). Tiêu chí đó phù hợp với yêu cầu “không tự đánh đổi metric”, nhưng không phù hợp nếu mục tiêu khoa học là đo Pareto frontier giữa field, temporal và MVPE. Ở lượt kế tiếp, mình sẽ tách hai câu hỏi trước khi chạy: (a) có cấu hình nào cải thiện TKE với field giữ trong tolerance định trước không, và (b) nếu MVPE tăng, mức đánh đổi đó có ổn định theo Reynolds không. Khi đó sẽ báo cáo toàn bộ frontier, không gọi các điểm không được chọn là “thất bại”.

Validation chọn **lambda = 0**. Trung bình hai seed, lambda 0.1 giảm TKE validation từ 0.653040 xuống 0.641275, nhưng field RelL2 tăng từ 0.077891 lên 0.077922 và MVPE tăng từ 0.068983 lên 0.069123. Cả lambda 0.3 và 1 cũng không vượt điều kiện chọn. Nhánh xác nhận trên nhóm Reynolds khác được **bỏ qua đúng điều kiện dừng đã công bố**, vì chưa có lambda dương đạt tiêu chí. Không nới ngưỡng sau khi nhìn thấy test để biến một cấu hình thành kết quả đạt yêu cầu.

Tất cả cấu hình vẫn được báo cáo minh bạch trên cùng 80 cửa sổ test sau khi lambda đã được chốt. Các hàng trainable dưới đây là trung bình metric của hai seed, không phải ensemble. Bảng có thể mô tả các đánh đổi của cấu hình không được chọn, nhưng không thay thế lựa chọn bằng validation.

| Cấu hình | RelL2 error ↓ | TKE error ↓ | MVPE error ↓ | SPS ↑ | Increment RelL2 ↓ |
|---|---:|---:|---:|---:|---:|
| CNO incumbent | 0.087406 | 0.649071 | 0.082121 | 44.3683 | 1.028693 |
| CNO head, lambda 0 — được chọn | 0.086572 | 0.635797 | **0.082100** | 44.6044 | 1.012206 |
| CNO head, lambda 0.1 | 0.086575 | 0.622199 | 0.082407 | **44.6980** | **1.010923** |
| CNO head, lambda 0.3 | 0.087184 | **0.620294** | 0.082194 | 44.6227 | 1.025218 |
| CNO head, lambda 1 | 0.087406 | 0.649071 | 0.082121 | 44.3683 | 1.028693 |

Với lambda 0.1 so với lambda zero, TKE error giảm **2.14%**, SPS tăng **0.0936 điểm**, MVPE error tăng **0.37%**, và RelL2 error tăng **0.0035%** khi lấy trung bình seed. RelL2 gần như không đổi ở aggregate che đi khác biệt giữa seed: seed 42 xấu hơn, seed 43 tốt hơn. TKE giảm và MVPE xấu hơn ở cả hai seed. Vì final-score combination chưa được công bố, không thể kết luận từ đây rằng final leaderboard score chắc chắn tăng hay giảm.

Lambda 0.1 cũng giảm nhẹ increment error (**0.13%**), temporal PSD-shape L1 (**3.13%**) và ACF lag-1 absolute error (**2.62%**) ở aggregate. Các metric này cải thiện ở cả hai seed, nên kết quả không chỉ là tăng tổng năng lượng. Tuy nhiên, fluctuation MSE thay đổi rất ít và không cùng chiều giữa hai seed; chưa có bằng chứng về một bước tiến lớn trong dự báo dao động. Cần giữ giới hạn này bên cạnh kết quả TKE tốt hơn.

Lambda 0.3 chọn epoch zero ở seed 42 và epoch 10 ở seed 43; lambda 1 chọn epoch zero ở cả hai seed. Vì epoch zero là head bằng zero, các kết quả trùng incumbent ở những trường hợp này là **quyết định chọn checkpoint bằng validation**, không phải lỗi bỏ qua huấn luyện. Training log lưu đủ 12 epoch cho mỗi run. Kết quả này áp dụng cho kiến trúc, ngân sách và cách chọn epoch hiện tại; chưa bác bỏ mọi loss TKE hoặc mọi hệ số khác.

### Kiểm định bổ sung: vì sao khớp năng lượng chưa sửa được residual?

Một phép chẩn đoán dùng target, chỉ để phân tích và không thể triển khai tại inference, so sánh hai hệ số cho từng cửa sổ: hệ số scalar giảm fluctuation MSE nhiều nhất, và hệ số khớp đúng tổng TKE. Trên 80 cửa sổ CNO:

- Hệ số tối ưu fluctuation MSE có trung vị **0.8149**; hệ số khớp năng lượng có trung vị **1.4093**.
- Ngay cả scalar oracle tối ưu MSE theo từng target cũng chỉ giảm tổng fluctuation squared error **2.37%** trong họ phép biến đổi này.
- Khớp đúng tổng năng lượng làm fluctuation MSE **tăng ở cả 80/80 cửa sổ**, tổng mức tăng **23.51%**.
- Cosine giữa fluctuation dự đoán và target có trung vị **0.5777**.

Đây là bằng chứng trực tiếp rằng hiệu chỉnh một biên độ chung cho toàn cửa sổ không giải quyết được phần lớn sai số trong họ mô hình đã thử. Sai khác về cấu trúc/độ khớp dao động vẫn còn ngay cả khi oracle biết tổng năng lượng target. Phép đo không phân biệt đầy đủ lỗi pha, lỗi vị trí và bất định không thể dự đoán; không được suy từ đó rằng residual chắc chắn là noise hoặc stochastic model chắc chắn tốt hơn. Khớp tổng năng lượng cũng không tương đương khớp toàn bộ bản đồ TKE. Evidence: [script oracle](research/score_followup/diagnose_amplitude.py), [từng cửa sổ](research/score_followup/amplitude_oracle_diagnostic.csv), [tổng hợp](research/score_followup/amplitude_oracle_summary.json).

### Quyết định sau lượt kiểm định này và evidence

Giữ ZIP incumbent; chưa chọn model loss TKE để thay thế. Giả thuyết “phần lớn residual là lỗi dao động” được hỗ trợ trên tập test hiện tại. Giả thuyết mạnh hơn “chỉ thêm loss TKE sẽ cải thiện năng lượng mà bảo toàn field/MVPE” **chưa đạt tiêu chí** trong grid này. Cấu hình lambda 0.1 là một đánh đổi nhỏ có thể nghiên cứu tiếp, không phải kết quả hoàn toàn âm tính và cũng không phải breakthrough.

Kiểm định tiếp theo đáng chuẩn bị là **correction có trung bình theo thời gian bằng zero quanh dự đoán của CNO + MSE head đã cố định**: `delta_centered = delta - mean_time(delta)`. MVPE của scorer dùng trung bình theo thời gian tại các probe, nên ràng buộc này bảo toàn đại lượng đầu vào MVPE của base về mặt đại số; cần kiểm tra lại bằng scorer với tolerance số học. Việc nó có cải thiện TKE mà bảo toàn field hay không vẫn phải được thử bằng đối chứng cùng ràng buộc, lambda zero versus lambda dương, trên split đã định trước. Đây là đề xuất cho thí nghiệm kế tiếp, **chưa chạy trong lượt này**; không coi ràng buộc mean là lời giải đã được chứng minh. Giữ nguyên tiêu chí đã công bố của lượt loss screen hiện tại.

Đã tải đủ output và kiểm tra lại chín bộ dự đoán (incumbent + tám run): sai khác lớn nhất khi tính lại field/TKE/MVPE là **1.79e-7**, SPS và diagnostics thời gian khớp tolerance đã định, 96 bản ghi epoch đầy đủ, source/checkpoint hash khớp, split Reynolds không bị trộn, và quy tắc chọn lambda tái lập đúng bằng validation. [analyze_loss.py](research/score_followup/analyze_loss.py) tạo [bảng tổng hợp](research/score_followup/loss_screen/summary.csv), [so sánh theo cặp](research/score_followup/loss_screen/paired_comparisons.csv) và [biên bản kiểm tra](research/score_followup/loss_screen/analysis.json). Output gốc gồm [results](research/score_followup/loss_screen/kaggle_results/loss_screen/results.json), [selection](research/score_followup/loss_screen/kaggle_results/loss_screen/selection.json), [runtime](research/score_followup/loss_screen/kaggle_results/loss_screen/runtime.json), [training log](research/score_followup/loss_screen/kaggle_results/loss_screen/training.csv), [predictions](research/score_followup/loss_screen/kaggle_results/loss_screen/primary_predictions.npz) và tám checkpoint. Hai lượt Kaggle đã hoàn tất; hiện không còn job huấn luyện nào của chuỗi thực nghiệm này đang chạy.

### What the leaderboard result establishes

| User-reported metric | `sub1_cno_sps.zip` |
|---|---:|
| rel_l2_score | 94.504578 |
| tke_score | 73.717525 |
| mvpe_score | 93.484537 |
| time_score | 87.472436 |
| sps_score | 35.149137 |
| final_score | 78.421963 |

The user reports other baselines around 73–75, putting this artifact approximately 3.42–5.42 final-score points ahead. Those comparisons are not paired experiments with matched subscore records. The ZIP's wrapper uses the organizer's Sim+Real CNO checkpoint and constant uncertainty bands, with half-width `0.85 × 0.010925 = 0.00928625`. It does not establish that a new architecture outperforms CNO: its point predictor is CNO. The checkpoint SHA-256 is `82e842928a25dbf5a74c4e336bdd28e89bcf40e68bb8cdd213547f1246af4f61`; the Kaggle run requires an exact match before inference. The ZIP hash is recorded in the local controls' configuration.

The result makes field accuracy worth preserving while testing temporal-energy fidelity and interval quality. It does **not** show that the smallest subscore has the largest attainable final-score gain. The supplied scorer does not publish the final-score combination. We therefore compare field, TKE, MVPE, SPS, and inference costs separately and do not invent a local final score.

### Completed local hypothesis tests

Executed [run_controls.py](research/score_followup/run_controls.py) on 126 nonoverlapping 20-history/20-future windows, covering 21 conditions and 11 Reynolds groups, sampled at cached window indices `[0,8,16,24,32,40]`. Each cached prediction target was checked against its Real trajectory cache. Calibration leaves each Reynolds group out in turn. This isolates calibration fitting, but the organizer CNO may already have seen these conditions during its Real fine-tuning. These are postprocessing and mechanism screens, not clean estimates of generalization to new conditions.

Reproduce from the project root:

```powershell
python research/score_followup/run_controls.py
```

| Test | Observation | Decision |
|---|---|---|
| Is TKE mostly fixable by globally scaling CNO fluctuations around its predicted 20-frame mean? | Median predicted/target TKE energy ratio is 0.5071, but median TKE-map correlation is only 0.7523. Every leave-Re-out fold selects amplitude **1.0** from `[0.75,1,1.25,1.5,2,2.5,3]` when training field RelL2 may worsen by at most 2%. Aggregate errors remain RelL2 0.08906, TKE 0.69017, MVPE 0.08354. | Underpredicted energy exists, but this amplitude grid offers no improvement under the field constraint. Do not apply a blanket variance boost. This does not rule out finer or input-dependent corrections. |
| Can better intervals alone substantially improve local SPS? | Champion SPS 44.8415; leave-Re-out constant-width fit 45.4635; channel-by-horizon residual-RMS fit 44.4796. All constant-width folds choose half-width 0.008. Coverage changes from 85.73% to 83.02% for the constant fit. | A small local gain of 0.6220 SPS points does not justify replacing the leaderboard incumbent's bands. The richer tested band model does not improve the aggregate. Keep interval choices fixed during architecture screening. |
| Does causal transport help an actual 20-frame forecast? | With damping `0.9^h`, transport reduces RelL2 from 0.12157 to 0.11893 and TKE error from 0.94214 to 0.87158 versus the identical zero-transport control. Without damping, transport gives RelL2 0.12478 and TKE 0.79638 versus persistence 0.13463 and 1.0. CNO remains stronger at 0.08906 and 0.69017. | Transport has forecast utility in this simple control but cannot replace CNO. Test it as an input to a learned residual predictor. Do not infer from shift correlations alone that a transport architecture will win. |

The shift estimator uses only the 20 observed frames: shared u/v increment correlation at lag two, horizontal shifts −4…4 on identical interior support, and velocity equal to selected shift divided by two. It transports the last observed fluctuation around the history mean. Targets never choose its shift or its damping. Boundary handling is zero fill and the solid mask is derived from history. This is a deliberately restricted transport hypothesis; a failure would not rule out spatially varying flow or subpixel velocity estimation.

Local evidence: [summary](research/score_followup/results/summary.json), [configuration and hashes](research/score_followup/results/config.json), [window identities](research/score_followup/results/window_manifest.csv), [TKE decomposition](research/score_followup/results/cno_error_decomposition.csv), [amplitude sweep](research/score_followup/results/amplitude_sweep.csv), [amplitude folds](research/score_followup/results/amplitude_leave_re_out.csv), [interval folds](research/score_followup/results/interval_leave_re_out.csv), and [causal forecast controls](research/score_followup/results/causal_transport.csv). The optimized SPS calculation was checked against the supplied scorer with agreement within 0.0001 SPS points. A JSON serialization error after computation was corrected and the complete script reran successfully in 51.39 seconds.

### Kaggle experiment: two paired architecture tests

The submitted [experiment implementation](research/score_followup/kaggle_experiment.py) trains eight small models: two bases × two prior inputs × two seeds. Each trainable head has **55,528 parameters**, three 3×3 spatial convolutions, width 32, GroupNorm, GELU, and a zero-initialized final layer. Its input channels contain the observed 20-frame u/v history, a base 20-frame forecast, and a causal prior forecast. The output is a deterministic correction to all 20 future frames. Only the prior changes within each matched pair:

1. **Real-history model:** base = the observed history's temporal mean repeated over the future. Compare a damped stationary fluctuation prior against the same prior with history-estimated transport. This model uses no pretrained weights and its normalization and residual scales are fitted only on its training groups.
2. **Incumbent adapter:** base = the frozen, hash-verified champion CNO point predictions. Compare the same two prior inputs with identical head capacity and training settings. This measures adaptation of the incumbent, with the explicit caveat that its organizer checkpoint and original normalization may have seen held-out conditions. It cannot establish clean transfer generalization.

Both pairs use seeds 42 and 43, 12 epochs, AdamW, learning rate 0.001, weight decay 0.0001, batch size 12, and the same normalized residual MSE loss. Validation selects the epoch by field RelL2; epoch zero is eligible and exactly reproduces each base. Thus a training run is allowed to select no correction. There is no TKE auxiliary loss or noise injection in this experiment, so neither is confounded with the transport input.

The planned data are all 81 nonduplicate Real trajectories, subsampled to 32×64, with window start frames `[0,160,320,480,640,800]` when the full 40-frame block exists. Actual HDF5 Reynolds values determine group splits: test `{6306,13977,24204}`, validation `{10142,20369}`, all other groups for training. All angles from a held Reynolds group remain in the same fold. These groups are fixed before this training run; they are not a pristine final benchmark, since prior exploratory analysis has inspected this dataset. Group-level consistency and a later independent evaluation remain necessary.

Pressure is zero, and the models receive no Reynolds or AoA metadata and no synchronized Sim frames at inference. All architecture comparisons use the incumbent's constant interval half-width. The test report includes per-window errors, SPS/coverage, selected epochs, two seeds, model checkpoints, and saved predictions. Timings separate CNO inference from the head's GPU forward pass; the latter is not an end-to-end submission timing because it excludes prior construction and feature preparation.

A local smoke check passed output shape, exact zero-head baseline reproduction, a nonzero training gradient, finite history-only transport with solid masking, and existence of all chosen Reynolds groups. [prepare_kaggle.py](research/score_followup/prepare_kaggle.py) packages only the experiment and 47 dependency source files; [upload_manifest.json](research/score_followup/upload_manifest.json) hashes the payload. The original notebook and its credentials are not uploaded. The dataset attachment is `phancanh/realpde-competition-data`; no additional code dataset is required. The private job requests a T4 accelerator, has a 3,600-second service limit, and an internal 3,300-second wall-time guard.

### How the experiment changes the decision

Do not promote an architecture based on training loss or one aggregate score. First compare each transport head to its parameter-matched zero-transport head, then compare the adapter to the unmodified champion. A useful candidate should improve held-group field accuracy consistently across seeds, preserve MVPE, and improve or maintain TKE and fixed-band SPS. If field accuracy improves while TKE degrades, record the tradeoff explicitly; a separate matched loss ablation would be needed before attributing a benefit to architecture.

If only the CNO adapter improves, retain it as an incumbent-specific candidate and test it with an independent evaluation before drawing broader claims. If the clean Real-history pair benefits from transport but the CNO pair does not, transport may duplicate information already captured by CNO. If neither transport pair benefits, prioritize a local residual head or a separate temporal-energy loss test according to the measured results; do not expand the warp model automatically. No result here can establish the value of Sim given Real history without a further matched Sim-pretraining comparison.

### Completed Kaggle results and current recommendation

**Kaggle reports COMPLETE.** The experiment used a Tesla T4, PyTorch 2.10.0+cu128, and 341.74 seconds of script wall time (about 5.7 minutes, excluding service provisioning). It processed 477 windows from 81 conditions: 339 training, 58 validation, and 80 test windows. Short trajectories supply fewer than six windows. The test set covers three Reynolds groups. The checkpoint loaded strictly with no missing or unexpected keys and its hash matched the incumbent.

Errors below are better when smaller; SPS is better when larger. Trainable rows average the separately evaluated seed-42 and seed-43 models. **These are averages of scores, not an ensemble forecast**, and they are local held-group results rather than leaderboard scores.

| Predictor on the same 80 test windows | RelL2 error ↓ | TKE error ↓ | MVPE error ↓ | Fixed-band SPS ↑ |
|---|---:|---:|---:|---:|
| History mean | 0.131611 | 1.000000 | 0.120283 | 36.6242 |
| Causal transport, no learned head | 0.116894 | 0.854766 | 0.108596 | 39.2342 |
| Real-history CNN, stationary prior | 0.109182 | 0.887771 | 0.101053 | 39.6781 |
| Real-history CNN, transport prior | **0.103101** | **0.829145** | **0.097401** | **40.7102** |
| Champion CNO | 0.087406 | 0.649071 | 0.082121 | 44.3683 |
| Champion + CNN, stationary prior | **0.086572** | **0.635797** | 0.082100 | **44.6044** |
| Champion + CNN, transport prior | 0.086816 | 0.643143 | **0.082013** | 44.5257 |

For the small Real-history model, transport lowers field error by approximately **5.57%** and TKE error by **6.60%** relative to its matched stationary-prior model, with a 1.0321-point SPS gain. Both seeds improve on all four aggregate metrics. Seed-averaged field, TKE, and MVPE errors improve in each of the three held Reynolds groups. This supports the restricted causal-transport input as useful to this small model under the tested training budget. It does not prove a transport model has reached its optimum or that it can beat the champion: its field and TKE errors remain materially higher than CNO's.

For the incumbent adapter, adding transport gives slightly worse field and TKE errors than the stationary-prior head in both seeds. After averaging seeds, that difference has the same sign in all three held Reynolds groups. Thus the next step should not assume transport is missing from CNO or expand the warp branch on the strength of correlation plots. Transport may duplicate information CNO already represents, or this head may not exploit it effectively; the experiment distinguishes the observed outcome from those unproven explanations.

The stationary-prior CNO head improves field error by approximately **0.95%** and TKE error by **2.05%** over the champion, with a **0.2361-point local SPS gain**. MVPE is almost unchanged: seed 42 improves slightly and seed 43 worsens slightly. These are modest screening gains. They do not justify claiming a new leaderboard best or discarding `sub1_cno_sps.zip`. Both seeds improve field and TKE aggregates; the model-pretraining exposure caveat still applies.

All selected checkpoints were from epochs 10–12 of the 12-epoch budget. This is an initial capacity-controlled screen, not a convergence study. Head GPU forward passes take approximately 0.38–0.46 milliseconds at batch size one on this T4. The measured CNO batch-one path averages 195.76 milliseconds, including normalization, transfers, and its first invocation. These timing definitions differ, and prior construction plus feature preparation are not included in head timing. Measure the complete wrapper before making a time-score claim.

**Experiment proposed after the architecture screen, now completed above:** keep the stationary-prior CNO head architecture and fixed intervals, then compare the existing normalized MSE objective with a matched temporal-energy objective on the same training/validation protocol and at least the same two seeds. The completed loss screen selected lambda zero under its validation safeguards; the separate confirmation branch therefore did not run. The mean-preserving correction described in the latest decision is the next untested proposal. Spatial TKE-map error, temporal diagnostics, and field/MVPE tradeoffs remain necessary; do not change architecture, interval fitting, and loss simultaneously.

For the broader scientific claim, retain the successful small Real-history transport pair as a control and run a separate matched Sim-pretraining experiment with fold-pure Real training. The present champion adapter cannot answer whether Sim helps given Real history. The organizer checkpoint's exact Real training membership and the unpublished final-score combination remain unresolved; neither prevented this experiment, but both limit what its scores can establish. No further clarification is required to run the next bounded ablation.

The private job was launched with `python research/score_followup/kaggle_remote.py push`. Its helper reads the designated notebook cell in memory and excludes credentials from the uploaded payload and logs. Downloaded evidence is under [kaggle_results/architecture_screen](research/score_followup/kaggle_results/architecture_screen): [results](research/score_followup/kaggle_results/architecture_screen/results.json), [runtime and checkpoint identity](research/score_followup/kaggle_results/architecture_screen/runtime.json), [split manifest](research/score_followup/kaggle_results/architecture_screen/manifest.csv), [per-window errors](research/score_followup/kaggle_results/architecture_screen/per_window.csv), and eight trained head checkpoints. [Saved predictions](research/score_followup/kaggle_results/architecture_screen/test_predictions.npz) and [training histories](research/score_followup/kaggle_results/architecture_screen/training.csv) accompany those outputs. The raw user notebook was not executed or uploaded.

Post-download verification passed: all nine saved predictor arrays (champion plus eight trained heads) have finite predictions and zero pressure; their recomputed RelL2/TKE/MVPE agree with reported values within 0.000001; SPS agrees within 0.0001 points; all 96 training-epoch records are present; no Reynolds group crosses folds; and the executed source hash matches the uploaded payload. [analyze_kaggle.py](research/score_followup/analyze_kaggle.py) reproduces these checks and the [paired comparisons](research/score_followup/paired_comparisons.csv), [per-Reynolds differences](research/score_followup/paired_by_Re.csv), and [analysis/verification JSON](research/score_followup/kaggle_analysis.json). The paired table confirms that the Real-history transport head improves field, TKE, and MVPE in all three Reynolds groups **in each seed**. The stationary-prior CNO head improves field and TKE over the champion in all three groups in each seed. Three groups and two seeds remain a pilot; windows from the same trajectory are not independent replicates, so no significance claim is made.




## Nhận định về hướng tìm breakthrough sau tám lượt huấn luyện

**Hướng suy luận này hợp lý để chọn giả thuyết tiếp theo: CNO có sai số trường thấp nhưng chưa tái tạo tốt bản đồ năng lượng dao động trong cửa sổ dự báo. Cần kiểm tra xem phần thiếu hụt đó có thể học thêm từ lịch sử hay không.** Chưa đủ bằng chứng để khẳng định toàn bộ residual nằm ở thành phần dao động, hoặc đã xác định được bottleneck chính của CNO.

RelL2 tốt và MVPE tốt hỗ trợ nhận định CNO dự báo trường và một số đại lượng trung bình tốt. Tuy nhiên, chưa có phép phân rã sai số theo trung bình/dao động và thang không gian của chính dự đoán CNO để chứng minh rằng model đã học tốt mọi coarse dynamics. Với mỗi cửa sổ và mỗi pixel, có thể kiểm tra trực tiếp đẳng thức:

\[
\frac{1}{T}\sum_t\|\hat y_t-y_t\|^2
=\|\bar{\hat y}-\bar y\|^2
+\frac{1}{T}\sum_t\|(\hat y_t-\bar{\hat y})-(y_t-\bar y)\|^2.
\]

Đây là phân rã MSE thành sai số trung bình và sai số dao động, không phải phân rã cộng trực tiếp các điểm RelL2 đã chuẩn hóa. Tỷ lệ năng lượng CNO/target trung vị 0.5071 và tương quan bản đồ TKE trung vị 0.7523 cho thấy thiếu năng lượng đi kèm sai khác phân bố không gian. Chúng chưa xác định phần sai khác nào có thể dự đoán từ 20 frame đầu. Thất bại của global amplitude boost chỉ áp dụng cho grid đã thử và giới hạn tăng field error 2%; nó không loại trừ mọi hiệu chỉnh biên độ phụ thuộc đầu vào.

Transport đã có bằng chứng thực nghiệm về khả năng cải thiện dự báo của mô hình Real-history nhỏ: cả hai seed cải thiện field/TKE/MVPE trên cả ba Reynolds group. Cách diễn đạt chính xác là **transport prior đã được kiểm chứng là hữu ích trong các đối chứng này**. Đây chưa phải xác nhận duy nhất về cơ chế vật lý. Việc prior không giúp adapter CNO chỉ cho thấy phép bổ sung transport cụ thể này chưa tạo lợi ích thêm; chưa chứng minh CNO không còn lỗi transport ở thang khác hoặc không gian khác. Tương tự, bằng chứng Real có thành phần nhanh hơn Sim là mô tả dữ liệu; nó không tự chứng minh thành phần nhanh đó là residual có thể dự báo của CNO. Các so sánh rộng Real–Sim vẫn chủ yếu dựa vào output notebook; lần tái lập độc lập trước đó chỉ có một cặp điều kiện.

**Giới hạn quan trọng của thí nghiệm loss:** scorer tính TKE tại mỗi pixel bằng một nửa tổng phương sai u/v qua toàn bộ 20 frame tương lai. Hoán vị thứ tự các frame dự đoán giữ nguyên TKE. Vì vậy, loss TKE kiểm tra việc khớp bản đồ năng lượng dao động trong cửa sổ; bản thân nó không bảo đảm đúng pha, tần số, thứ tự thời gian, hoặc vị trí sai số theo horizon. Giữ field loss trong `L = L_field + lambda * L_TKE` sẽ tiếp tục ràng buộc dự đoán theo từng frame, nhưng phải đo kết quả mới biết hai mục tiêu có tương thích hay không.

Thí nghiệm tiếp theo vẫn nên giữ nguyên stationary-prior CNO head, checkpoint base, intervals và ngân sách huấn luyện; dùng `lambda = 0` làm đối chứng, chọn lambda trên validation. `L_TKE` cần so sánh **bản đồ TKE**, không chỉ tổng năng lượng toàn miền. Ngoài các metric hiện có, cần báo cáo phân rã mean/fluctuation MSE, lỗi theo horizon và một kiểm tra phụ thuộc thứ tự thời gian như increment error hoặc ACF. Nếu cần kết luận về thang không gian/tần số, bổ sung phân rã phổ tương ứng; TKE đơn lẻ không đủ.

Một kết quả đáng theo là TKE-map error giảm đồng thời field/MVPE được bảo toàn và cấu trúc thời gian cải thiện qua seed và nhóm điều kiện. Nếu TKE tăng điểm nhưng phase/field xấu đi, kết quả mới chỉ cho thấy đánh đổi giữa khớp năng lượng và dự báo quỹ đạo. Khi đó cần kiểm tra mức độ bất định của residual trước khi kết luận rằng một deterministic head lớn hơn sẽ giải quyết được. **Insight hiện tại là sự khác biệt giữa dự báo trường chính xác và tái tạo dao động chính xác; khả năng khai thác khác biệt đó để tạo bước tiến lớn vẫn là giả thuyết.** Phần này làm rõ cách diễn giải kết quả đã có; chưa khởi chạy thêm thực nghiệm.

## Initial audit decision (2026-09-05; refined by the experiments above)

**Choose E: calibrated statistical structure plus Real-history dynamics, with a small causal transport ablation as the next modeling test.** Spatial transport now has broad held-out support in the available Real data. Temporal acceleration helps a held-out ACF fit on the one available matched condition, but its fitted scale depends strongly on lag range; there is no evidence for a universal alpha. Mean/RMS normalization does not close the dynamics gap and can amplify quiet, fast-varying pixels. High-frequency residual modeling remains relevant, but a stochastic model is premature until deterministic transport and normalization controls have been evaluated.

No new evidence here establishes that Sim improves forecasting **given Real history**. That requires the controlled B versus C/D comparisons below. Statistical calibration has strong saved-notebook evidence, whereas its full condition-level reproduction remains limited by missing Sim data.

## Scope and evidence provenance

The supplied instructions reference both `hypothesis-tree.ipynb` and `explore-data.ipynb`. Only `explore-data.ipynb` is present. Its 21 cells were inspected in stored order, including their code and saved numerical outputs. Execution counts reset several times; cells 13 and 14 are identical; cell 20 has `execution_count=null` and no output. Thus this is not a verified clean-kernel execution record. The original Kaggle paths are absent locally. The three requested diagnostics were implemented as standalone Python functions and executed against available local data instead of overwriting the notebook or pretending to rerun the unavailable full dataset.

There are two different evidence levels:

- **Saved notebook evidence:** 82 Real trajectories and 100 Sim trajectories scanned; subsequent broad pairwise diagnostics use 82 pairs matched by filename. The raw Sim data needed to independently reproduce those broad results are mostly absent locally.
- **New local evidence:** 81 continuous Real trajectory caches at 32×64, 12 native Real HDF5 files at 64×128, and one native Sim HDF5 file. Native files overlap the caches and are resolution checks, not extra independent conditions. The only newly executable matched domain comparison is nominal `5025_5`: actual Real Re=5028, Sim Re=5025, AoA=5. A single condition cannot establish condition dependence or generalization of a Real–Sim time-scale map.

The cache builder explicitly excludes `7575_0.h5` and describes it as a duplicate; the notebook's 82-pair analyses do not make that exclusion. The missing duplicate's raw bytes are not available here for independent verification. Cache continuity and agreement with available raw fields are checked in the new run. No unobserved Sim trajectories were invented or replaced by the single available Sim condition.

## What the supplied baseline actually predicts

| Interface or component | Verified behavior and consequence |
|---|---|
| Data/window loader | `CCCCCC/src/build_cache.py` reads native `[T,64,128]` u/v, samples `[:,::2,::2]`, stacks `[u,v,p]`, and sets absent Real pressure to zero. It takes 20 input frames followed by 20 target frames, stride 20. Adjacent cached windows share raw frames even though their targets do not overlap. Split by condition or raw-frame intervals before constructing windows. |
| Input and output | Both are `[N,20,32,64,3]`, in `[u,v,p]` order. The target is the **next 20 Real frames**. There is no synchronized Sim target in this interface. |
| Inference | The kit's `submission_example_fno.py` calls the model once per batch and returns the complete 20-frame future block. It does not autoregressively feed predicted frames back into a one-step model. Changing to one-step delta prediction would change temporal formulation and rollout behavior. |
| Normalization | The example uses channelwise Gaussian input means `[0.154960856,-0.000513992854,0]`, stds `[0.0968056545,0.015960684,1]`; target means `[0.154962569,-0.000517793698,0]`, stds `[0.0968104079,0.0159636438,1]`. These are not per-condition temporal mean/RMS maps. Decode with target statistics and force output p=0. |
| Fold purity | The supplied constants are described as fitted on all train_real. They reproduce the organizer wrapper, but they do not establish fold-pure normalization for a new leave-condition-out experiment. Refit learned preprocessing on each outer training fold. Organizer Sim+Real fine-tuned checkpoints may already have seen a proposed local validation condition; use Sim-only pretraining or retrain for a clean causal B/C/D comparison. |
| FNO | `FNO3d` mixes time and space through 3D Fourier layers, adds pointwise Conv3d branches, BatchNorm, and coordinate grids, and projects to a future block. The checked configuration has width 64, four layers, Fourier modes `(4,12,16)`. The existing pointwise branch is not a spatial local convolution branch. |
| CNO | `CNO3d` permutes to channel-first, applies lifting, multiscale encoder/decoder, residual blocks, skip connections, and projection, then restores channel-last output. Kit construction uses three encoder/decoder layers. |
| Transolver | The kit constructs three blocks, hidden width 256, eight heads, 16 slices. Its forward flattens the input field volume and returns the same time/spatial layout. `fun_dim=0`, `space_dim=3`, `fx=None` means the three supplied velocity/pressure features enter preprocessing; the parameter name is not evidence that physical coordinates are supplied. No local Transolver checkpoint was found, so its claimed checkpoint compatibility is not newly verified. Preserve/check the flatten-to-structured-mesh axis convention before any geometry change. |
| Loss | FNO/CNO `train_loss` use MSE. Transolver's method returns elementwise squared error; an outer loop must reduce it. The existing project fine-tuning loop in `CCCCCC/src/train.py` bypasses these methods and uses physical-space Charbonnier plus positive TKE, spatial spectrum, and vorticity penalties, with AdamW. It is already more than an MSE-only baseline. Its `avg4` model-selection heuristic is not a published official final-score formula. |
| Existing project variant | `work/submission.py` has different normalization and constant uncertainty bands and expects its own packed `model.pth`. It is a separate wrapper, not interchangeable with the organizer normalization/checkpoints. |
| Submission condition information | `submission_template.py` explicitly says scored calls receive an empty metadata dictionary. Re, AoA, timestamps, and matching raw Sim frames are **not guaranteed inference inputs**. A condition-indexed oracle is not deployable as written. Infer a latent condition/statistical prior from Real history or train/distill a history-to-prior mapping; evaluate the inferred version. |

The authoritative local sources for these statements are the files under `realpde_t1_starting_kit_v9/realpde_t1_starting_kit_v9/`, `CCCCCC/src/`, and `work/submission.py`. This report audits the supplied kit version; it does not certify that current remote competition rules are unchanged. A hard submission-size or runtime limit was not newly established from an authoritative current source, so no unverified numeric cap is assumed.

## Notebook hypothesis audit

Cell numbers below are **zero-based notebook indices**, not execution counts. Values in this section are saved outputs, not new full-dataset reproductions. Rel-L2 denotes an error, with lower values better.

| Hypothesis | Exact test and scope | Saved numeric output | Evidence and caveat | Modeling consequence |
|---|---|---|---|---|
| Real/Sim interfaces and conditions agree | Cells 2–4: HDF5 shapes, scalar Re/AoA, spatial grids, timestamps; 82 Real and 100 Sim files, 82 filename pairs | Real lengths 282/492/607/868; Sim length 1000. Only 4 exact scalar-Re/AoA matches. Maximum coordinate difference x=1.0833e-5, y=1.9970e-5. dt≈0.020000 vs 0.02002002. | Approximate spatial/condition correspondence supported; not exact physical identity. Cell 2 samples at most 64 time positions, so its `lag1_corr_sampled` is not adjacent-frame autocorrelation. Cell 3's hypothetical 10→1 window counts are not the 20→20 submission task. | Preserve both nominal filename condition and measured Re; use stored time grids. |
| Timestamp interpolation or constant lag restores instantaneous pairing | Cell 5: linear Sim interpolation to Real time, then lag search −100…100 in steps of 5; `3750_0`, `8850_10`, `13950_15`, `20325_20`, `26700_10` | Fluctuation corr u: −0.0294…0.0010; v: −0.0272…0.0206. Selected lags +15,+70,+90,+25,−35 frames. | Does not support synchronized pairing on the five tested cases. Search/evaluation share data, global correlation contains static means, and no exhaustive nonlinear alignment was tested. It is not proof that all possible alignments fail. | Do not train raw Sim(t)→Real(t) as if phase-aligned. |
| Statistics transfer better than phase | Cell 5: spatial correlations of temporal mean and RMS maps on the same five pairs | Mean-u corr 0.3836–0.6604; mean-v 0.3308–0.7214; RMS-u 0.4753–0.7847; RMS-v 0.7643–0.9199. | Supported, with weaker mean-v and low-Re cases. Entire trajectories estimate these statistics; they are not available from future Real data at inference. | Sim statistical priors merit a controlled forecasting ablation. |
| Affine calibration explains much of amplitude mismatch | Cells 6–7: least-squares `RealMap=a*SimMap+b`; five selected pairs and then all 82 | Per-file oracle averages from cell 8: mean-u 0.3758; mean-v 0.7908; RMS-u 0.5168; RMS-v 0.3790 after calibration. | Oracle fits are an achievable in-sample reference, not generalization. Cell 7's parameter R² is also in-sample; its separate CV parameter MAE is not itself a map-accuracy score. | Use simple supervised calibration before a complex network; retain nonnegative RMS and mask handling. |
| Calibration transfers across Re | Cell 8: leave one measured Re out; fit eight affine coefficients from `[Re/30000,AoA/20]` using other Re values; reconstruct held-out maps | Mean-u raw→CV: 6.3000→0.3862; mean-v 18.3170→0.8528; RMS-u 12.9799→0.6272; RMS-v 22.1899→0.4857. All 82 files improve versus raw Sim. | Code implements genuine held-Re coefficient prediction. However, all AoAs can remain in training, model choice was explored on the same dataset, one duplicate may remain, and comparison is against uncalibrated Sim with enormous scale error. No condition-only or Real-history-only comparator establishes incremental Sim value. | A is promising; reproduce without the duplicate and compare condition/history-only statistical predictors before claiming Sim helps. |
| TKE differs only by scale | Cell 9: full-trajectory `0.5*(var_t(u)+var_t(v))`, all 82 pairs; scale/affine fits | Mean spatial corr 0.6667, range 0.1359–0.8786. Raw TKE Rel-L2 281.3468; oracle scale 0.6284; oracle affine 0.6184. | Large amplitude mismatch plus substantial remaining shape mismatch. This is a 2D resolved TKE proxy, not full 3D turbulent energy. TKE calibration was not condition-CV tested here. | Do not declare TKE solved by scalar calibration. Audit velocity units/nondimensionalization before interpreting enormous raw errors physically. |
| Mask/boundary pixels explain the TKE gap | Cells 10–11: six illustrative TKE maps; all-pair dilation 0,1,2,4,8 of near-zero-mean u/v masks | Excluding about 3.71%→15.71% of pixels changes mean scale-fit error only 0.6284→0.6167 and corr 0.6617→0.6611. | Mask-only explanation not supported by this proxy. Zero mean is a heuristic for a solid/invalid mask and can miss PIV artifacts or exclude fluid points. Fits are oracle. | Broad flow statistics need attention beyond boundary cleanup. |
| Real and Sim temporal dynamics agree after amplitude removal | Cell 12: per-pixel normalized correlation, spatial stride 2, physical lag conversion, five shuffled nulls; 82 pairs | At lag≈0.1: Real/Sim u=0.2734/0.8077, v=0.2073/0.7618. At ≈0.02: u=0.8659/0.9882, v=0.8710/0.9847. Nulls near zero. | Dynamics gap supported and **already partially amplitude-invariant**: per-pixel Pearson correlation is unchanged by positive affine rescaling. The requested new normalized experiment must also inspect pooled metrics/PSD, rather than claim the notebook never normalized dynamics. | Statistical calibration alone cannot solve temporal mismatch. |
| Smoothing reconciles slow dynamics | Cells 13–14: moving averages of width 1,3,5,9,15 applied to **both** domains; ACF curves divided by their lag-1 values | Normalized-curve discrepancy w=1→15: u 0.2750→0.0990; v 0.2893→0.1156. | Supported descriptively. It is not a test of smoothing only Real to match untouched Sim. Repeated cells are not independent replications. Smoothing mechanically alters ACF. | A slow/fast representation is plausible; do not infer measurement noise from smoothing alone. |
| Real has a stronger fast component | Cell 15: centered moving-average residual, 82 pairs, widths 5/9/15 | At width 9, fast variance/original variance Real/Sim: u=0.2116/0.0406, v=0.2428/0.0561. | Supported for this filter. Slow/fast components are correlated, so their variances need not add to the original variance. Centered filters use future samples and are diagnostic only. | A deployable slow/fast predictor must use causal history features. |
| Real PSD contains more medium/high frequency energy | Cell 16: Welch, up to 256 samples, spatial stride 2, full-trajectory mean removal | 0.25–0.50 Nyquist band Real/Sim: u=0.05113/0.00658, v=0.06543/0.00880. Spectral shape L1 averages u=0.7290, v=0.9441. | Supported as a descriptive comparison. This averages dimensional PSD before normalizing total area, so high-variance pixels dominate. Separately trapezoiding disjoint bands omits bin-edge intervals; listed fractions do not exactly sum to one. | Repeat with per-pixel RMS normalization and common physical frequencies; retain spectral checks. |
| Real increments lose memory faster | Cell 17: first differences; equal-pixel temporal correlation and global variance ratio, 82 pairs | Lag-1 increment corr Real/Sim: u=0.2420/0.8953, v=0.3552/0.8916. Increment/signal variance ratio: u=0.1724/0.0356, v=0.1757/0.0493. | Supported. Adjacent increments share a frame; differencing white observation noise itself induces negative lag-1 correlation, so this alone cannot identify a physical mechanism. | Measure incremental predictability, not amplitude alone. |
| Fast variation is independent pixel noise | Cell 18: flattened increment correlation at native distances 1,2,4,8, 82 pairs | Nearest-neighbor mean correlation Real u=0.7263, v=0.8855; Sim u=0.8416, v=0.8903. | Independent pixelwise white noise as the sole explanation is not supported. Coherent physics and spatially correlated PIV processing remain possible. | Do not add independent noise merely to inflate TKE. |
| A fixed spatial shift exposes coherent transport | Cell 19: search `(di,dj)∈[-4,4]²`, increment lags 1/2, every other temporal pair; 82 pairs | In-sample gains Real u=0.1091/0.2184, v=0.1396/0.3729. Sim selects `(0,1)` at lag 1 and `(0,2)` at lag 2 in all 82 cases. | Promising but oracle: same samples select/evaluate shift, and support changes with shift. Many Real lag-2 optima hit +4, so range censoring is plausible. | Require held-out validation before a warp model. |
| Learned shifts generalize across halves | Cell 20: intended two-way temporal-half validation | No saved output; never marked executed. | Unresolved in the notebook. Splitting base increment indices still overlaps raw frames around the boundary because each evaluated pair uses frames through `t+lag+1`. | The new experiment splits raw frames first, drops a guard interval, and fixes support across candidates. |

Additional correction to existing prose: a 0.1% dt difference produces only about 0.01736 stored time units of drift over 867 intervals. The 17.42 versus 20.08 end-time difference primarily reflects **868 versus 1000 samples**, not 2.66 units of timestamp drift accumulated over the same 868 samples.

## New experiments: methods

### 1. Cross-validated spatial shift

Hypothesis: transport visible in increments persists in an unseen time block, instead of being an artifact of selecting the best shift on the evaluation samples.

For every available cache/native trajectory and each u/v channel, split **raw frames** around the midpoint and remove a 20-frame guard interval (approximately 0.4 stored time units). Compute increments independently inside each block. Search integer `(di,dj)` from −4 to +4 at lags 1 and 2 using A only, evaluate zero and the chosen shift on B, then reverse. Positive `(0,dj)` means comparison of `dx(t,i,j)` with `dx(t+lag,i,j+dj)`; it is a displacement in this array convention.

A training-only variance threshold excludes effectively constant pixels. Eroding that mask by a square of radius four provides one fixed anchor support for every candidate. There is no circular wrap, and zero versus shifted evaluation uses the same anchor pixels and pair count. Correlations pool time and space and therefore weight energetic regions more than equal-pixel correlations. FFT cross-products were checked against direct Pearson calculation on all shifts of a synthetic field, including known shift recovery.

Saved rows include file identity, actual Re/AoA, resolution, train/test frame ranges, optimal training shift, training correlation, test zero/shift correlation, gain, boundary optimum, and agreement between the two learned shifts. Aggregate conditions equally; do not treat millions of pixels or two complementary folds as independent observations. This is within-trajectory generalization, not leave-condition-out transport estimation. An increment-correlation gain is not automatically a field-forecast improvement.

### 2. Mean/RMS-normalized dynamics

Hypothesis: static mean and amplitude differences explain the domain dynamics gap. Training-half temporal means and population RMS maps define `z=(x-mu)/(sigma+epsilon)` on the other half. Epsilon is `max(1e-8, 0.001*median(positive training RMS))`; near-constant pixels are omitted. Reverse the halves. Report the dimensional fluctuation field and normalized field on identical valid pixels.

Compute pooled Pearson ACF and mean per-pixel Pearson ACF; exact centered overlap moments use an FFT numerator. Also compute increment variance and its signal-variance ratio, increment ACF through lag 10, and horizontal/vertical nearest-neighbor increment correlations. Welch uses a Hann window, constant detrending, up to 256 contiguous samples and 50% overlap. PSD bin masses sum to one; band edges do not drop gaps. All time steps are contiguous: no temporal decimation is used to accelerate these diagnostics.

The direct paired analysis uses 32×64 data, the first 860 samples of nominal `5025_5` in each domain, dt=0.02 for Real and the HDF5-derived Sim dt. Truncating the Sim record equalizes sample counts and nearly equalizes duration; it does not assert synchronization. Native Sim shift checks additionally use its full 1000 frames. Full Real-cache results describe Real condition variation; only the actually matched condition supports new Real–Sim comparisons.

Per-pixel Pearson correlation is invariant to a fixed positive scale and offset, so an unchanged per-pixel ACF is expected algebraically. Changes in pooled ACF or average PSD after normalization reflect spatial reweighting as well as amplitude removal. This experiment is a diagnostic; full-half Real statistics are not a deployable forecast input.

### 3. Temporal-scale fitting

Hypothesis: `C_real(tau)≈C_sim(alpha*tau)` generalizes across time blocks. Alpha>1 means Real traverses the Sim correlation curve faster. Estimate a curve in each block independently; select alpha on one block and evaluate the frozen alpha on the other, then reverse. Use the actual domain dt and interpolate the Sim ACF without extrapolation.

The predeclared search is 0.25–10 on a 301-point logarithmic grid plus alpha=1. Evaluate two fixed lag ranges, `(0,0.1]` and `(0,0.4]`, as sensitivity analyses, not a test-selected choice. Record held-out MSE at alpha=1 and learned alpha, boundary optima, and the near-optimal training range. For alpha fitting, normalize each block using **that block alone**; using the opposite block's RMS to build a training curve would leak test information. These descriptive curve estimates are distinct from experiment 2's train-to-test normalization.

With only one matched Sim condition, Re/AoA regression of alpha and leave-condition-out alpha transfer are not identifiable. No region-specific alpha is fitted: the current sample coverage does not justify another selection dimension.

## New results and their implications

### Experiment 1 result: transport survives held-out evaluation

The full run produced **752 shift-CV rows**: 81 cached Real trajectories × 2 channels × 2 lags × 2 directions, plus 12 native Real and one native Sim trajectory with the same tests. All 12 cache/raw overlaps match exactly after float32 conversion and spatial subsampling; cached windows reconstruct contiguous frames without disagreement. The retained cached lengths are 280/480/600/860 frames; incomplete final windows account for the omitted native tail. Native tests retain the complete raw records.

The table averages conditions equally. The 95% intervals resample whole conditions, averaging the two folds within a condition first (2,000 bootstrap draws, seed 42). They describe this condition set; correlated conditions at the same Re and the absence of independent experimental replicates limit population-level inference.

| Domain/resolution | Channel | Lag | Conditions | Test zero corr | Test shifted corr | Gain | 95% condition bootstrap interval | Positive test folds | Same shift across halves |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| Real 32×64 | u | 1 | 81 | 0.44415 | 0.54096 | 0.09681 | [0.08354, 0.10957] | 85.19% | 100.00% |
| Real 32×64 | u | 2 | 81 | 0.21368 | 0.48276 | 0.26909 | [0.25058, 0.28713] | 100.00% | 98.77% |
| Real 32×64 | v | 1 | 81 | 0.65628 | 0.78265 | 0.12637 | [0.10125, 0.15265] | 79.01% | 97.53% |
| Real 32×64 | v | 2 | 81 | 0.32958 | 0.74952 | 0.41994 | [0.36352, 0.48565] | 100.00% | 98.77% |
| Real 64×128 | u | 1 | 12 | 0.40928 | 0.51614 | 0.10685 | [0.09019, 0.12125] | 100.00% | 100.00% |
| Real 64×128 | u | 2 | 12 | 0.19736 | 0.43637 | 0.23901 | [0.21188, 0.26686] | 100.00% | 100.00% |
| Real 64×128 | v | 1 | 12 | 0.66584 | 0.78368 | 0.11784 | [0.07914, 0.16026] | 100.00% | 100.00% |
| Real 64×128 | v | 2 | 12 | 0.38251 | 0.73422 | 0.35171 | [0.26044, 0.45286] | 100.00% | 100.00% |
| Sim 64×128 | u | 1 | 1 | 0.85781 | 0.96388 | 0.10607 | Not estimable across conditions | 100.00% | 100.00% |
| Sim 64×128 | u | 2 | 1 | 0.59184 | 0.91847 | 0.32663 | Not estimable across conditions | 100.00% | 100.00% |
| Sim 64×128 | v | 1 | 1 | 0.84639 | 0.96752 | 0.12113 | Not estimable across conditions | 100.00% | 100.00% |
| Sim 64×128 | v | 2 | 1 | 0.56577 | 0.93271 | 0.36695 | Not estimable across conditions | 100.00% | 100.00% |

At 32×64, all selected shifts have `di=0`. For u lag 1, `dj=0/1/2` occurs in 24/114/24 folds; for v it occurs in 33/110/19 folds. Thus a native one-pixel displacement can be below the coarse grid's integer resolution. A zero shift can be the correct coarse-grid choice; it produces zero gain by construction. At lag 2, most coarse shifts are +1 to +3, with six u and four v folds at the +4 boundary. The native Sim condition selects +1 then +2 in both channels and both halves, reproducing the direction pattern in the notebook.

Native Real lag-2 searches hit the boundary in **50.0% of u folds and 41.7% of v folds**. Consequently the native magnitude is censored; its apparent stability cannot prove the true optimum is exactly four pixels. A future radius/fractional-shift sensitivity test should choose its search settings on training data. Native and coarse results differ in support, spacing, and temporal tails, so their numeric gains are not a pure resolution ablation.

Condition dependence is visible. At 32×64, mean lag-2 gain for AoA 0 versus AoA 20 is u=0.3169 versus 0.2250 and v=0.6996 versus 0.2393. Pearson correlation between Re and mean learned `dj` is u=0.780/0.937 and v=0.802/0.921 at lag 1/2. These are descriptive trends with confounded and incomplete Re/AoA coverage, not a validated Re→transport predictor. Full per-Re/AoA tables and every fold's shift are saved.

**Verdict:** supports reproducible transport in energetic Real increments and rejects the idea that the entire earlier gain was caused by the same-sample shift search. **Decision change:** C now warrants a small causal transport experiment. **Remaining weakness:** shifts were estimated using hundreds of training frames, whereas submission history contains only 20; no leave-condition-out or 20-frame online shift estimator was tested. Correlated PIV processing is still compatible with a transported pattern. **Next highest-value test:** estimate a shared u/v transport from only the available 20 history frames and compare held-out 20-frame forecasts with and without that transport, using the same residual predictor and losses.

![Held-out spatial shift gains](/D:/Project/NeurIPS/research/architecture_audit/results/shift_cv_gain_by_condition.png)

### Experiment 2 result: normalized dynamics still differ substantially

New paired results below are for **one condition only**, nominal `5025_5`, averaged over the two test halves at 32×64. “Fluctuation” subtracts the training temporal mean without per-pixel RMS scaling. ACF and increment ACF in this table are pooled correlations. High-frequency mass denotes frequencies ≥0.25 of that domain's Nyquist frequency.

| Channel/domain | Representation | Lag-1 ACF | Increment/signal variance | Lag-1 increment ACF | Increment spatial corr x / y | High-frequency PSD mass |
|---|---|---:|---:|---:|---|---:|
| u Real | Fluctuation | 0.94420 | 0.11161 | 0.31578 | 0.17541 / 0.18986 | 0.04279 |
| u Real | Normalized | 0.85015 | 0.30011 | 0.18616 | 0.06080 / 0.08561 | 0.12980 |
| u Sim | Fluctuation | 0.99119 | 0.01761 | 0.85267 | 0.69181 / 0.56838 | 0.00245 |
| u Sim | Normalized | 0.99191 | 0.01618 | 0.84827 | 0.70150 / 0.63683 | 0.00250 |
| v Real | Fluctuation | 0.97772 | 0.04456 | 0.57755 | 0.48047 / 0.59355 | 0.01048 |
| v Real | Normalized | 0.76941 | 0.46204 | −0.00963 | 0.06415 / 0.22452 | 0.16596 |
| v Sim | Fluctuation | 0.98428 | 0.03143 | 0.84687 | 0.64881 / 0.82104 | 0.00478 |
| v Sim | Normalized | 0.98984 | 0.02032 | 0.83821 | 0.64678 / 0.82131 | 0.00346 |

| Channel | Representation | Real–Sim pooled ACF RMSE over lags 0.02–0.40 | Normalized PSD L1 on common physical-frequency grid |
|---|---|---:|---:|
| u | Fluctuation | 0.36991 | 0.95533 |
| u | Normalized | 0.54863 | 1.21185 |
| v | Fluctuation | 0.24923 | 0.79549 |
| v | Normalized | 0.55130 | 1.19010 |

Normalization **increases**, rather than closes, these pooled gaps. The normalized Real increment/signal variance ratio remains about 18.5× Sim for u and 22.7× for v. Equal-pixel ACF and increment-ACF columns are also saved; their affine invariance means they should not be interpreted as independent normalization gains.

This does not prove that every fast Real component is coherent transport. The nearest-neighbor correlations become much smaller when quiet pixels receive greater weight, especially at the coarser grid spacing. Energetic transported variation and low-amplitude, weakly coherent variation can coexist. The PSD plot also shows a pronounced Real dip near 12.5 inverse stored-time units in both channels; its cause is unresolved and should be checked against PIV acquisition/preprocessing before labeling the extra band as physical turbulence or noise.

**RMS-floor sensitivity:** increasing epsilon from 0.001 to 0.1 times median positive training RMS, while keeping the valid-pixel set fixed, changes normalized Real lag-1 ACF from u=0.8502→0.8808 and v=0.7694→0.9013. Real increment ratios decrease to 0.2384/0.1974; Sim remains 0.01834/0.02290. High-frequency mass remains Real=0.1089/0.0786 versus Sim=0.00289/0.00391. Magnitudes, especially v, are normalization-sensitive; the domain gap survives this 100× floor change. The tested floors are sensitivity analyses, not a selected model hyperparameter.

**Verdict:** rejects mean/amplitude differences as a sufficient explanation on this pair; supports a remaining dynamics/spectral mismatch. **Decision change:** keep Real history as the source of instantaneous state and use a robust RMS floor in any residual formulation. A does not replace temporal adaptation. **Next highest-value test:** the causal transport control above, plus region/energy-stratified residual diagnostics after that control. Do not assume a stochastic high-frequency generator is necessary merely because unconditional PSD differs.

![Normalized dynamics and temporal spectra](/D:/Project/NeurIPS/research/architecture_audit/results/normalized_dynamics_acf_psd_5025_5.png)

### Experiment 3 result: alpha helps, but a single scale is not an adequate dynamics model

For alpha alone, each evaluated block is independently standardized using its own mean/RMS (RMS threshold `max(1e-8,0.001*median positive RMS)`, division by RMS+1e-8). This avoids borrowing a held-out RMS map when estimating the training curve. This estimand differs from experiment 2's transfer of training RMS to the test block. No alpha uses its test-block error to select a shift, lag range, or value.

The directions below refer to the actual raw time blocks. In the CSV, `test_curve_fold=A_to_B` identifies the curve measured on B, hence an alpha evaluation A→B; `B_to_A` identifies evaluation on A.

| Channel | Fit lag range | Train→test | Learned alpha | Test ACF MSE, alpha=1 | Test ACF MSE, learned alpha | Test MSE reduction |
|---|---|---|---:|---:|---:|---:|
| u | ≤0.10 | A→B | 6.2672 | 0.167930 | 0.004424 | 97.37% |
| u | ≤0.10 | B→A | 5.2761 | 0.213840 | 0.005779 | 97.30% |
| u | ≤0.40 | A→B | 3.0714 | 0.210985 | 0.035739 | 83.06% |
| u | ≤0.40 | B→A | 2.6829 | 0.303690 | 0.045554 | 85.00% |
| v | ≤0.10 | A→B | 4.0256 | 0.116197 | 0.001027 | 99.12% |
| v | ≤0.10 | B→A | 3.8323 | 0.135565 | 0.000627 | 99.54% |
| v | ≤0.40 | A→B | 3.1479 | 0.180463 | 0.013904 | 92.30% |
| v | ≤0.40 | B→A | 2.8181 | 0.227817 | 0.008641 | 96.21% |

All eight fits improve held-out ACF error and none hits the alpha search boundary. Within a lag range, the two estimates differ by about 17.2%/13.5% of their average for u at 0.10/0.40 and 4.9%/11.1% for v. However, changing the fixed lag range approximately halves the u scale: 5.28–6.27 at short lags versus 2.68–3.07 over the forecast horizon. The v ranges are 3.83–4.03 versus 2.82–3.15. The narrow near-optimal training intervals in the CSV are objective-profile sensitivity intervals, not statistical confidence intervals.

The scaled Sim curve still develops negative correlations at long lags that do not match Real well, especially for u. A substantial MSE reduction is therefore compatible with a wrong curve shape. The 0.1% timestamp-step difference is far too small to explain factors of roughly 3–6 in correlation decay; alpha reflects the behavior of these statistical curves, not a recovered clock offset. The short-lag fit has only five nonzero lag points and the longer fit only twenty, with correlated errors.

**Verdict:** supports B as an inexpensive time-scale adaptation hypothesis for this condition; rejects treating a single fitted alpha as a complete explanation of its dynamics. Re/AoA dependence and generalization remain **unresolved**, since only one matched Sim condition is available. **Decision change:** test fixed training-selected temporal resampling/augmentation of Sim pretraining when more Sim data are available; do not hard-code alpha≈3 or infer an instantaneous Real–Sim synchronization. **Next highest-value alpha test:** replicate the same blocked protocol on additional Sim conditions, then fit a simple alpha predictor with leave-Re-out validation and compare ACF and PSD, not just the fitted ACF loss.

![Held-out time-scaled ACF curves](/D:/Project/NeurIPS/research/architecture_audit/results/alpha_heldout_acf_5025_5.png)

### Forecasting control and checkpoint checks

Persistence was executed on **1,672 disjoint 40-frame windows across 81 conditions**, with 20 history frames and 20 forecast frames, at 32×64. It copies the last observed Real frame at every future step. Condition-equal averages are:

| Metric | Persistence result |
|---|---:|
| Field relative L2 | 0.13667 |
| Temporal mean-field relative L2 | 0.10740 |
| RMS/fluctuation error | 1.00000 |
| TKE-map relative L2 | 1.00000 |
| MVPE relative L2 | 0.13665 |
| SPS, kit default ±5% prediction band, mean of per-condition scores | 16.96 / 100 |
| Forecast construction time per window, CPU | ≈0.000268 s |
| Checkpoint size | 0 bytes |

Persistence's forecast is constant through time, so its forecast-window RMS and TKE are zero; a finite TKE-map correlation, temporal ACF, and normalized temporal PSD are undefined. They are not reported as successful zeros. The modest raw-field error and complete fluctuation loss are a concrete demonstration that pointwise accuracy alone is an insufficient selection criterion. These numbers are descriptive local controls, not a held-out competition score. The SPS value is an average of per-condition scores, not a single concatenated official scoring run. Construction timing excludes loading and metric computation.

Local checkpoints were inspected, strictly loaded into the kit architectures, and tested for finite output of the exact expected shape:

| Checkpoint | State parameters as counted by PyTorch | File bytes | Warm batch-1 model forward, median of 3 |
|---|---:|---:|---:|
| `sim_real_fno_fp16.pth` | 50,357,955 | 201,396,349 | 0.05290 s |
| `sim_real_cno.pth` | 7,960,467 | 32,154,928 | 0.14885 s |

Timings use this workstation's NVIDIA GeForce RTX 2050 and PyTorch 2.11.0+cu128, with synchronization and a warm-up. They exclude model loading, transfers, normalization and interval production. FNO complex parameters count as complex elements in PyTorch's count, so that number is not directly a count of real-valued scalar degrees of freedom. These checks establish interface/checkpoint compatibility, not pretrained quality or official runtime feasibility. No local Transolver checkpoint was available.

## Recommended next experiment order

| Direction | Decision after the evidence | Required validation before adopting it |
|---|---|---|
| A. Statistical calibration | Retain as the leading Sim-use hypothesis. Start with linear/ridge affine maps or a compact prior predictor. Saved leave-Re-out results are strong versus raw Sim, but mean-v CV error 0.85 and RMS-u 0.63 leave substantial error. | Restore/access additional Sim trajectories; repeat condition CV, compare condition-only and Real-history-only statistics, and use inferred rather than oracle condition inputs at submission time. |
| B. Temporal rescaling | Worth a cheap controlled pretraining/resampling ablation. Evidence is a one-condition pilot with lag-range sensitivity. | Fit on training conditions only, test on held-out Re/AoA, anti-alias when temporally decimating, and verify both PSD and ACF. |
| C. Advection/warping | Highest-value next causal forecasting ablation because broad Real shift validation now passes. Begin with a shared transport estimate or small shift bank, not a large learned optical-flow subsystem. | Estimate transport from 20 history frames, handle fractional/coarse-grid displacement and boundaries, then compare full 20-frame forecast quality and cost on held-out conditions. Never use hundreds of unseen frames to set the inference shift. |
| D. Stochastic/high-frequency residual | Keep as a later option. Extra fast variance persists, but its predictability and physical/PIV origin are unresolved; quiet pixels are especially sensitive to normalization. | First diagnose residuals after deterministic history and transport models. Introduce a cheap heteroscedastic head only after deterministic quality is established; distinguish interval calibration from generating actual forecast fluctuations. |
| E. Combination | Recommended research direction: static Sim statistical structure + Real-history residual dynamics; transport first as an isolated ablation, temporal rescaling as a separate subsequent factor. | Demonstrate incremental gains under the same validation split, initialization, parameter budget, training budget, and inference information. |

The next experiment should establish **B: RealPast→RealFuture** using a fold-pure existing backbone and robust normalization, and compare it to an otherwise identical version with transport estimated from the same RealPast. Preserve the 20→20 interface first; test a direct normalized residual block `z_future = z_last + F(z_history)` before introducing one-step delta rollout, which is a separate major formulation change. The increment-shift diagnostic does not prove that warping the entire nonuniform mean field is appropriate: keep a history-derived mean/static component separate and transport fluctuations or increments, then measure the outcome.

The complete Sim-value controls remain mandatory:

| Control | Information at prediction time | Status here |
|---|---|---|
| A: persistence | Last observed Real frame | Executed and measured above. |
| B: Real-history-only | Same 20 Real history frames; fold-pure learned preprocessing | Existing backbone interface checked; a clean training/evaluation comparison is not executed in this diagnostic pass. |
| C: raw-Sim fusion | Real history plus a Sim-derived representation selected **without** hidden test metadata or presumed synchronization | Not executed. A timestamp-aligned raw Sim oracle is not a valid implementation of this control. |
| D: statistical-Sim prior | Real history plus calibrated Sim mean/RMS/TKE inferred from available history | Not executed. Full new condition-CV calibration is blocked by only one local Sim condition; saved notebook calibration is evidence, not its reproduction. |

**Do not claim Sim helps until C or D beats B.** If all methods use a Sim-pretrained initialization, B versus C/D tests the value of additional Sim inputs, not whether Sim pretraining itself helps. A separate scratch-versus-Sim-pretrained comparison is required for that latter claim.

For all learned comparisons, split outer folds by measured Re (and include an AoA/extrapolation holdout where coverage permits), select hyperparameters inside training folds, fit normalizers/calibrators only there, and build windows afterward. Keep original file/condition identity with every sample. Do not treat a Sim+Real checkpoint that may have seen all Real conditions as a clean held-out initialization. The 20-frame history cannot supply a precise long-run mean/RMS without uncertainty; compare history-only estimates, shrinkage to training statistics, and inferred Sim priors.

Retain field Rel-L2, mean error, RMS error, TKE error/correlation, MVPE, horizon-resolved errors, ACF/PSD discrepancy, runtime, checkpoint size, and eventual SPS/coverage. Use the supplied scorer's **forecast-window** TKE definition: full-trajectory TKE calibration does not guarantee accurate variance over a 20-frame future block. The scorer rewards a tight interval only when it covers the target and also weights accuracy factors; optimizing width alone is insufficient. If adding losses, start with one extra positive-weight penalty, e.g. `L_field + lambda_TKE*L_TKE`, and ablate it. Negative coefficients on error penalties would reward larger error; the minus signs in the pasted illustrative loss should not be implemented literally.

No giant architecture, diffusion model, ensemble, new loss stack, neural training run, or submission was introduced. The primary requested diagnostics are complete on the data available locally; broad Sim replication and causal neural B/C/D comparisons remain explicitly future work.

## Reproduction and evidence files

Run from `D:\Project\NeurIPS` with the existing scientific Python environment:

```powershell
python research/architecture_audit/run_experiments.py
python research/architecture_audit/check_normalization.py
python research/architecture_audit/check_baselines.py
python research/architecture_audit/summarize_results.py
python research/architecture_audit/verify_results.py
```

The primary diagnostic run took **676.97 seconds** on this Windows workstation, excluding supplementary checks and final QA. This is analysis wall time, not model inference time. Seeds, NumPy/SciPy/Python versions, input file metadata, notebook/script fingerprints, frame ranges, and statistic maps are recorded. The raw-data file timestamps/size manifest is provenance metadata, not a full content hash. Timestamp validation allows float32 rounding (observed adjacent-step error about 1.45e-6) and checks nominal Real dt=0.02 against all 12 available raw files. Timing results will vary with hardware and load.

Final numerical QA checks FFT correlations against direct calculation, confirms all raw-frame guard gaps and exact cache matches, verifies PSD masses sum to one, reproduces the eight alpha fits, and checks affine invariance of equal-pixel Pearson correlation. This last check caught a small scale-dependent cutoff artifact at pixels constant in the test block but variable during training. The implementation now explicitly excludes constant test pixels from per-pixel correlation, and those columns were refreshed. The pooled results, shift experiments, alpha estimates, and decisions above are unaffected. The initial run fingerprint is preserved in `config.json`; final script fingerprints and check outcomes are in `verification.json`. All three plotted figures were visually inspected.

| Artifact | Contents |
|---|---|
| [run_experiments.py](/D:/Project/NeurIPS/research/architecture_audit/run_experiments.py) | Data reconstruction, raw-frame partitioning, FFT shift validation, normalized dynamics, independent-block alpha fitting. |
| [manifest.csv](/D:/Project/NeurIPS/research/architecture_audit/results/manifest.csv) and [cache_verification.csv](/D:/Project/NeurIPS/research/architecture_audit/results/cache_verification.csv) | Every input identity, actual Re/AoA, resolution, lengths, metadata, and all 12 raw/cache comparisons. |
| [shift_cv.csv](/D:/Project/NeurIPS/research/architecture_audit/results/shift_cv.csv) and [shift_summary.csv](/D:/Project/NeurIPS/research/architecture_audit/results/shift_summary.csv) | All 752 folds with selected shifts, held-out correlations, gains, stability, and summarized condition-bootstrap intervals. |
| [shift_by_Re.csv](/D:/Project/NeurIPS/research/architecture_audit/results/shift_by_Re.csv), [shift_by_AoA.csv](/D:/Project/NeurIPS/research/architecture_audit/results/shift_by_AoA.csv), [shift_counts.csv](/D:/Project/NeurIPS/research/architecture_audit/results/shift_counts.csv) | Re/AoA/channel/lag dependence and every selected-shift frequency. |
| [dynamics.csv](/D:/Project/NeurIPS/research/architecture_audit/results/dynamics.csv), [acf.csv](/D:/Project/NeurIPS/research/architecture_audit/results/acf.csv), [psd.csv](/D:/Project/NeurIPS/research/architecture_audit/results/psd.csv) | Mean/RMS-normalized and fluctuation diagnostics; full ACF, increment ACF and PSD curves, including all cached Real conditions. |
| [paired_dynamics_summary.csv](/D:/Project/NeurIPS/research/architecture_audit/results/paired_dynamics_summary.csv) and [paired_dynamics_gaps.csv](/D:/Project/NeurIPS/research/architecture_audit/results/paired_dynamics_gaps.csv) | Matched-condition aggregates and held-out ACF/PSD gaps. |
| [alpha_cv.csv](/D:/Project/NeurIPS/research/architecture_audit/results/alpha_cv.csv) and [alpha_profiles.csv](/D:/Project/NeurIPS/research/architecture_audit/results/alpha_profiles.csv) | Eight independent-block alpha fits and complete training-objective profiles. |
| [normalization_floor_sensitivity.csv](/D:/Project/NeurIPS/research/architecture_audit/results/normalization_floor_sensitivity.csv) | Four predetermined RMS floors, two channels/domains/temporal directions. |
| [persistence_control.csv](/D:/Project/NeurIPS/research/architecture_audit/persistence_control.csv) | 1,672 windows, condition identities, errors, time, and default-band SPS context. |
| [checkpoint_audit.json](/D:/Project/NeurIPS/research/architecture_audit/checkpoint_audit.json) | Every loaded state tensor's shape/dtype, strict-load result, parameter/file sizes and forward timings. |
| [config.json](/D:/Project/NeurIPS/research/architecture_audit/results/config.json) and [baseline_config.json](/D:/Project/NeurIPS/research/architecture_audit/baseline_config.json) | Fixed protocol, dependencies, seeds and machine/GPU context. |
| [verification.json](/D:/Project/NeurIPS/research/architecture_audit/results/verification.json) | Final numerical checks, corrected-correlation validation, reproduced alpha fits and final script fingerprints. |
| [notebook_execution.txt](/D:/Project/NeurIPS/research/architecture_audit/notebook_execution.txt) | Export of the original stored cells and textual outputs in notebook order; an evidence snapshot, not a rerun. |

Per-half mean/RMS maps are saved as `statistics_real_<condition>.npz` and `statistics_sim_5025_5.npz`; plots have descriptive filenames alongside the CSVs. All interpretation and recommendations are kept in this document.
## Matched delta/context branch separation — validation FAIL, temporal branch closed (2026-09-12)

After sub8's hidden-confirmed first-difference gain, one final temporal
inductive-bias experiment was preregistered and run. The fresh sub8 shared-head
control reproduced the historical matched validation result within 0.0002%
relative error. The candidate routed the same 38 adjacent-delta channels to a
22-wide branch and the same 82 anchor/CNO/prior channels to a 40-wide branch,
then used one sigmoid fusion and the existing ZeroMean output. It had 55,110
parameters versus control's 55,528 (−0.753%), with identical data, seeds 42/43,
batch orders, 12 epochs, field+0.30 TKE loss, selection, ensemble, mask and SPS.

| Metric | Fresh sub8 control | Two-path candidate | Relative / point change |
|---|---:|---:|---:|
| RelL2 error | 0.0777606 | 0.0778700 | +0.1407% |
| TKE error | 0.5869540 | 0.5917104 | +0.8104% |
| MVPE error | 0.0693463 | 0.0693463 | unchanged |
| Selection proxy | 82.891459 | 82.844362 | −0.047097 points |
| Adaptive SPS | 47.742357 | 47.679167 | −0.063190 points |

Both seeds regressed TKE. Both Re groups and all five AoA groups regressed TKE.
Mean framewise RelL2 over horizons 6–15 regressed 0.1101%, and cumulative TKE
improved in 0/10 mid horizons. A paired 10-trajectory bootstrap gave TKE
candidate-minus-control +0.004756, 95% CI [+0.001557,+0.008163]. The candidate
failed 10 of 18 mandatory gates, including every effect/robustness gate except
the maximum subgroup-regression bounds.

**Decision: VALIDATION FAIL — TEMPORAL BRANCH SHOULD STOP.** The reused
80-window development set remains sealed; no candidate package or leaderboard
submission was produced. Keep sub8 as incumbent, SHA256
`eeb0504211ca2fb90ddfbcb9353f14634c1b7c3e7909a013e534baaa6c2ddb8b`.
This identifies only failure of this same-capacity two-path separation. It does
not disprove all temporal architectures or establish a physical fast/slow,
phase, noise, or timescale mechanism. Given first-difference PASS plus hidden
confirmation, increment-loss FAIL, Haar FAIL, and now learned separation FAIL,
do not architecture-fish further in the temporal branch. If new work is
authorized, preregister the separate learned spatial fluctuation-energy
allocation hypothesis. Full protocol, gates, artifacts and operational
amendments are in `research/fast_slow_architecture_12_9/report.md`.

## SPS additional-feature probe — exploratory FAIL (2026-09-12)

Implemented the requested fixed-point A/B/C falsification probe in
`research/sps_feature_probe_12_9/`. A uses exact saved sub8 adaptive widths;
B is a shallow log-error tree on horizon/channel/history-std/current-width;
C uses the same learner and sampled rows plus history-delta RMS, forecast
fluctuation, spatial gradient and CNO correction magnitude. Two outer folds
hold out measured Re; inner three-fold trajectory CV selects width multipliers
or fallback A. All feature/learner choices and gates were locked before running.

All four inner selections fell back to A. Selected outer policies A/B/C therefore
all score **47.741400 offline SPS**, C-A=C-B=0; the symmetric fixed-center
target-informed oracle scores **63.292821**. Best learned inner widths were
inferior to A in both folds. **Stop this feature/learner family; no retuning or
neural uncertainty head is justified by this result.** This is not evidence
that residual-error information is absent for all models/features. Oracle
headroom does not establish attainable gain.

The 58 validation windows did not train point-head weights but selected its
checkpoint and have been repeatedly inspected. The new probe is exploratory,
not independent confirmation; CNO training provenance remains incomplete.
No 80-window holdout, package or leaderboard submission was used. Synthetic
official-scorer parity, target-mutation isolation, nested group separation and
per-window aggregate reconstruction all passed independent verification.

**Sub8 remains hidden-leaderboard verified**, reconfirmed by the user:
RelL2 94.582535, TKE 75.183675, MVPE 93.487926, time 87.219300,
SPS 37.747438, final **79.246324**. Those hidden scores are distinct from the
new probe's offline scores and retain their existing evidence status.

See `research/sps_feature_probe_12_9/report.md`, `protocol.md`,
`decision.json`, `inner_selection.json`, and `independent_verification.json`.

## Mean constraint falsification — partial mechanism support, candidate FAIL (2026-09-12)

User authorized the mean diagnostic and subsequent assumption-isolating
experiments. Protocol and code/input hashes were locked in
`research/mean_constraint_12_9/` before measurement.

On the 58 selection-exposed validation windows, exact sub8 temporal-mean error
accounts for **32.2248% of field SSE**. An unimplementable target-informed
mean-only correction reduces RelL2 error **19.1299%**; TKE changes by at most
2.22e-16. Mean SSE is almost entirely in the active region. This establishes
headroom, not learnability, and passed the preregistered stage-1 gate.

Stage 2 retrained 12 heads: ZeroMean/Ordinary x seeds 42/43 x 3 outer Re folds,
using only the existing 339 training windows. Whole Re and trajectory groups
were excluded; every normalization statistic was fit inside each training
fold. Both arms used identical initial raw weights, 55,528 parameters, batch
orders, AdamW, field+.30 TKE loss and 12 fixed epochs with no evaluation-based
checkpoint selection. Historical data reuse and incomplete frozen-CNO training
provenance remain limitations; this is not a pristine whole-pipeline test.

| OOF ensemble metric | ZeroMean | Ordinary | Relative change |
|---|---:|---:|---:|
| RelL2 error | 0.08557767 | 0.08508748 | -0.5728% |
| MVPE error | 0.08014643 | 0.07857941 | -1.9552% |
| TKE error | 0.58846867 | 0.59453837 | +1.0314% |
| Mean SSE | 3.78632011 | 3.72934488 | -1.5048% |

Both seeds improve RelL2/MVPE. Ensemble mean SSE improves in all three folds
and 10/13 Re groups; paired trajectory bootstrap CI for mean-SSE difference
is [-0.100315,-0.017789]. **Only the TKE gate fails**: +1.0314% exceeds the
locked +0.5% limit. Candidate status is FAIL, but this is positive evidence
that some mean correction is learnable in this setup. Do not summarize it as
"mean CNO must be frozen" or "all mean adaptation failed".

Descriptively removing Ordinary's mean correction restores MVPE to ZeroMean,
raises RelL2 error to 0.08568112 (erasing the benefit), and leaves Ordinary TKE
unchanged. A temporal-constant correction cannot itself change TKE; removing
the constraint also changes the jointly learned fluctuations. Isolating mean
adaptation from fluctuation learning is a separate test, not a proven solution.

Official metric calculations, SSE decomposition, source hashes, fold-local
normalization, initialization/parameter parity, batch membership and fixed
epoch counts passed verification. Detailed artifacts: `report.md`,
`decision.json`, `verification.json`, `groups.csv`, `horizon.csv`, and the
per-seed checkpoints under `research/mean_constraint_12_9/`.
No 80-window holdout or leaderboard submission was used. Sub8 remains the
hidden-verified incumbent at final 79.246324.

## Fixed OOF mean transplant — post-hoc support (2026-09-12)

User requested implementation and authorized using the last leaderboard slot
if a useful concrete assumption test warrants it. Implemented
`research/mean_constraint_12_9/hybrid.py`: ZeroMean prediction plus the temporal
mean of Ordinary minus the temporal mean of ZeroMean. Coefficient is exactly
one; no target, fitted calibration, or coefficient search enters prediction.
This is explicitly post-hoc and does not relabel the previous Ordinary FAIL.

| OOF metric | ZeroMean | Hybrid | Relative change |
|---|---:|---:|---:|
| RelL2 error | 0.08557767 | 0.08497871 | -0.6999% |
| MVPE error | 0.08014643 | 0.07857941 | -1.9552% |
| TKE error | 0.58846867 | 0.58846867 | unchanged |
| Mean SSE | 3.78632011 | 3.72934488 | -1.5048% |

The fixed transplant meets the previous numerical effect/robustness criteria,
but these are descriptive checks on a post-hoc candidate, not preregistered
independent confirmation. Both seeds improve RelL2/MVPE; ensemble mean SSE
improves on all folds and 10/13 Re groups. Seed 42 mean SSE alone regresses
0.0821%, so do not claim every seed improves every mean metric. Paired
trajectory bootstrap intervals for differences are RelL2
[-0.00086323,-0.00037220], MVPE [-0.00216736,-0.00094018].

Verified source hashes/sample identities, TKE invariance (maximum change
2.22e-16 in float64 diagnostic), equality of hybrid fluctuation SSE to
ZeroMean and mean SSE/MVPE to Ordinary, and shared zero-mask preservation.
The float64 diagnostic does not certify float32 deployed-package parity.
It supports separating mean adaptation from fluctuation prediction in this
setup, not a physical Sim2Real mechanism or a guaranteed hidden-score gain.

No leaderboard attempt was spent: these are fold-specific OOF heads, not a
full-data mean adapter integrated and verified against exact deployed sub8.
Submission authorization is acknowledged; package readiness is not established.
See `hybrid_report.md`, `hybrid_decision.json`, `hybrid_windows.csv` and
`hybrid_groups.csv` under `research/mean_constraint_12_9/`.

## Exact-sub8 mean-adapter bridge — validation FAIL (2026-09-12)

Implemented the authorized full-training bridge in `research/sub8_mean_adapter/`.
Two Ordinary heads were trained on 339 training windows, seeds 42/43, fixed 12
epochs and the previous matched loss/optimizer. Only their temporal mean output
is added to fresh exact-sub8 predictions, using train-only normalizers, the same
history mask, coefficient one and unchanged adaptive interval rule.

On 58 reused validation windows, fresh sub8 vs candidate:
RelL2 error 0.07776183 -> 0.07751972 (-0.3113%); MVPE
0.06935396 -> 0.06871518 (-0.9210%); TKE 0.58696133 -> 0.58696133;
SPS 47.741399 -> 47.871434 (+0.130035 offline points).
The locked RelL2>=0.5% and MVPE>=1% improvement gates fail. Re 20369
regresses RelL2 (+0.1642%), and seed 43 regresses RelL2/MVPE. TKE and SPS
gates pass. No coefficient, seed or checkpoint retuning was performed.

This is a failed robustness bridge from the post-hoc OOF mean-transplant
finding to exact sub8, not a negation of that earlier descriptive result.
Validation remains development evidence. Original sub8 bytes were preserved.
Experimental staging exists at `CCCCCC/submissions/sub8_mean_adapter_candidate`,
marked VALIDATION FAIL; it is not a ready submission. Per protocol, metric
failure stopped before runtime benchmarking, ZIP creation and leaderboard.
The user's authorization to use the last submission is retained; no slot was
spent. See `research/sub8_mean_adapter/report.md`, `decision.json`, and
`verification.json` for the artifacts and checks.

## Direct mean objective and sub10 update (2026-09-13)

Completed `research/mean_only_12_9/`: direction diagnostic on the fixed
full-training adapters, then six matched direct-mean runs across the existing
three Re folds and seeds42/43. The same 55,528-parameter network, normalization,
orders and 12-epoch budget were retained; output was temporally averaged and
trained with normalized mean-residual MSE only. This tests the combined output
projection/objective intervention, not removing TKE loss in isolation.

The prior adapter ensemble has positive target alignment in both validation Re
groups. At Re10142 the descriptive optimal scalar is 0.9231 and mean SSE improves;
at Re20369 it is 0.4065 and the unscaled correction increases mean SSE. These
are target-informed descriptive values, never applied or selected as calibration.

Direct-mean hybrid OOF: RelL2 0.08553743, MVPE 0.07986816, TKE 0.58846867,
mean SSE 3.80538683. Against ZeroMean, RelL2 improves only 0.0470%, MVPE
0.3472%, and mean SSE regresses 0.5036%. Against Ordinary-mean hybrid it loses
mean SSE in **all 13 Re groups**; both seeds regress RelL2/MVPE. Paired
trajectory bootstrap mean-SSE difference CI is [0.052851,0.100511]. Only TKE
invariance passes the gates. **FAIL: stop this direct-mean configuration without
retuning; no full-data integration, ZIP or leaderboard use.** This closes the
tested objective/configuration, not all possible mean adaptation.

User supplied the following **sub10 hidden-leaderboard result**: RelL2 94.582535,
TKE 75.183675, MVPE 93.487926, time 87.277645, SPS 37.661454,
final 79.230261. Relative to sub8, point scores are unchanged, SPS is -0.085984,
time +0.058345 and final -0.016063. The result is user-reported leaderboard
evidence, not an independently fetched server record.

Both local sub10 ZIP copies have SHA256
`4a686c321cf862c02857ad84a051bbc2e25f23a6c4c766a642460ec942cf8db3`.
Archive comparison with sub8 found identical entries and identical file bytes
except submission.py. Ignoring newline encoding, the sole code change multiplies
clamped u half-width by 0.80 and v half-width by 0.95. This is a hidden FAIL for
that specific bounds-narrowing rule, not for mean correction or all UQ methods.
Preserve original sub8 bounds and do not transfer the sub10 narrowing into a
future candidate. Sub8 remains the best supplied hidden result (79.246324).
The current remaining submission quota is not verified after the new sub10
information; no assumption about a still-available final slot is warranted.
