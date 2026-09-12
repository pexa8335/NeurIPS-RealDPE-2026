# Kiểm định bản chất RealPDE Track 1 — 11/09/2026

Đã triển khai tuần tự Q1–Q5. Kết luận chính: sub7 thiếu năng lượng dao động rõ rệt; nhưng chưa xác định được nguyên nhân là objective, representation, thông tin đầu vào hay observation model. Chưa có cơ sở để chọn ngay tăng amplitude, thêm noise hoặc một kiến trúc mới.

## Phạm vi

Kiểm kê 182 HDF5, gồm 82 Real và 100 Sim. README archive yêu cầu loại Real7575_0 vì trùng6300_0; tensor coarse đã xác nhận bằng nhau chính xác. Thống kê chính dùng81 Real +100 Sim. Không coi số file là số thí nghiệm độc lập về mặt thống kê.

TKE đo trên UV stride2 tại32×64; ACF và history descriptors dùng block average xuống8×16. Các kết luận coarse không đại diện toàn bộ phổ không gian. Các thống kê domain tính ngang trọng số trajectory; số liệu package tính ngang trọng số80 development windows, ID-joined. Không có confirmation set mới hay kết quả leaderboard trong lượt này.

Không train, submit hoặc sửa incumbent. SHA256 sub7 vẫn là 2e42b16da2eba7e794d90579003e31c9417d3e720f7babdb932bd376f810762a. Protocol và toàn bộ CSV/script nằm ở research/fundamentals_11_9/.

## Q1 — History20/horizon20 so với thang tương quan

ACF pooled sau trừ temporal mean, lag1–80. Median lần đầu xuống dưới1/e:

| Domain | Channel | ACF1 | ACF20 | Crossing lag |
|---|---|---:|---:|---:|
| Real | u | 0.9513 | 0.0421 | 6 |
| Real | v | 0.9440 | −0.0575 | 4 |
| Sim | u | 0.9917 | 0.1746 | 14 |
| Sim | v | 0.9875 | −0.1456 | 10 |

Real giữ median6/4 ở cả hai nửa trajectory; Sim13/9 ở nửa đầu,14/10 ở nửa sau. dt lưu trữ khoảng0.02000 Real và0.02002002 Sim; chênh lệch sampling nhỏ hơn nhiều chênh lệch ACF. Chưa xác minh đơn vị nên không tự gọi dt là giây.

Cùng số frame không đồng nghĩa cùng khoảng dự báo theo tương quan. Điều này tạo căn cứ kiểm tra temporal representation, nhưng không chứng minh history20 thiếu thông tin. ACF âm cũng có thể phản ánh dao động.

Welch peak period median u/v Real128/23.27 frame, Sim128/64; peak-bin mass0.109/0.143 và0.203/0.176. Chưa được gọi là shedding period vật lý vì phổ rộng, xu thế và record length có ảnh hưởng.

![ACF](D:/Project/NeurIPS/research/fundamentals_11_9/Q1_timescales.png)

Evidence: Q1_summary.csv, Q1_timescales.csv, Q1_acf.csv, Q1_spectra.csv.

## Q2 — Biên có phải nguồn gap chính?

Dải trái/phải4 cột, trên/dưới4 hàng; interior loại các dải này. Góc overlap giữa các dải, không cộng vùng để suy ra tổng error. Trung bình horizon và80 windows:

| Vùng | MSE | Local RelL2 |
|---|---:|---:|
| Trái | 0.000031 | 0.0692 |
| Phải | 0.000113 | 0.0723 |
| Trên | 0.000017 | 0.0318 |
| Dưới | 0.000076 | 0.0547 |
| Interior | 0.000168 | 0.1016 |

Không ủng hộ phát biểu đơn giản “gap chủ yếu ở mép miền”: interior cao nhất cả MSE và local relative error. Điều này chưa bác bỏ thông tin ngoài miền ảnh hưởng interior.

Chọn lag edge–center trong[-20,20] bằng nửa đầu, giữ nguyên đánh giá nửa sau, so với lag0/opposite lag. Lag dương nghĩa edge đi trước center. Ở Real, chỉ9.9–33.3% trajectory tùy edge/channel vừa có lag dương vừa cải thiện held-half correlation so với lag0. Median lag biên phải−14u/−12v. Không có bằng chứng nhất quán để mặc định một biên là đầu vào động lực chi phối. Spatial averaging, tín hiệu hằng, giới hạn lag và dao động chung làm phép kiểm này không mang ý nghĩa nhân quả.

![Boundary](D:/Project/NeurIPS/research/fundamentals_11_9/Q2_boundary.png)

Evidence: Q2_edge_lags.csv, Q2_lag_summary.csv, Q2_package_boundary_errors.csv, Q2_boundary_summary.csv.

## Q3 — History tương đồng có cho future tương đồng?

Block40 không overlap:20 history+20 future. Donor là60% block đầu, bỏ thêm block40 guard, phần còn lại làm query. Tổng1,535 queries; donor cùng trajectory, k=1; scale fit chỉ trên donor. Hai distance cố định raw/temporal-centered history, không dùng future để chọn neighbor.

Centered-history results, trung bình theo trajectory:

| Domain/comparator | Field RelL2 coarse | Fluctuation RelL2 coarse | TKE-map RelL2 | Scalar energy error |
|---|---:|---:|---:|---:|
| Real nearest | 0.1099 | 1.2371 | 0.7543 | 0.2235 |
| Real mean donor errors | 0.1260 | 1.4261 | 0.8116 | 0.2519 |
| Real last-frame persistence | 0.1075 | 1.0000 | 1.0000 | 1.0000 |
| Real history TKE | — | — | 0.7653 | 0.1727 |
| Sim nearest | 0.2252 | 1.2459 | 0.8865 | 0.2157 |
| Sim mean donor errors | 0.2596 | 1.4231 | 0.9528 | 0.2416 |
| Sim last-frame persistence | 0.1944 | 1.0000 | 1.0000 | 1.0000 |
| Sim history TKE | — | — | 0.8750 | 0.1626 |

Mean donor errors là trung bình lỗi từng donor, không phải lỗi ensemble. Nearest raw có field RelL2 Real0.1095/Sim0.2186, cùng kết luận. Nearest centered có future fluctuation cosine0.196/0.200, so với gần0 của donor bất kỳ: history có tín hiệu hữu ích trong phép đo này. Tuy nhiên nearest thua persistence về field và fluctuation error.

Không suy ra irreducible uncertainty: thư viện nhỏ, descriptor coarse và distance chưa bảo đảm latent state tương đương. Scalar energy và energy map là hai estimand khác nhau; không so số trực tiếp để tuyên bố một bài toán dễ hơn tuyệt đối.

Trên đúng80 package windows, history TKE map error0.7971, thua sub7 0.5806. Không đề xuất thay sub7 bằng baseline này.

![Analogs](D:/Project/NeurIPS/research/fundamentals_11_9/Q3_analogs.png)

Evidence: Q3_guard_manifest.csv, Q3_analog_queries.csv, Q3_trajectory_means.csv, Q3_summary.csv, Q4_history_energy_same80.csv.

## Q4 — Năng lượng thiếu, switching zeros và ensemble

Scorer K=0.5×tổng population variance theo20 frame của u,v. Đây là statistic của cửa sổ quan sát, không tự động là TKE3D vật lý.

| Exact package, mean80 windows | Giá trị |
|---|---:|
| Tổng K prediction/target, mean window ratios | 0.6257 |
| Variance ratio u/v | 0.5797/0.7174 |
| TKE-map cosine | 0.8235 |
| TKE-map RelL2 | 0.5806 |
| Future fluctuation cosine/RelL2 | 0.5959/0.8092 |
| Squared TKE error tại Kpred<Ktarget | 92.67% |
| Target variance từ switching zero/nonzero | 0.3504% |
| Squared TKE error tại pixel mixed zero/nonzero | 4.31% |

Thiếu năng lượng là dấu hiệu mạnh, nhưng không chỉ là một hệ số amplitude: map cosine chưa bằng1, fluctuation error còn lớn. 92.67% là tỷ trọng bình phương lỗi ở nơi underprediction, không phải tỷ lệ pixel hay mức cải thiện mà rescale chắc chắn đạt.

Phân rã Var(U)=p1 Var(U|nonzero)+p0 p1 Mean(U|nonzero)^2. Switching term là đại số, không được gọi là sensor noise. So length20/40/80 dùng cùng prefix floor(T/80)×80 mỗi trajectory:

| Domain | Length | Mean observed K | Mean switching fraction |
|---|---:|---:|---:|
| Real | 20 | 0.000163 | 0.3354% |
| Real | 40 | 0.000194 | 0.2806% |
| Real | 80 | 0.000211 | 0.2514% |
| Sim | 20 | 0.010180 | gần0 |
| Sim | 40 | 0.016121 | gần0 |
| Sim | 80 | 0.020266 | gần0 |

Window definition ảnh hưởng K; objective cần khớp window20. Chưa dùng chênh magnitude Sim/Real để suy ra turbulence vật lý vì chưa xác minh units.

Legacy saved research predictions, tách biệt exact package: seed42 TKE error0.580579; seed43 0.578251; ensemble0.580840. RelL2 ensemble tốt hơn (0.086491 so0.086761/0.086966) nhưng TKE kém hơn cả hai seed. Identity K(mean)=mean(K)−disagreement variance đúng với max error2.90e−18. Ensemble loại0.701% tổng mean seed energy; cơ chế này có thật nhưng không đủ giải thích thiếu năng lượng cỡ37% của package.

![Variance](D:/Project/NeurIPS/research/fundamentals_11_9/Q4_variance.png)

Evidence: Q4_sub7_TKE.csv, Q4_variance_windows.csv, Q4_matched_length_trajectory.csv, Q4_ensemble.json.

## Q5 — Động lực hay observation model?

[Nguồn chính thức](https://realpdecompetition.github.io/) mô tả NACA4418 trong water tunnel, PIV đo vận tốc mặt cắt và CFD3D. UV lưu trữ là phép quan sát của hệ; chưa đủ để mặc định trạng thái2D khép kín. Archive README lưu ở dataset_README.md, thông tin nguồn ở external_sources.json.

Ghép81 nominal filenames bằng actual metadata, không giả định Re bằng nhau hay chuỗi đồng bộ. Median ACF crossing vẫn Real6/4, Sim14/10. Median power fraction trên0.25 cycles/frame: Real u/v0.004027/0.000691; Sim0.000060/0.000089. Real có fraction cao hơn nhưng chưa phân biệt động lực nhanh, noise PIV hay preprocessing.

Median correlation increment láng giềng x: Real u/v0.545/0.697; Sim0.596/0.534. Không ủng hộ coi toàn bộ variance Real là noise trắng độc lập. Coherence cũng không chứng minh toàn bộ là chuyển động vật lý: pipeline PIV có thể tạo tương quan.

Các câu hỏi chưa có câu trả lời đủ từ nguồn đã đọc:

1. Units/normalization của t,x,y,u,v có giống nhau giữa file/domain? Cần U∞ và chord để đo convective time.
2. PIV interrogation window, overlap, filtering, vector replacement và zero encoding là gì? Có validity/confidence mask trước interpolation không?
3. CFD xuất mặt cắt nào, snapshot hay temporal average; có observation filtering tương ứng PIV không?
4. Có inflow/boundary time series, vị trí spanwise và repeated experiments cùng điều kiện không?
5. Record đã bỏ transient chưa; start time có ý nghĩa ghép cặp; acquisition có thay đổi theo Re/AoA không?

Không tự vay thông số từ benchmark foil khác geometry. Evidence: Q5_nominal_pairs.csv, trajectories.csv.

## Quyết định tiếp theo

1. Ưu tiên tách scalar energy, spatial energy allocation và temporal fluctuation thành các kiểm định dự báo riêng. Calibrator/predictor chỉ fit train/validation, khóa trước confirmation; so sub7 trên RelL2/TKE/MVPE và per-condition regressions. Target-derived oracle chỉ giới hạn họ biến đổi được thử, không phải theoretical ceiling.
2. Q1 tạo căn cứ cho matched temporal-representation ablation, với data/compute/seed/selection cố định. Chưa đủ để chọn ngay history dài hơn, warp hay stochastic head. Analog thất bại không bác bỏ representation mạnh hơn.
3. SPS cần conditional calibration. Audit phase1 trước trên fixed points cho adaptive SPS45.09→47.84, coverage84.38%→90.26%; u/v coverage86.75%/93.82%, nhóm history-std u cao nhất74.22%. Các số này kế thừa audit, không chạy lại trong Q1–Q5. Cần residual scale theo channel/horizon/history-energy và width–coverage tradeoff, fit validation rồi khóa; không tăng width toàn cục hay mặc định aleatoric uncertainty.
4. Giảm ưu tiên sửa mask như lời giải chính cho TKE, tăng seed để bù năng lượng, hoặc mặc định biên là nơi tập trung lỗi. Bằng chứng hiện tại không hỗ trợ chúng như hướng chính.

Chưa triển khai model mới. Đây là kết luận mô tả có giới hạn, không có confidence interval hay hiệu chỉnh multiple comparisons.

## Tái lập

Thứ tự: ingest.py → q1_summarize.py → q2_boundary.py → q3_analogs.py → q4_predictor.py → q5_domains.py → summarize_all.py → plots.py → verify.py.

lock.json, io_change.json, ingest_complete.json lưu provenance. Ingestion đổi sang đọc HDF5 contiguous rồi stride trong NumPy để tránh I/O chậm, đã kiểm tra tensor tương đương. Q4 dùng common prefix cho length comparison; source manifest cuối lưu riêng vì protocol được bổ sung ghi chú này sau khi ingest.

verification.json PASS: ACF bounds, donor/query guard≥40, loại duplicate khỏi primary tests, variance decomposition, ensemble identity, alignment history/future mẫu kiểm tra và hash sub7 bất biến. Đây là focused checks, không phải kiểm định toàn bộ raw archive hay leaderboard. Chart Q4 dùng đúng common-prefix như bảng.

Hypothesis1 audit trước đếm82 Real có duplicate; không coi đó là82 record riêng biệt. Báo cáo này loại duplicate theo README, giữ nguyên artifacts lịch sử. Không chuyển giữa legacy cache và exact package để làm đẹp kết quả.

## Daily-submission decision — channel và coarse-horizon intervention

Vì leaderboard chỉ cho một submission mỗi ngày, lượt này coi leaderboard là confirmation set đắt. Không submit global alpha=1.05. Hai protocol mới đều được khóa trước khi đọc kết quả tương ứng; selection chỉ dùng 58 validation windows, SPS và mọi thành phần còn lại của sub7 giữ nguyên.

### Channel-conditional correction

Family 16 cấu hình: alpha_u thuộc {1.00,1.05,1.10,1.15}, alpha_v thuộc {1.00,1.025,1.05,1.075}. Selection yêu cầu RelL2 degradation không quá1%, MVPE không quá0.5%, worst-Re RelL2 không quá2%, sau đó chọn TKE error thấp nhất. Candidate chỉ được mở holdout nếu giảm validation TKE error ít nhất3%.

Validation chọn `(alpha_u,alpha_v)=(1.05,1.075)`:

- TKE error giảm1.723%.
- RelL2 error tăng0.835%.
- MVPE gần như không đổi.
- Worst-Re RelL2 degradation1.054%.

Candidate feasible theo point-cost constraints nhưng rơi vào Case B vì TKE signal dưới3%. Theo rule khóa trước: không đọc80-window holdout, không package, không submit. Việc alpha_v được chọn lớn hơn alpha_u cũng cho thấy variance-ratio diagnosis không chuyển trực tiếp thành hệ số correction tối ưu; point/TKE geometry vẫn chi phối trade-off.

### Coarse-horizon follow-up

Không có candidate chưa kiểm định leaderboard nào với provenance đủ rõ để thay thế ngay, nên mở family một bậc tự do như decision tree đã định. Giữ base channel coefficients ở trên, chia horizon1–5/6–10/11–15/16–20 và đặt `alpha_c(g)=alpha_c_base+beta*r_g`, với `r={-1,-1/3,1/3,1}` và beta thuộc {0,0.025,0.05,0.075}. Dùng cùng constraints và gate TKE3% trên validation.

Validation chọn beta=0. Nghĩa không có horizon trend trong family thử nghiệm cải thiện được channel-only candidate. Kết quả tốt nhất vẫn là TKE −1.723%, dưới gate; không mở holdout, không package và không submit.

**Quyết định:** đóng post-hoc fluctuation-amplitude family gồm global, channel-only và coarse-horizon form đã thử. Không suy rộng thành mọi dạng calibration đều vô ích. Spatial correction không được mở vì channel gate không đạt. Hướng kế tiếp là matched temporal-representation/retraining experiment; cần protocol riêng với comparator, compute, seed, selection và stop rule trước khi chạy. Giữ nguyên sub7 làm incumbent và giữ lượt leaderboard cho candidate có local signal mạnh hơn.

Artifacts: `research/channel_intervention_11_9/` và `research/horizon_intervention_11_9/`. Mỗi thư mục có protocol, validation grid, selection lock và decision. Không có holdout result hay submission ZIP vì gate không đạt.

## Temporal-span experiment — Stage 0 deployment blocker

Đã audit trước coding/training theo yêu cầu. Current cache tạo history `[s,...,s+19]` và target `[s+20,...,s+39]`. Candidate có cùng forecast origin phải dùng `[s-19,s-17,...,s+19]`, vẫn đúng20 frame nhưng phủ38 stored intervals thay vì19.

Candidate dựng được offline chỉ khi `s>=19`. Do mỗi trajectory có một sample `s=0`, matched subset giảm từ339/58/80 xuống282/48/66 ở train/validation/reused development holdout. Assertions xác nhận đủ20 input,20 target, history nằm hoàn toàn trước target, target control/candidate giống hệt và trajectory không qua fold.

Tuy nhiên submission interface chỉ nhận tensor `(N,20,32,64,3)` chứa20 frame liên tiếp. Nó không nhận39-frame buffer, earlier-frame indices, trajectory ID hay callback để lấy thêm dữ liệu. Vì vậy hidden inference không thể tạo `[t-38,t-36,...,t]`. Subsample input được cấp chỉ còn10 observations trên cùng temporal support; duplicate/interpolate frame hoặc đổi target là intervention khác; train stride-2 rồi infer consecutive tạo distribution mismatch.

Theo stop rule của yêu cầu, **không khóa Stage-1 training protocol, không train, không đọc holdout, không package và không submit**. Hypothesis wider temporal support vẫn chưa được kiểm định; điều bị bác bỏ là khả năng deploy candidate này qua interface hiện tại.

Stage-0 cũng xác nhận residual heads sub7 có55,528 parameters/head, 120 packed input channels, ZeroMean output, lambda TKE0.30, seeds42/43, epochs chọn10/12, và split theo Re. Current source mô tả AdamW lr1e-3, weight decay1e-4, batch12,12 epochs, seeded permutation, không scheduler/augmentation. Checkpoint weights/metadata/normalization khớp package, nhưng immutable historical training-log linkage không đầy đủ. Packaged CNO training provenance cũng chưa khôi phục được; local `cno_baseline/config.json` không có checkpoint/history nên không được coi là nguồn của packaged backbone. Một test tương lai phải so retrained matched control và candidate, đồng thời báo original sub7 riêng.

Chi tiết: `research/temporal_span_11_9/stage0_audit.md`; machine-readable evidence: `stage0_verification.json` và `index_audit.csv`. Hash sub7 sau audit vẫn `2e42b16da2eba7e794d90579003e31c9417d3e720f7babdb932bd376f810762a`.

Next branch có information value cao nhất phải dùng đúng20 frame được cung cấp, chẳng hạn matched temporal encoder/preprocessing trên cùng observations. Nó kiểm định cách biểu diễn thông tin temporal hiện có, không còn là wider temporal support, nên cần hypothesis và protocol mới trước khi triển khai.

## Matched first-difference representation — PASS, candidate packaged

Stage0 xác nhận residual-head input120 channels gồm40 normalized UV-history,40 frozen-CNO future và40 damped-prior channels. Nối thêm19 UV differences sẽ tăng10,944 weights. Thay vào đó candidate dùng matched-capacity basis `[u0,v0,du1,dv1,...,du19,dv19]` thay cho history block; biến đổi khả nghịch, dùng đúng20 frame và giữ55,528 parameters/head. Delta normalization chỉ fit339 train windows.

Hai arm control/delta dùng cùng cached CNO, targets,339/58 split, loss MSE+0.30 TKE, AdamW1e-3/wd1e-4,12 epochs,batch12,seeds42/43, batch orders và selection proxy. Control tái lập historical selected epochs10/12 và scores82.734088/82.745028 gần stored82.734087/82.745178; weights không bit-identical nên causal comparator vẫn là retrained control.

Validation58 windows: proxy82.755267→82.891462 (+0.136195); RelL2 0.078313→0.077760 (−0.706% error); TKE0.599996→0.586953 (−2.174%); MVPE giữ0.069346; adaptive SPS47.508633→47.742278. Worst Re/AoA subgroup vẫn cải thiện RelL2−0.257% và TKE−0.931%. Toàn bộ promotion gates khóa trước đều PASS.

Candidate recovery chủ yếu ở u: validation u variance ratio0.5661→0.6028, v0.7136→0.7154; energy ratio0.6080→0.6365; TKE-map cosine0.8284→0.8311; fluctuation cosine0.5570→0.5719. u RelL2 cải thiện0.85%, v regress0.25%. Field gain tập trung h1–7; cumulative TKE tệ hơn h2–5, tốt hơn từ h6 và gain mạnh nhất khoảng h14 (~2.9%).

80-window reused development sanity được mở sau validation và không dùng selection: proxy+0.139324; RelL2−0.332%; TKE−2.398%; MVPE unchanged; SPS+0.206178. Energy ratio0.6257→0.6576, u variance0.5797→0.6300, v0.7173→0.7218. Mọi Re group cải thiện TKE; worst subgroup RelL2+0.075%, dưới gate3%. Đây không phải independent confirmation.

Artifact mới `CCCCCC/submissions/sub8_cno_delta_zeromean_adaptive_sps.zip`, SHA256 `eeb0504211ca2fb90ddfbcb9353f14634c1b7c3e7909a013e534baaa6c2ddb8b`,63 entries. Extracted-ZIP smoke test PASS shape `(N,20,32,64,3)`, float32, finite, ordered bounds, zero pressure. Live batch16 development metrics RelL20.0859462/TKE0.5666893/MVPE0.0820055/SPS48.044394; warm local runtime median0.1458s/sample. Original sub7 hash vẫn `2e42b16da2eba7e794d90579003e31c9417d3e720f7babdb932bd376f810762a`.

**Decision: READY FOR LEADERBOARD, chưa upload.** Kết quả củng cố vừa phải giả thuyết rằng explicit short-time changes hữu ích trên Real, nhưng không chứng minh Sim/Real timescale mismatch là cơ chế nhân quả. Equal capacity loại confound số parameter, không loại optimization/conditioning effect của basis mới. Report đầy đủ ở `research/delta_representation_11_9/report.md` cùng protocol, manifests, normalization, checkpoints, predictions, raw metrics, plots, hashes và verification.

## Hidden confirmation của sub8 và conditional-SPS follow-up

Leaderboard sub8: RelL2 94.582535, TKE75.183675, MVPE93.487926, SPS37.747438, Time87.219300, Final79.246324. So sub7: RelL2+0.030799, TKE+0.360288, MVPE unchanged, SPS+0.168434, Time−0.058081, Final+0.088143. Delta representation generalize đúng hướng lên hidden Real; missing fluctuation energy ít nhất một phần có thể dự báo từ20 input frames. Kết quả củng cố temporal-representation hypothesis, chưa chứng minh Sim/Real timescale mismatch là cơ chế duy nhất.

Với một lượt submission còn lại, chọn clean hypothesis: giữ exact sub8 points, kiểm tra interval rule cũ có phân bổ adaptive excess width sai không. Protocol khóa trước dùng train-q75 threshold u=0.009498917, grid9 configurations `m_u_high={1,1.15,1.30}` × `m_v={1,0.95,0.90}`, chọn trên58 validation; gate SPS gain≥0.15, aggregate coverage loss≤1pp, per-Re loss≤2pp.

Baseline validation SPS47.741399/coverage89.685%. Best `(1.0,0.9)` đạt47.806179/89.211%: gain chỉ+0.064780, dưới gate. Mở rộng high-std u tăng coverage nhưng SPS giảm: m_u1.15 cho90.079%/47.656567; m_u1.30 cho90.391%/47.566090 khi m_v1. Width penalty lớn hơn benefit coverage.

**FAIL:** structured conditional miscoverage không chuyển thành leaderboard-worthy gain bằng rule hai vùng này. Không đọc holdout, không package, không submit. Giữ lượt leaderboard chưa dùng và giữ sub8 incumbent hash `eeb0504211ca2fb90ddfbcb9353f14634c1b7c3e7909a013e534baaa6c2ddb8b`. Next highest-information branch: một matched temporal-change objective trên delta representation để kiểm tra residual energy gap là loss-allocation problem hay feature-access problem. Chi tiết ở `research/sub8_sps_recalibration_11_9/report.md`.


## Intervention 1 — Frozen scalar energy và lane SPS (đã chạy)

Chuyển từ diagnosis sang intervention theo chỉ đạo mới. Protocol được ghi trước tại research/scalar_intervention_11_9/protocol.md. Dùng58 validation windows để chọn, khóa selected_parameters.json trước khi đánh giá80 development holdout đã dùng trong diagnosis. Không gọi holdout này là confirmation độc lập.

Scalar: p_new = mean_t(p) + alpha × (p−mean_t(p)); alpha trong {1,1.05,1.10,1.15,1.20,1.25}. Điều kiện pass: giảm TKE error ít nhất3%, tăng RelL2 không quá1%, MVPE không quá0.5%, worst-Re RelL2 không quá2%; áp dụng cả validation và holdout. Chọn TKE thấp nhất trong candidates thỏa point-cost constraints, tie chọn alpha nhỏ nhất.

| alpha | Validation RelL2 | Validation TKE error | ΔRelL2 relative | ΔTKE relative | Point-cost feasible |
|---:|---:|---:|---:|---:|---|
| 1 | 0.078314 | 0.600003 | +0.000% | +0.000% | True |
| 1.05 | 0.078912 | 0.589959 | +0.763% | -1.674% | True |
| 1.1 | 0.079635 | 0.588517 | +1.686% | -1.914% | False |
| 1.15 | 0.080480 | 0.596577 | +2.766% | -0.571% | False |
| 1.2 | 0.081444 | 0.614453 | +3.996% | +2.408% | False |
| 1.25 | 0.082521 | 0.641975 | +5.371% | +6.995% | False |

Validation chọn alpha1.05: TKE giảm1.674%, RelL2 tăng0.763%, MVPE gần như không đổi. Không đạt mức giảm TKE3%. Holdout khóa alpha1.05: TKE0.580615→0.566122 (−2.496%), RelL2 0.086232→0.086855 (+0.723%), MVPE0.082005→0.082005. Worst-Re RelL2 tăng0.795%. Vẫn không pass.

**Quyết định scalar: FAIL theo tiêu chí khai báo trước.** Gain có thật trên development nhưng chưa đủ theo tradeoff đã khóa. Không đổi ngưỡng sau khi thấy kết quả; không mở spatial-energy experiment, không retrain và không tạo submission candidate từ thử nghiệm này. Không diễn giải FAIL thành bác bỏ mọi conditional correction.

Lane SPS riêng giữ nguyên points, thử channel-only halfwidth multipliers (u,v): (1,1),(1.1,1),(1.2,1),(1,.9),(1.1,.9),(1.2,.9). Chọn SPS validation cao nhất, yêu cầu gain≥0.5 score point và coverage giảm≤1 percentage point ở cả hai split.

Validation chọn(1,0.9): SPS 47.508141→47.548607, coverage 89.464%→88.445%. Holdout: SPS47.839039→47.927938 (+0.088899), coverage90.264%→89.326%. **Không đạt gate SPS**; không đưa thay đổi này vào sub7. Đây chỉ là family channel-only nhỏ, chưa exhaust conditional SPS.

Kiểm tra algebra temporal mean/variance và hash incumbent PASS (verification.json). Sửa cách biểu diễn coverage gate thành0.01 vì scorer trả fraction, đúng protocol1 percentage point; không đổi lựa chọn hay kết luận. Artifacts gồm validation_predictions.npz có sample IDs, validation_results.json, selected_parameters.json, holdout_results.json và run.py. Không phối hợp hai lane thành một predictor mới.

Ưu tiên có điều kiện sau lượt này: giữ incumbent; SPS vẫn là lane riêng có thể nghiên cứu tiếp bằng một protocol mới. Spatial correction đang đóng theo scalar gate; temporal/architecture/stochastic chưa được mở. Các ngưỡng này là quyết định thực nghiệm, không phải trọng số leaderboard chính thức.

## Matched future-increment objective — FAIL, giữ lượt submission cuối

Sau hidden PASS của sub8, kiểm định assumption còn lại bằng đúng một thay đổi: giữ nguyên CNO, delta representation, ZeroMean head55,528 params, cached features, data, normalization, field loss, TKE coefficient0.30, AdamW, batch12,12 epochs, seeds42/43, batch orders, checkpoint proxy, ensemble, inference và SPS; chỉ thêm train-scale-normalized ordered future-increment MSE với `lambda_delta=0.5`. Hệ số được khóa từ audit train-only trước khi train candidate.

| Metric | sub8 control | + increment loss | Thay đổi |
|---|---:|---:|---:|
| Selection proxy | 82.891462 | 82.868184 | −0.023279 point |
| RelL2 error | 0.077760 | 0.077737 | −0.030% |
| TKE error | 0.586953 | 0.589589 | +0.449% |
| MVPE error | 0.069346 | 0.069346 | 0.000% |
| Ordered increment MSE | 0.224450 | 0.223520 | −0.414% |

Candidate không đạt ba positive-effect gates đã khóa: proxy phải tăng≥0.08, TKE error phải giảm≥1%, increment error phải giảm≥3%. Cả hai paired seeds đều fail combined gate: TKE seed42 `0.586454→0.587234`, seed43 `0.588830→0.593474`. Không Re group nào cải thiện TKE; chỉ2/5 AoA groups cải thiện. Point guardrails pass nhưng không đủ để promote khi mechanism và proxy đều fail.

**Quyết định: validation FAIL; không mở80-window development holdout, không package, không submit.** Lượt leaderboard cuối vẫn còn nguyên. Trong intervention đã khóa này, thêm scalar first-order increment incentive không giải quyết phần TKE còn thiếu; increment chỉ nhích0.414% rồi TKE xấu hơn. Evidence hiện nghiêng về feature accessibility/inductive bias của delta basis là nguồn gain của sub8, thay vì thiếu trọng số cho first-order temporal-change loss. Kết luận chỉ áp dụng cho objective và lambda đã preregister, không bác bỏ mọi temporal objective.

Không stack thêm loss để dùng lượt cuối. Branch nghiên cứu hợp lý tiếp theo là multiscale temporal representation hoặc fast-dynamics branch, vẫn giữ field/TKE objective của sub8; chỉ cân nhắc leaderboard khi matched offline signal mạnh hơn rõ rệt. Report và raw artifacts: `research/increment_objective_11_9/`.

## Same-capacity temporal multiscale basis — FAIL, đóng manual-lag branch

Candidate dùng recursive lifting/Haar basis khả nghịch trên đúng20 input frames: một final coarse coefficient và19 detail coefficients ở năm scale cho mỗi velocity channel. Nó thay đúng40 history channels của sub8, vẫn giữ120 total channels và ZeroMean head55,528 parameters. Control first-difference và candidate multiscale được retrain matched với cùng seeds42/43, epoch permutations, optimizer, field+0.30 TKE loss, checkpoint proxy và ensemble. Normalization chỉ fit339 train windows; max reconstruction error dưới2e-7. Protocol và gate cao được khóa trước training.

| Metric | First-difference control | Multiscale | Thay đổi |
|---|---:|---:|---:|
| Selection proxy | 82.891440 | 82.879136 | −0.012304 point |
| RelL2 error | 0.077760 | 0.077987 | +0.292% |
| TKE error | 0.586955 | 0.587196 | +0.041% |
| MVPE error | 0.069346 | 0.069346 | 0.000% |
| Adaptive SPS | 47.742467 | 47.693481 | −0.048986 point |

Cả hai paired seeds đều xấu hơn ở RelL2 và TKE. Chỉ1/2 Re và2/5 AoA groups cải thiện TKE. Mean framewise RelL2 ở h6–15 tốt hơn0.263%, nhưng cumulative TKE không tốt hơn tại horizon nào trong10 mid horizons. Vì vậy statistic thuận lợi duy nhất không khớp energy-recovery mechanism.

**Quyết định: validation FAIL; holdout vẫn sealed, không package, không submit.** Lượt leaderboard cuối vẫn còn. Kết hợp evidence hiện tại: first-difference hidden PASS cho thấy adjacent-change accessibility hữu ích; increment-loss FAIL cho thấy scalar objective incentive không đủ; fixed Haar multiscale FAIL cho thấy không phải cứ trải thông tin qua nhiều hand-engineered timescales là tốt hơn.

## Evidence synthesis — cái gì đã biết, đã fail và phải dừng

Bảng tổng hợp canonical nằm ở `research/evidence_synthesis_11_9.md`. Evidence được tách thành hidden-verified, offline PASS, offline FAIL, diagnostic-only và blocked/untested để tránh biến local proxy thành leaderboard truth hoặc suy rộng một FAIL hẹp thành cả family.

### Hidden-verified

1. Deployed residual/TKE-specialized pipeline của sub6/sub7 cải thiện hidden TKE so original CNO `73.7175→74.8234`; đây là evidence cho cả pipeline, chưa cô lập ZeroMean/TKE loss/mask.
2. Adaptive SPS được clean hidden comparison sub6→sub7 trên identical points: SPS `35.5132→37.5790`, final `78.6395→79.1582`.
3. First-difference same-capacity được offline PASS rồi hidden-confirmed sub7→sub8: RelL2 `+0.030799`, TKE `+0.360288`, MVPE unchanged, final `+0.088143`. Kết luận:20-frame history có deterministic adjacent-change signal mà raw basis khai thác kém hơn; không phải thêm information và chưa chứng minh timescale mismatch.

### Diagnostic support nhưng chưa phải causal solution

- Sub7 giữ khoảng62.6% target fluctuation energy; underprediction chi phối TKE error.
- TKE-map cosine khoảng0.82 cho thấy spatial allocation cũng sai, nên scalar deficit không phải toàn bộ vấn đề.
- Boundary-only không giải thích gap chính; history-nearest-neighbor/copy-energy yếu hơn sub7.
- Real/Sim decorrelation khác nhau nhưng dynamics-vs-PIV observation cause vẫn unresolved.

### Interventions đã FAIL hoặc blocked

| Family | Evidence | Stop decision |
|---|---|---|
| Global amplitude | best feasible TKE −1.674%, RelL2 +0.763%, dưới gate | Dừng global rescale. |
| Channel amplitude | TKE −1.723%, RelL2 +0.835%, dưới gate | Dừng tune channel scalars. |
| Coarse-horizon amplitude | beta=0 thắng grid | Dừng hand-tuned horizon curve. |
| Small SPS conditional grids | gains khoảng+0.04 đến+0.065 validation, dưới gates | Không dùng lượt cuối cho SPS multiplier tweak. |
| Future-increment loss | proxy −0.0233, TKE +0.449%, increment chỉ −0.414% | Không stack/tune first-order increment loss. |
| Fixed Haar multiscale basis | proxy −0.0123, RelL2 +0.292%, TKE +0.041%, both seeds worse | Đóng manual lag/wavelet basis search. |
| Wider stride-2 support | hidden API không cung cấp frames cần thiết | Không train candidate undeployable. |

Không ưu tiên boundary-only mask, nearest-history copying hoặc injected noise khi chưa có predictive evidence. Không tuyên bố remaining TKE do phase, timescale hay PIV noise vì chưa identify được.

Các FAIL trên không bác bỏ learned spatial correction, adaptive SPS nói chung, mọi temporal objective hay multiscale architecture. Hypothesis còn mạnh nhất là **learned temporal inductive bias**: fast-dynamics path học từ adjacent deltas, slow/base path giữ state context, learned fusion, same capacity và giữ sub8 loss/SPS. Chỉ dùng lượt leaderboard cuối nếu matched paired-seed, subgroup và mid-horizon gates đều PASS.

Theo stop rule, đóng manual lag/wavelet tuning. Branch kế tiếp là fast-dynamics architecture có slow/base path và learned local-change path, vẫn phải same-capacity và vượt paired-seed, subgroup, mid-horizon gates trước khi dùng lượt cuối. Chi tiết và raw artifacts: `research/multiscale_representation_11_9/`.
