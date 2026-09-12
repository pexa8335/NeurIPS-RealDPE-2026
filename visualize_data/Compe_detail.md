# Findings in EDA.

## 1. "Paired" không có nghĩa là khớp Re/AoA chính xác
- **Sim**: Re = {3750, 5025, 6300, 7575, 8850, ...} — các số tròn, "thiết kế" (nominal setpoints)
- **Real**: Re = {3750, 5028, 6306, 7585, 8863, ...} — các số lệch nhẹ so với sim (5028 vs 5025, 6306 vs 6300...)
- **Chỉ có 4 cặp khớp chính xác** trong số tối đa 82 file real!


**Ý nghĩa cho modeling**: bạn không thể dùng chiến lược "paired translation" kiểu ghép từng cặp (real_i ↔ sim_i) theo đúng key Re/AoA. Thay vào đó nên:
- Coi (Re, AoA) như biến điều kiện liên tục — model học ánh xạ hàm số theo tham số, không phải tra bảng
- Ghép cặp gần đúng theo nearest-neighbor trên (Re, AoA) nếu cần một baseline pairing thô
- Hoặc dùng domain adaptation kiểu unpaired (model học biểu diễn chung, không cần từng cặp khớp tuyệt đối)

## 2. Real thiếu channel áp suất `p`
- Sim có: `TKE_proxy, p, speed, u, v, vorticity_proxy`
- Real có: `TKE_proxy, speed, u, v, vorticity_proxy` — **không có `p`**
→  sim có p, real thì không.

**Ý nghĩa**: 
- Model của bạn có thể học dùng `p` từ sim như tín hiệu phụ trợ (auxiliary supervision) khi pretrain, nhưng khi infer trên real thì input/output không có `p` — cần thiết kế model chấp nhận **thiếu modality** giữa hai domain.
- Việc metric chính thức (Rel L2, TKE, MVPE) không dùng p trực tiếp cũng khớp với thực tế này.

## 3. Độ dài trajectory không đồng nhất
- **Sim**: tất cả đều dài đúng 1000 bước — vì mô phỏng số dễ chạy đủ số bước cố định.
- **Real**: dao động 282–868 bước — vì thí nghiệm thật bị giới hạn bởi thời gian chạy tunnel, dung lượng camera, v.v.

**Ý nghĩa**: dataloader cần xử lý **chiều dài chuỗi biến thiên** cho real (padding/masking hoặc crop theo window cố định), trong khi sim có thể xử lý đơn giản hơn.

## 4. Độ phủ lưới tham số không đầy đủ ở real
- Sim: 20 Re × 5 AoA = lưới đầy đủ 100 tổ hợp, khớp đúng 100 file.
- Real: chỉ có 82/90 tổ hợp (18 Re × 5 AoA), tức **thiếu 8 tổ hợp** — ví dụ Re=17775 thiếu AoA=0, Re=26700 thiếu AoA=5 và 20...

**Ý nghĩa**: có những vùng tham số real hoàn toàn không được quan sát → mô hình phải **ngoại suy (extrapolate)**, không chỉ nội suy, đúng với mô tả "novel angles-of-attack/Reynolds numbers" trong private test set.

## Tổng kết — bài học rút ra cho chiến lược modeling
1. Đừng thiết kế theo hướng "ghép cặp chính xác" — hãy coi bài toán là **domain adaptation có điều kiện liên tục** trên (Re, AoA).
2. Model cần xử lý được **input/output thiếu modality** (p chỉ có ở sim).
3. Pipeline dữ liệu cần **hỗ trợ độ dài chuỗi biến thiên**.
4. Cần chiến lược tốt cho **ngoại suy** ra ngoài vùng tham số đã thấy ở real, vì test set chắc chắn sẽ test đúng khả năng này.

# Findings 2.

## Độ lệch giữa sim(re) và real(re) ko random
Ý nghĩa quan trọng: vì độ lệch là hệ thống (không phải nhiễu), bạn hoàn toàn có thể:

Fit một hàm hiệu chỉnh đơn giản Re_real ≈ f(Re_sim) (gần như tuyến tính, có thể fit bằng linear regression 1 biến) để map ngược giữa 2 domain khi cần.
Đây cũng gợi ý rằng khi model dự đoán trên tập test thực (private, Re/AoA mới), bạn nên cân nhắc dùng Re hiệu chỉnh thay vì Re nominal khi encode điều kiện đầu vào, để giảm mismatch giữa train (sim, có Re sạch) và test (real, có Re lệch).

## Type mismatch
Real: Hai trường vận tốc $u, v$ được lưu dưới dạng float64 (Double precision).  
Sim: Toàn bộ $u, v, p$ lại lưu dưới dạng float32 (Single precision).

## Miss Re có hệ thống

Sim phủ kín lưới chữ nhật hoàn hảo ($20 \text{ Re} \times 5 \text{ AoA} = 100$ files).Real bị khuyết có hệ thống:Khuyết hoàn toàn 2 mốc Re: $Re = 15.225$ và $Re = 27.975$ (trong khi Sim có đủ cả 5 góc AoA cho 2 mốc này).Khuyết góc tấn ở Re cao: Ở các dải $Re \ge 22.926$, góc $\text{AoA} = 5^\circ$ biến mất hoàn toàn; góc $\text{AoA} = 20^\circ$ bị khuyết ở 22.926 và 26.761.Ý nghĩa: Rất nhiều khả năng ban tổ chức đã giữ lại các ca này làm Hidden Validation / Private Test để kiểm tra:Khả năng nội suy (interpolation) tại $Re = 15.225$.Khả năng ngoại suy (extrapolation) tại $Re = 27.975$.Khả năng khái quát hóa góc tấn khi luồng khí bắt đầu tách dòng (flow separation) mạnh ở góc lớn và vận tốc cao.Giải pháp: Khi chia tập Local Cross-Validation, bạn nên tách riêng ra 1 mốc Re (ví dụ giữ lại toàn bộ $Re = 13.950$ hoặc $16.500$) làm validation set thay vì cắt ngẫu nhiên theo timestep.

## Time trajectory mismatch
Sự không đồng đều về độ dài chuỗi ($T$) & Cắt Sliding WindowSim có $T = 1.000$ cố định cho toàn bộ 100 files.Real có 79 files đạt $T = 868$, nhưng có 3 files dị biệt:10125_0.h5: $T = 607$21600_5.h5: $T = 492$24150_20.h5: $T = 282$Quy chuẩn cuộc thi: Đề bài yêu cầu cửa sổ trượt $T_{\text{in}} = 20 \rightarrow T_{\text{out}} = 20$ (tổng chiều dài cửa sổ $W = 40$).Vì file ngắn nhất $T = 282 > 40$, tất cả 81 files Real hợp lệ đều cắt được window an toàn ($282 - 40 + 1 = 243$ windows).Nếu dùng stride=1, tập Real sinh ra khoảng $66.000$ mẫu, tập Sim sinh ra hơn $96.000$ mẫu. Để tránh quá tải RAM và tăng tốc huấn luyện trên Kaggle, bạn nên đặt stride = 5 hoặc 10 trong các epoch đầu.

## Tọa độ u, v được lưu dưới x, y cố định

Cả Real và Sim đều lưu kèm dataset x và y với kích thước cố định (64, 128).

Chiến lược: Đừng chỉ nạp vận tốc $[u, v]$ thuần túy vào mạng. Hãy chuẩn hóa $x, y$ về $[-1, 1]$ và nối (concatenate) chúng thành kênh tọa độ đầu vào:$$\text{Input Tensor} = [u, v, x_{\text{norm}}, y_{\text{norm}}] \quad (\text{shape: } B, T_{\text{in}}, H, W, 4)$$Cách làm này giúp các mô hình Neural Operator (như FNO, Transolver, CNO) nhận biết chính xác vị trí mép trước (leading edge), mép sau (trailing edge) và bề mặt cánh airfoil.

## Thiếu p ở mọi file train real
Modality: xác nhận chắc chắn p chỉ có ở sim, và cấu trúc dữ liệu rất sạch/nhất quán
Sim: mọi file đều có đúng bộ u, v, p, x, y, t, re, aoa — shape chuẩn (1000, 64, 128).
Real: mọi file đều có đúng u, v, x, y, t, re, aoa — không có p, temperature — shape khớp với T riêng của từng file.

→ Việc thiếu p là nhất quán 100% (không phải thiếu ngẫu nhiên ở vài file) — nên bạn có thể thiết kế kiến trúc cố định: encoder chung học từ (u,v), decoder sim học thêm dự đoán p làm auxiliary task, còn khi infer trên real thì chỉ cần nhánh (u,v).

# Finding 3.

## Spatial alignment exact.

Chung tọa độ x, y cho cả sim và real, sai số chỉ khác nhau ở fp32 và fp64 (rất nhỏ) -> ko cần nội suy/ngoại suy ko gian hay regriding khi domain adaption sim->real. Ma trận (x, y) có thể chuẩn hóa và dùng chung.

## Important - dt của sim và real ko bằng nhau.

dt của sim và real KHÔNG bằng nhau
real_dt_mean = 0.02000000   (constant, std ≈ 4e-7 → gần như tuyệt đối)
sim_dt_mean  = 0.02002002   (constant, std ≈ 1e-17 → tuyệt đối chính xác)

Chênh lệch chỉ 0.02002 − 0.02 = 0.00002 giây/bước (~0.1% relative) — cực nhỏ, nhưng vì đây là sai số hệ thống tích lũy theo từng bước, nó cộng dồn tuyến tính theo thời gian:

File đầy đủ 868 bước: real_t_end = 17.42s nhưng sim_t_end = 20.08s — sim "chạy nhanh hơn" theo đồng hồ danh nghĩa, tích lũy chênh lệch lên tới ~2.66 giây sau 868 bước.
max nearest-time error = 0.010s — đúng bằng nửa chu kỳ dt (0.02/2 = 0.01) → đây chính là dấu hiệu kinh điển của drift tích lũy vượt quá nửa bước, khiến việc map "gần nhất" bắt đầu bị trùng index.

### Cách xử lí (not sure, chưa ra qdinh)
Tuyệt đối không được ghép cặp theo index i trực tiếp (real[i] ↔ sim[i]) nếu bạn cần cặp thời gian chính xác — sẽ sai lệch tăng dần, tới nửa bước thời gian ở cuối trajectory dài.
Nếu cần alignment theo thời gian thực (t), phải dùng nearest-neighbor hoặc linear interpolation theo t (giống code bạn đã viết) — và nó rẻ vì cả hai đều gần như đều đặn (dt gần constant), có thể tính closed-form thay vì searchsorted cho nhanh:
python
   nearest_idx_sim = round((t_real - t0) / dt_sim)
Sai số alignment tối đa chỉ 0.01s / dt_sim ≈ 0.5 bước — quá nhỏ so với timescale vật lý của dòng chảy (Re~10⁴, tần số shedding thường ở scale giây) nên không đáng lo về mặt vật lý, nhưng vẫn cần xử lý đúng kỹ thuật để không tích lũy lỗi khi tạo sliding window dài.
Nếu bạn dùng chiến lược "pretrain trên sim, fine-tune trên real" mà cần cùng bước thời gian giữa 2 domain (ví dụ multi-task loss dùng cả 2 batch cùng lúc theo cùng chỉ số t), bạn nên resample sim về đúng lưới thời gian của real (hoặc ngược lại) bằng interpolation, thay vì giả định index khớp nhau.

## Đéo hiểu lắm, lưu trc.

4. Tính bất biến tịnh tiến thời gian (Time-Translation Invariance)Dữ liệu Real và Sim đều bắt đầu từ $t = 0.08\,\text{s}$ (sau khi dòng chảy đã vượt qua giai đoạn khởi động chuyển tiếp và đạt trạng thái phát triển đầy đủ - fully developed flow).Quy tắc tiền xử lý: Cột mốc thời gian tuyệt đối $t$ trong file .h5 không mang giá trị làm đặc trưng đầu vào. Phương trình Navier-Stokes trong điều kiện dòng dừng (steady incoming flow) có tính bất biến thời gian:$$\mathcal{F}(X_{t : t+20}) = X_{t+20 : t+40}$$Mô hình chỉ cần học toán tử chuyển dịch trạng thái $\Delta t = 20 \times 0.02\,\text{s}$, không cần nạp vector thời gian tuyệt đối $t$ vào mạng.

### Decision of gemini (chưa verify)
Tổng kết các quyết định đưa vào mã nguồnThành phầnQuyết định kỹ thuậtTọa độ $(x, y)$Chuẩn hóa Min-Max về $[-1, 1]$ và đưa vào mô hình làm static positional embedding (Shape: $2 \times 32 \times 64$).Kênh thời gian $t$Bỏ qua hoàn toàn mảng t. Coi mọi chuỗi đầu vào $T_{\text{in}}=20$ là $t \in [0, 19]$ với $dt=1$.Chiến lược học Sim2RealHuấn luyện toán tử động lực học ($S_{t+1} = \mathcal{M}(S_t)$) trên CFD trước, sau đó fine-tune thích nghi phân phối (distribution adaptation) trên PIV, thay vì ép khớp cặp trạng thái tức thời.

### Decision of claude (not verified)
Tóm tắt insight chính
Phát hiện	Ý nghĩa
Lưới không gian khớp gần tuyệt đối	Không cần regrid không gian
dt real=0.02 vs dt sim=0.02002 (lệch hệ thống, không nhiễu)	Cần align theo t, không align theo index
Drift tích lũy vượt dt/2 sau ~500 bước	Giải thích chính xác vì sao hầu hết file có 1 index bị trùng khi map gần nhất
2 file ngắn (Tr<500) mapping hoàn hảo	Xác nhận công thức drift tuyến tính là đúng, không phải lỗi ngẫu nhiên

## Finding 4.

1. Correlation & Rel L2 tổng thể: khá tệ, và giảm dần theo Re
File	Re	corr u	corr v	relL2 u	relL2 v
3750_0	thấp	0.36	0.12	17.8	41.7
26700_10	cao	0.60	0.33	2.28	11.6

relL2 > 1 nghĩa là sai số lớn hơn cả norm của tín hiệu thật — về mặt point-wise, sim gần như "không dự đoán được" real ở Re thấp. Ở Re cao thì đỡ hơn nhưng vẫn corr chỉ ~0.6.
(only finding, đéo biết lmj, ch hỏi).

## what shud i do with this info?! huhu
1. Dao động tức thời hoàn toàn lệch pha (fluc_u_corr $\approx 0$)Hệ số tương quan của trường dao động tức thời ($u' = u - \bar{u}$) giữa Sim và Real trên cả 5 ca đều xấp xỉ bằng 0 (dao động từ $-0.029$ đến $+0.020$).Phép thử dịch chuyển thời gian (best_lag_frames) cho kết quả ngẫu nhiên ở mỗi ca ($+15, +70, +90, +25, -35$ frames) và tương quan cực đại vẫn rất thấp ($\approx 0.25 - 0.49$).

### Gemini decision (not verified)
Bản chất vật lý: Dòng chảy khí động học quanh cánh NACA4418 là dòng chảy rối/hỗn loạn (turbulent/chaotic flow). Dù chung góc tấn và số Reynolds, các xoáy vi mô được sinh ra ngẫu nhiên theo thời gian thực trong hầm gió PIV và không bao giờ đồng pha (phase-locked) với mô phỏng CFD số.Quy tắc ML bắt buộc: Tuyệt đối không dùng kỹ thuật Paired Supervision ép frame-to-frame giữa Sim và Real ($\mathcal{L} = \Vert{}u_{\text{real}}(t) - u_{\text{sim}}(t)\Vert{}$). Nếu ép mạng học điểm-nối-điểm giữa hai file cùng tên, mô hình sẽ bị phân kỳ (collapse) và chỉ dự báo ra trường vận tốc nhòe trung bình.

## vai lon finding j day
2. Cảnh báo đỏ về độ lệch thang đo (relL2 cực lớn: 228% – 4167%)Sai số tương đối relL2 giữa Real và Sim ở mức bất thường:$Re = 3750$: relL2 u = 17.81 ($1781\%$), relL2 v = 41.67 ($4167\%$).$Re = 26700$: relL2 u = 2.28 ($228\%$), relL2 v = 11.56 ($1156\%$).Nghịch lý toán học: Hệ số tương quan trường trung bình (mean_u_corr, mean_v_corr) đạt mức dương khá cao ($0.65 - 0.72$), nhưng relL2 lại bùng nổ hàng nghìn phần trăm.Nguyên nhân cốt lõi: Lệch thang đo vật lý (Scale / Unit Mismatch).Tương quan Pearson độc lập với độ phóng đại ($r(aX + b, Y) = r(X, Y)$), nhưng $L_2$ phụ thuộc trực tiếp vào biên độ.Hiện tượng này chứng minh dữ liệu CFD mô phỏng (train_sim) và dữ liệu PIV thực nghiệm (train_real) chưa được quy chuẩn về cùng một thứ nguyên (ví dụ: một bên là vận tốc thực $\text{m/s}$, một bên là vận tốc không thứ nguyên $u / U_\infty$; hoặc do chuẩn hóa mật độ).Giải pháp cấp bách: Bạn cần in ngay giá trị $\min, \max, \text{mean}, \text{std}$ của $u, v$ giữa 2 file để tìm hệ số tỉ lệ $\alpha \approx \frac{\text{std}(u_{\text{real}})}{\text{std}(u_{\text{sim}})}$ và chuẩn hóa Z-score riêng biệt từng miền trước khi nạp vào mạng.

## lag inconsistency

2. Best lag không nhất quán → đây KHÔNG phải vấn đề lệch pha đơn giản

Nếu chỉ là do drift thời gian (như bạn phát hiện ở bước trước), ta kỳ vọng lag tối ưu sẽ giống nhau hoặc có xu hướng đơn điệu theo Re/thời lượng file. Nhưng thực tế:

3750_0:    lag = +15 frames (+0.3s)
8850_10:   lag = +70 frames (+1.4s)
13950_15:  lag = +90 frames (+1.8s)
20325_20:  lag = +25 frames (+0.5s)
26700_10:  lag = -35 frames (-0.7s)

Dấu và độ lớn nhảy lung tung, kể cả đổi dấu (âm ở file cuối) → không tồn tại một "global time offset" cố định giữa 2 domain. Đây là dấu hiệu của shedding phase ngẫu nhiên, không phải lỗi đồng bộ đồng hồ.

## hiu ung' canh buom' ?
3. Đây là insight cốt lõi: Mean field khớp tốt, nhưng fluctuation (nhiễu loạn) hoàn toàn KHÔNG khớp

So sánh 3 tầng:

Thành phần	Correlation	Ý nghĩa
Mean field (trung bình theo t)	0.38 → 0.72 (tăng theo Re)	Cấu trúc dòng chảy trung bình (wake shape, vortex trung bình) khớp khá tốt
Fluctuation (u - mean, tức thời)	≈ 0 (dao động quanh 0, có âm)	Các dao động turbulence tức thời hoàn toàn không tương quan — như nhiễu độc lập
RMS fluctuation (biên độ dao động,	u'	)

→ Đây chính là bản chất vật lý của dòng chảy hỗn loạn quanh vật cản (vortex shedding): nó là hệ hỗn loạn (chaotic). Sim và real cùng xuất phát từ cùng điều kiện biên (Re, AoA) nên sẽ cho ra cùng thống kê (mean flow, RMS/TKE, tần số shedding trung bình) — nhưng pha chính xác của từng xoáy tại từng thời điểm là không thể dự đoán được, vì sai khác cực nhỏ ban đầu (điều kiện khởi tạo, nhiễu đo đạc...) sẽ khuếch đại theo thời gian (sensitivity to initial conditions — đặc trưng kinh điển của chaos/turbulence).

## Insight deo j nx day nhin co ve qtrong
Ý nghĩa cực kỳ quan trọng cho chiến lược của bạn

Đây chính là lý do tại sao competition thiết kế bộ metric như vậy, giờ bạn đã hiểu rõ tại sao:

Không có metric nào là "point-wise instantaneous L2" thuần túy trên trường đầy đủ — vì chính họ biết matching tức thời giữa real và sim gần như không khả thi vượt quá vài bước ngắn.
TKE (turbulent kinetic energy) — chính là metric đo "RMS fluctuation" mà bạn vừa thấy khớp khá tốt (0.48–0.92) → đây là điều học được.
Mean Velocity Profile Error (MVPE) — đo trường trung bình dài hạn → đúng thành phần "mean field" khớp tốt (0.38–0.72) mà bạn vừa kiểm chứng → đây cũng là điều học được.
Relative L2 Error (metric core) — chắc chắn sẽ được tính trên horizon ngắn (autoregressive rollout, không phải toàn bộ 868 bước), vì sai số point-wise tăng theo cấp số nhân với horizon do chaos.

Hệ quả trực tiếp cho thiết kế model
Đừng kỳ vọng model dự đoán chính xác pha xoáy tức thời ở horizon dài — đây là giới hạn vật lý cơ bản (chaos), không phải do model yếu.
Chiến lược tối ưu nhiều khả năng là:
Học tốt mean flow + spectral/statistical properties (TKE, tần số shedding) làm mục tiêu chính cho horizon dài.
Chỉ kỳ vọng độ chính xác point-wise cao ở horizon ngắn (vài bước đầu autoregressive), giảm dần theo thời gian.
Cân nhắc loss function kết hợp cả point-wise (ngắn hạn) + statistical loss (TKE, spectrum, mean profile) thay vì thuần MSE trên toàn trajectory.
"Safe Prediction Score" trong metric chính thức (dự đoán khoảng tin cậy, không phải giá trị đơn) cũng khớp hoàn hảo với phát hiện này — vì bản chất bài toán đòi hỏi model biết định lượng độ bất định của mình khi horizon dài, thay vì cố "đoán đúng" một giá trị không thể đoán được.

Bạn có muốn mình đề xuất cụ thể một loss function kết hợp (point-wise ngắn hạn + TKE/spectral matching + uncertainty calibration cho SPS) phù hợp với những gì vừa phát hiện không?


