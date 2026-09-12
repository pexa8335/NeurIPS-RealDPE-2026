# Phân tích Hypothesis 03: Temporal Stationarity & History-Mean Forecasting Bias

## 1. Mục tiêu của hypothesis

Hypothesis 03 kiểm tra mức độ ổn định theo thời gian của velocity field trong dữ liệu RealPDE Track 1. Câu hỏi chính là liệu trung bình của 20 frame quan sát đầu tiên có thể được dùng làm một structural prior ổn định cho 20 frame tương lai hay không.

Cụ thể, notebook kiểm tra ba vấn đề:

1. Mean velocity field của cửa sổ `0:20` có gần mean của cửa sổ dự báo `20:40` không.
2. Mean velocity field có tiếp tục ổn định tại các thời điểm xa hơn, đến cửa sổ `580:600`, hay xuất hiện drift dài hạn.
3. Một baseline không có tham số, chỉ lặp lại history mean cho mọi future frame, đạt forecast RelL2 bao nhiêu.

Nếu mean field ổn định, model có thể dự báo theo dạng residual:

$$
\hat{\mathbf{u}}(t+h)
=
\bar{\mathbf{u}}_{0:20}
+
\mathcal{N}_\theta(\mathbf{u}_{0:20}),
$$

trong đó mạng chỉ cần học phần dao động động lực học quanh mean thay vì tái tạo toàn bộ velocity field từ đầu.

## 2. Lý do thực hiện

Trong một dòng chảy đã đi vào limit cycle hoặc chế độ shedding quasi-periodic ổn định, có thể phân rã velocity field thành:

\[
\mathbf{u}(t)=\bar{\mathbf{u}}+\mathbf{u}'(t),
\]

với `mean field` tương đối ổn định và fluctuation có mean gần zero theo thời gian.

Nếu giả định này đúng:

- History mean cung cấp sẵn global streamline topology.
- Model có thể dành capacity để học vortex propagation và fluctuation.
- Zero-initialized residual head bắt đầu từ một baseline hợp lý thay vì prediction ngẫu nhiên.
- Training có thể ổn định hơn và giảm nguy cơ catastrophic divergence.

Ngược lại, nếu dữ liệu có transient ban đầu hoặc secular drift, history mean của 20 frame đầu có thể tạo systematic bias cho toàn bộ forecast.

## 3. Các giả thuyết được notebook đặt ra

### Null hypothesis H0

Dòng chảy non-stationary; history mean khác future-window mean trên 10% relative L2, khiến static history-mean prior thất bại nghiêm trọng.

### Alternative hypothesis H1

Notebook đặt ba yêu cầu:

1. Các trajectory ở mọi Reynolds number đã ở chế độ statistically stationary trong toàn bộ 607 timestep.
2. Drift từ mean `0:20` đến mean `20:40` nhỏ hơn 3.5% ở mọi Reynolds number.
3. Drift ở horizon dài, bao gồm `580:600`, vẫn nhỏ hơn 5.5%.

Đây là các ngưỡng khá chặt. Để chấp nhận H1 như được phát biểu, output phải thỏa đồng thời các điều kiện trên.

## 4. Dữ liệu được kiểm tra

Notebook không kiểm tra toàn bộ dataset. Code chỉ chọn sáu trajectory Real:

| File danh nghĩa | Reynolds thực tế | AoA |
|---|---:|---:|
| `3750_0.h5` | 3750 | 0 |
| `3750_10.h5` | 3750 | 10 |
| `13950_0.h5` | 13977 | 0 |
| `13950_15.h5` | 13977 | 15 |
| `26700_0.h5` | 26761 | 0 |
| `26700_15.h5` | 26761 | 15 |

Ba mức Reynolds đại diện cho low-, mid- và high-Re. Mỗi mức có một trường hợp AoA bằng zero và một trường hợp AoA khác zero.

Do chỉ có sáu trajectory, kết quả là một sample audit, không đủ để chứng minh một phát biểu áp dụng cho tất cả Reynolds number và AoA trong dataset.

## 5. Phương pháp kiểm tra

### 5.1 History mean

Với mỗi trajectory, notebook tính mean của `u` và `v` trên 20 frame đầu:

\[
\bar{u}_{0:20}=\frac{1}{20}\sum_{t=0}^{19}u_t,
\qquad
\bar{v}_{0:20}=\frac{1}{20}\sum_{t=0}^{19}v_t.
\]

### 5.2 Window-mean drift

History mean được so sánh với mean của các cửa sổ:

- `20:40`
- `100:120`
- `200:220`
- `300:320`
- `400:420`
- `500:520`
- `580:600`

Drift được tính bằng:

\[
\delta_w=
\frac{
\sqrt{\sum[(\bar{u}_w-\bar{u}_{0:20})^2+(\bar{v}_w-\bar{v}_{0:20})^2]}
}{
\sqrt{\sum[\bar{u}_{0:20}^2+\bar{v}_{0:20}^2]}
}.
\]

### 5.3 History-mean forecast baseline

Notebook tạo prediction bằng cách lặp lại history mean cho cả 20 future frame:

\[
\hat{u}_t=\bar{u}_{0:20},
\qquad
\hat{v}_t=\bar{v}_{0:20},
\quad t=20,\ldots,39.
\]

Sau đó tính forecast RelL2 trên toàn bộ future window:

\[
\mathrm{RelL2}=
\frac{
\sqrt{\sum[(u_{target}-u_{pred})^2+(v_{target}-v_{pred})^2]}
}{
\sqrt{\sum[u_{target}^2+v_{target}^2]}
}.
\]

## 6. Output thực tế

### 6.1 Kết quả từng điều kiện

| Điều kiện | History-mean forecast RelL2 | Drift `20:40` | Drift `100:120` | Drift `200:220` | Drift `300:320` | Drift `400:420` | Drift `500:520` | Drift `580:600` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Re=3750, AoA=0 | 0.110886 | 9.51% | 9.49% | 11.95% | 10.75% | 7.17% | 11.31% | 7.54% |
| Re=3750, AoA=10 | 0.162015 | 14.37% | 16.46% | 13.37% | 16.18% | 17.44% | 15.88% | 18.38% |
| Re=13977, AoA=0 | 0.046345 | 2.79% | 3.41% | 3.19% | 2.60% | 2.88% | 2.71% | 2.40% |
| Re=13977, AoA=15 | 0.113293 | 8.00% | 8.21% | 8.72% | 8.30% | 6.73% | 7.78% | 8.29% |
| Re=26761, AoA=0 | 0.067764 | 4.52% | 3.91% | 5.53% | 4.57% | 5.49% | 4.86% | 5.26% |
| Re=26761, AoA=15 | 0.164781 | 9.28% | 12.18% | 12.13% | 12.30% | 13.34% | 11.71% | 8.88% |

### 6.2 Summary do code in ra

```text
Mean drift from window (0:20) to next window (20:40): 8.08%
Maximum immediate drift: 14.37%

Mean drift from window (0:20) to final window (580:600): 8.46%
Maximum final-window drift: 18.38%

Average history-mean forecast RelL2: 0.1108
```

## 7. Đối chiếu output với verdict trong notebook

Markdown cuối notebook ghi `VERDICT: ACCEPTED` và tuyên bố:

- Immediate drift nằm trong khoảng 1.68-2.87%, trung bình 2.14%.
- Long-term drift dưới 4.5%.
- History-mean forecast RelL2 khoảng 0.131.
- Strict quasi-stationarity được xác nhận ở mọi điều kiện.

Các tuyên bố này không khớp với output của code trong chính notebook:

| Đại lượng | Markdown verdict | Output code |
|---|---:|---:|
| Mean immediate drift | 2.14% | 8.08% |
| Maximum immediate drift | 2.87% | 14.37% |
| Maximum long-term drift | dưới 4.5% | 18.38% |
| Average history-mean RelL2 | 0.131 | 0.1108 |

Vì execution count của code là `1` và output đã được lưu, output code là bằng chứng định lượng cần ưu tiên. Phần verdict có khả năng được viết từ một phiên chạy, cách chuẩn hóa hoặc tập mẫu khác rồi chưa được cập nhật.

Ngoài ra, chuỗi in cuối code chứa:

```python
print(f"... {computed_mean:.4f} (approx 0.131)")
```

Số `0.131` là text viết cứng, không phải giá trị được tính trong lần chạy hiện tại. Giá trị thực tế được tính là `0.1108`.

## 8. Đánh giá hypothesis

### 8.1 H1 không được output hỗ trợ

H1 yêu cầu immediate drift dưới 3.5% và long-term drift dưới 5.5% ở mọi điều kiện. Output vi phạm rõ cả hai ngưỡng:

- Năm trên sáu điều kiện có immediate drift lớn hơn 3.5%.
- Bốn trên sáu điều kiện có final-window drift lớn hơn 5.5% nếu tính chính xác theo output; Re=26761, AoA=0 cũng gần ngưỡng ở 5.26%.
- Maximum immediate drift là 14.37%.
- Maximum final drift là 18.38%.

Do đó không thể chấp nhận phát biểu `strict quasi-stationarity across all examined conditions` theo các ngưỡng notebook đã định trước.

### 8.2 H0 cũng không được xác nhận hoàn toàn

H0 cho rằng drift lớn hơn 10% và history-mean prior thất bại nghiêm trọng. Output cũng không hỗ trợ phát biểu này cho mọi điều kiện:

- Immediate drift trung bình chỉ 8.08%, không vượt 10%.
- Một số điều kiện có drift rất thấp, đặc biệt Re=13977, AoA=0 chỉ khoảng 2.4-3.4%.
- History-mean baseline đạt RelL2 trung bình 0.1108, nên không thể gọi là catastrophic failure.

Vì vậy kết quả không phù hợp hoàn toàn với cấu trúc nhị phân H0/H1 của notebook.

### 8.3 Kết luận phù hợp nhất

Kết quả hỗ trợ một kết luận trung gian:

> Velocity field thể hiện quasi-stationarity phụ thuộc điều kiện. History mean là một structural prior hữu ích, nhưng không phải một mean bất biến với drift dưới 3.5-5.5% cho mọi Reynolds number và AoA.

AoA có vẻ liên quan mạnh đến drift trong sáu mẫu:

- Tại Re=13977, immediate drift tăng từ 2.79% ở AoA=0 lên 8.00% ở AoA=15.
- Tại Re=26761, immediate drift tăng từ 4.52% ở AoA=0 lên 9.28% ở AoA=15.
- Tại Re=3750, immediate drift tăng từ 9.51% ở AoA=0 lên 14.37% ở AoA=10.

Đây chỉ là pattern quan sát trong sáu trajectory. Chưa thể kết luận AoA là nguyên nhân nếu chưa kiểm tra toàn bộ trajectory và kiểm soát Reynolds number.

## 9. Những kết luận không thể suy ra từ experiment này

### 9.1 Không thể khẳng định không có initial transient

Notebook chỉ so sánh các window mean với mean của `0:20`. Sai khác nhỏ hoặc không tăng đơn điệu chưa chứng minh rằng cửa sổ đầu đã hoàn toàn thoát transient.

Để kiểm tra transient cần thêm các đại lượng như:

- Moving mean và moving variance theo thời gian.
- Xu hướng drift có dấu hoặc slope theo time.
- So sánh nhiều cửa sổ liên tiếp thay vì chỉ dùng `0:20` làm reference.
- Kiểm tra dependence của kết quả vào window length.

### 9.2 Không thể phân biệt secular drift với phase/windowing effect

Cửa sổ 20 frame có thể không chứa số chu kỳ vortex shedding nguyên vẹn. Khi đó mean của hai cửa sổ khác nhau có thể lệch nhau do phase sampling, dù process dài hạn vẫn stationary.

Vì vậy drift của 20-frame window mean có thể bao gồm:

- Secular drift thực sự.
- Dao động chậm.
- Phase mismatch.
- Window quá ngắn so với shedding period.

### 9.3 RelL2 không phải tỷ lệ năng lượng được giải thích

Notebook tuyên bố RelL2 `0.131` chứng minh trên 86% total field energy nằm trong stationary mean. Suy luận này không đúng về mặt toán học.

RelL2 là tỷ lệ norm của residual trên norm target:

\[
r=\frac{\|y-\hat y\|_2}{\|y\|_2}.
\]

Không thể lấy `1-r` làm tỷ lệ năng lượng được giải thích. Ngay cả `1-r^2` chỉ có thể được diễn giải như explained squared norm trong các điều kiện phân rã phù hợp; notebook chưa kiểm tra trực giao hoặc decomposition cần thiết.

Do đó output chỉ cho phép nói history-mean forecast có field RelL2 tương đối thấp, không cho phép nói chính xác bao nhiêu phần trăm field energy nằm trong mean.

### 9.4 Không thể khái quát cho toàn bộ dataset

Chỉ sáu trajectory được kiểm tra. Phát biểu `across all Reynolds numbers` hoặc `across all conditions` vượt quá phạm vi evidence.

## 10. Ý nghĩa đối với kiến trúc model

Experiment vẫn cung cấp một insight hữu ích: history mean là baseline đủ mạnh để cân nhắc residual formulation.

Một kiến trúc có thể dùng:

\[
\hat{\mathbf{u}}_{future}
=
\bar{\mathbf{u}}_{history}
+
\Delta\mathbf{u}_\theta.
\]

Lợi ích tiềm năng:

- Giữ sẵn phần spatial structure lớn của dòng chảy.
- Model tập trung vào dynamic fluctuation.
- Zero-initialized final layer tái tạo chính xác history-mean baseline tại epoch zero.
- Có một fallback hợp lý nếu residual head không học được cải thiện.

Tuy nhiên, notebook chưa chứng minh model `NEVER` nên dự báo raw field. Đây là một đề xuất kiến trúc cần được kiểm tra bằng matched ablation:

- Raw-field prediction.
- History-mean residual prediction.
- Cùng architecture, capacity, split, loss và training budget.

Chỉ khi residual formulation cải thiện validation/test ổn định mới có thể kết luận nó tốt hơn.

## 11. Verdict cuối cùng

### Verdict của notebook

`ACCEPTED`, nhưng không nhất quán với output code.

### Verdict sau khi phân tích output

**Strict H1: REJECTED.** Các ngưỡng drift được đặt trước không được thỏa mãn.

**H0: NOT FULLY SUPPORTED.** Một số điều kiện có drift trên 10%, nhưng không phải tất cả; history-mean forecast cũng không thất bại nghiêm trọng.

**Kết quả thực nghiệm chính:**

- History mean là một prior hữu ích với forecast RelL2 trung bình `0.1108` trên sáu trajectory.
- Mean field không ổn định ở mức chặt như notebook tuyên bố.
- Drift phụ thuộc đáng kể vào điều kiện, đặc biệt có pattern tăng tại AoA khác zero.
- Cần mô tả dữ liệu là condition-dependent quasi-stationary thay vì strict stationary.

## 12. Tóm tắt ngắn

Hypothesis 03 kiểm tra liệu mean của 20 frame history có ổn định đủ để làm mốc cho dự báo tương lai hay không. Code đo drift của history mean qua nhiều cửa sổ đến timestep 600 và đánh giá baseline lặp lại history mean. Output cho immediate drift trung bình 8.08%, long-term drift trung bình 8.46%, maximum 18.38%, và history-mean forecast RelL2 0.1108. Vì các giá trị này vượt xa ngưỡng H1 là 3.5% và 5.5%, verdict `ACCEPTED` trong notebook không đúng với output. Kết luận phù hợp là history mean hữu ích nhưng stationarity phụ thuộc Reynolds/AoA và không đủ mạnh để xem mean là bất biến trong mọi điều kiện.
