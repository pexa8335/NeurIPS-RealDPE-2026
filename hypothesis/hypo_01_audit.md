# Hypothesis 01 — Kiểm chứng lại grid, zero-mask và hình học

Ngày 2026-09-10. Đã đọc trực tiếp **182 HDF5: 82 Real + 100 Sim** trong `archive.zip`; chạy hoàn tất trong **332.0 giây**. Không train, sửa model, chọn ngưỡng triển khai hay chạy gate tiếp theo. Notebook gốc và ZIP sub7 giữ nguyên SHA-256. Đây là diagnostic trên dữ liệu đã có, không phải unseen confirmation.

## Kết luận

1. **Grid gần trùng, không trùng tuyệt đối:** giữ được nhận định thực dụng về grid dùng chung, nhưng phải ghi rõ tolerance. Giả thuyết ngưỡng `1e-5` không đạt cho Sim–Real; ngưỡng `1e-4` đạt. Không đổi ngưỡng sau khi xem kết quả.
2. **Zero-mask không thể được xem là một mask hình học cố định:** cần phân biệt tập zero tại từng frame, tập zero suốt trajectory và tập zero trong history20. Kiểm tra theo Re phải dùng metadata thực.
3. **Chưa xác lập pixel zero là airfoil/optical shadow:** dữ liệu và kiểm tra connectivity chỉ chứng minh đặc tính của mask quan sát được. Largest component có thể chọn vùng khác hẳn vùng Sim ở AoA10/15.
4. **Không có bằng chứng mới để đổi sub7:** kết luận Phase1 về lợi ích offline nhỏ và các regression của guardrail vẫn giữ nguyên. Audit này không giải quyết thiếu provenance training của G0.

## 1. Grid

| Domain | Files | Số frame min–max | max abs Δx so ref Real | max abs Δy so ref Real | ≤1e-5 | ≤1e-4 |
| --- | --- | --- | --- | --- | --- | --- |
| Real | 82 | 282–868 | 9.2e-08 | 7.9e-08 | PASS | PASS |
| Sim | 100 | 1000–1000 | 1.0832e-05 | 1.9936e-05 | FAIL | PASS |

Tất cả shape UV là `(T,64,128)` và coordinate `(64,128)`: **True**; dữ liệu/coordinate finite: **True**. Ref là Real `10125_0.h5`. Frame count lấy từ từng file, không áp đặt mọi trajectory có607 frame. Metadata Re của file này là 10142, không phải 10125.

* Median spacing theo các file: dx trong [0.001710832,0.00171086614], dy trong [-0.00171111111,-0.001710832].
* Sai khác coordinate cực đại tương đương khoảng **0.00633 cell theo x**, **0.01165 cell theo y**.
* Sai lệch spacing tương đối lớn nhất quanh median: x=5.84511e-07, y=5.84511e-07. Cross-axis variation lớn nhất: x=0, y=0. Monotone đúng cho mọi file: x=True, y=True.

Bằng chứng này hỗ trợ grid xấp xỉ đều và gần đồng nhất trong tọa độ được lưu. Coordinate chỉ có hai chiều không ghi chuyển động theo thời gian; không thể dùng nó để khẳng định thiết bị/geometry ngoài dữ liệu không chuyển động. Cũng chưa phải một thực nghiệm chứng minh mọi kiến trúc dùng grid sẽ cho chất lượng tốt.

## 2. Định nghĩa mask và độ nhạy ngưỡng

**Exact persistent-zero:** `all_t(u==0 and v==0)`. **Constant:** `max_t(u)==min_t(u)` và tương tự v; có thể constant nhưng nonzero. **Near-zero:** `max_t(abs(u))<epsilon` và tương tự v. Phân tích trước với exact/`1e-12`/`1e-10`/`1e-8`; không chọn một epsilon mới cho inference. `std==0` còn phụ thuộc số học reduction/underflow nên không phải phép kiểm tra zero trực tiếp.

Các case nominal10125 có **Real actual Re10142, Sim actual Re10125**; đây là cặp theo tên file, không phải hai phép đo có Re thực bằng nhau:

| AoA | Real exact-zero | Sim exact-zero | Real near-zero 1e-10 | Sim near-zero 1e-10 | Full-mask IoU 1e-10 | LCC IoU 1e-10 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 161 | 17 | 161 | 115 | 0.3869 | 0.5168 |
| 5 | 247 | 22 | 247 | 118 | 0.3670 | 0.6759 |
| 10 | 391 | 21 | 391 | 119 | 0.2687 | 0.0000 |
| 15 | 260 | 18 | 260 | 113 | 0.2032 | 0.0000 |
| 20 | 243 | 20 | 243 | 110 | 0.1206 | 0.1863 |

Kiểm tra precision bổ sung trên năm case Sim xác nhận `std` float32 cho **32/35/28/26/30** pixel zero-std theo AoA0/5/10/15/20, còn `std` float64 và phép kiểm tra constant trực tiếp cùng cho **17/22/21/18/20**. Có **7–15 pixel/case** bị float32 phân loại nhầm là constant, với vận tốc cực nhỏ cỡ1e-22. Đây là bằng chứng số học, không phải geometry. Xem [std_precision_check.csv](../research/hypothesis01_audit/std_precision_check.csv).

Số pixel từ `std==0` ở bản gốc không đồng nghĩa số exact-zero trong bảng mới. Xem cột `std_exact_zero_pixels` trong [per_file.csv](../research/hypothesis01_audit/per_file.csv). Kết luận cũ “Sim chỉ mask khoảng32 pixel” phụ thuộc định nghĩa đó, không phải một phép đo CAD geometry. Real pixel counts theo AoA cũng không tăng đơn điệu.

![Độ nhạy định nghĩa mask](../research/hypothesis01_audit/mask_threshold_sensitivity.png)

### Temporal stationarity

| Domain | File có zero-mask thay đổi | Tổng file | Pixel thay đổi/file, min–max |
| --- | --- | --- | --- |
| Real | 82 | 82 | 267–1303 |
| Sim | 0 | 100 | 0–0 |

Tập giao của zero-mask qua toàn bộ thời gian luôn là một mask tĩnh **do định nghĩa**. Điều đó không chứng minh mask từng frame bất biến. `instantaneous_zero_union_pixels`, `temporally_changing_zero_pixels` và số transition liền kề được đo riêng. Sự thay đổi này cũng không tự chứng minh vật thể vật lý chuyển động: nguyên nhân zero chưa được xác lập.

### Mask theo actual Re, giữ AoA cố định

Kiểm tra mọi cặp file trong từng domain/AoA, cho cả bốn định nghĩa mask. Bảng dưới là threshold `1e-10`, matching phần mở rộng notebook; bảng exact và các threshold khác nằm trong CSV.

| Domain | AoA | Số cặp | Min full-mask IoU | Max mismatch pixel | Min LCC IoU |
| --- | --- | --- | --- | --- | --- |
| Real | 0 | 136 | 0.2067 | 355 | 0.2933 |
| Real | 5 | 91 | 0.1780 | 346 | 0.0000 |
| Real | 10 | 153 | 0.4846 | 285 | 0.0000 |
| Real | 15 | 136 | 0.2592 | 303 | 0.0000 |
| Real | 20 | 120 | 0.3171 | 258 | 0.0000 |
| Sim | 0 | 190 | 1.0000 | 0 | 1.0000 |
| Sim | 5 | 190 | 1.0000 | 0 | 1.0000 |
| Sim | 10 | 190 | 1.0000 | 0 | 1.0000 |
| Sim | 15 | 190 | 1.0000 | 0 | 1.0000 |
| Sim | 20 | 190 | 1.0000 | 0 | 1.0000 |

**Kết quả cụ thể:** Real không bất biến theo file/Re ở bất kỳ AoA nào trong bảng. Sim bất biến hoàn toàn ở exact/1e-12/1e-10 trong tập đã đo; tại1e-8 xuất hiện sai khác nhỏ1–6 pixel giữa các file cùng AoA. Vì vậy ngay cả kết luận Sim cũng cần gắn với định nghĩa/ngưỡng mask.

Nếu IoU<1/mismatch>0 thì **zero-mask quan sát được** không bất biến hoàn toàn giữa các file ở AoA đó. Điều này không tách được tác động riêng của Re khỏi acquisition/preprocessing hay biến khác. Không chuyển nó thành luật nhân quả của Re hoặc kết luận cánh vật lý đổi hình dạng.

### Liên hệ history20 của sub7

Dùng các window rời nhau `[start,start+20)` history, `[start+20,start+40)` future, start=0,40,..., chỉ khi đủ40 frame. Tỷ lệ là pooled theo sample-pixel: pixel thuộc history mask nhưng có ít nhất một future frame/channel nonzero. Mask chỉ được dựng từ history. Một vị trí không gian ở các window khác nhau là các quan sát khác nhau.

| Domain | Stride | Số window | History-zero sample-pixels | Future-nonzero sample-pixels | Tỷ lệ activation |
| --- | --- | --- | --- | --- | --- |
| Real | 1 | 1693 | 1060043 | 99920 | 9.4260% |
| Real | 2 | 1693 | 269572 | 25045 | 9.2907% |
| Sim | 1 | 2500 | 49000 | 0 | 0.0000% |
| Sim | 2 | 2500 | 15000 | 0 | 0.0000% |

Phân tích này gồm toàn bộ82 Real, không phải tập80 window development của Phase1; các tỷ lệ không có cùng mẫu số. Stride1 là native64×128; stride2 là32×64 như inference. Đây là kiểm tra tính bền của observed zeros trên toàn dataset, **không** dùng để fit mask hay thay thế matched OFF/ON predictor evaluation. Dữ liệu Sim được báo riêng, không gộp vào Real.

## 3. Geometry: điều gì có và chưa có bằng chứng?

Không thấy top-level dataset có tên gợi explicit mask/geometry trong **182/182** file đã kiểm tra. Đây chỉ là kiểm tra tên/schema; không phủ nhận tài liệu vật lý bên ngoài hoặc metadata ở nơi khác. Re/AoA lấy trực tiếp từ datasets. Không có phép đối chiếu CAD/acquisition mask trong lượt này.

![Full mask và largest connected component](../research/hypothesis01_audit/mask_comparison.png)

Màu cam=Real only, xanh=Sim only, tím=intersection. Hàng trên là full near-zero mask; hàng dưới là 4-connected largest component. Plot dùng coordinate Real để hiển thị overlay theo chỉ số grid; sai khác coordinate Sim–Real đã định lượng ở phần1. Việc có một component liên thông/PCA orientation gần AoA không đủ định danh airfoil. AoA10/15 minh họa largest Real component lệch khỏi Sim component; chọn LCC làm geometry mặc định không được support.

## 4. Các kết luận notebook cần thay thế

| Nội dung cũ | Kết luận được phép sau audit |
|---|---|
| Grid strictly invariant ở1e-5 và PASS | FAIL tại1e-5 cho Sim–Real; PASS ở1e-4; báo sai khác/cell và spacing |
| `std=0` là solid/airfoil mask | Chỉ là thống kê temporal constancy dưới precision đang dùng; kiểm tra zero riêng |
| Mask tĩnh suốt607 frame ở mọi trajectory | Frame count thay đổi; persistent intersection tĩnh theo định nghĩa; zero-set từng frame có thể đổi |
| Mask chỉ đổi theo AoA và bất biến theo Re | Báo full/LCC pairwise IoU theo actual Re; không suy nhân quả từ grouping |
| Real mask lớn hơn do laser/shadow; Sim là CAD chính xác | Nguyên nhân và physical identity chưa được thiết lập |
| Mọi head nên nhận mask; mọi loss/scorer phải mask | Chưa được chứng minh bằng Hypo1; scorer hiện loại target-zero trong SPS/coverage, không toàn bộ point metrics |

**Quyết định:** ghi nhận grid gần đồng nhất, giữ tên `persistent_zero_mask`, không coi LCC là geometry mặc định và không thay loss/scorer/sub7. Nếu sau này muốn dùng hình học, cần đối chiếu nguồn geometry độc lập trước. Không mở thí nghiệm geometry hay gate mới trong lượt này.

## Bằng chứng và tái lập

Code/protocol/results: `research/hypothesis01_audit/`. Chạy từ repo root:

```powershell
python research/hypothesis01_audit/audit.py
python research/hypothesis01_audit/plot_results.py
python research/hypothesis01_audit/verify.py
python research/hypothesis01_audit/check_std_precision.py
```

File chính: `per_file.csv`, `fixed_AoA_pairs.csv`, `fixed_AoA_summary.csv`, `nominal10125_comparison.csv`, `history_future_windows.csv`, `history_future_summary.csv`, `masks.npz`, `archive_manifest.json`, `input_lock.json`, `completion.json`, `verification.json`. Mỗi ZIP member đã đọc qua kiểm tra CRC của ZipFile; manifest lưu path/size/CRC. Không gọi CRC là SHA-256 toàn archive. Notebook/source/protocol/sub7 có SHA-256 riêng. Kiểm tra số học về masks lồng nhau, bounds của tỷ lệ/IoU, sample IDs và bảo toàn artifact nằm trong verification.
