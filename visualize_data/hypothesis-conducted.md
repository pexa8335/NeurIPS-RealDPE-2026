PHẢN BÁC HOÀN TOÀN $H_2$: Dòng chảy khí động quanh cánh NACA4418 là dòng chảy hỗn loạn/rối (chaotic). Các xoáy tức thời không bao giờ đồng pha giữa mô phỏng và thực tế. Không thể dùng hàm loss ép trực tiếp frame-to-frame giữa Sim và Real.

NẢY SINH NGHỊCH LÝ MỚI: Tại sao hình thái xoáy không gian (RMS) và trường trung bình tương quan rất cao ($\sim 0.9$) mà relL2 lại bùng nổ hàng nghìn phần trăm? $\rightarrow$ Nghi vấn: Có sự sai lệch lớn về thang đo biên độ (Scale / Unit mismatch).

