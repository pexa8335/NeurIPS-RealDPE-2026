Dưới đây là **bản hoàn chỉnh, gọn nhưng đủ technical detail**, dựa trực tiếp trên `scoring.py`, `submission_template.py` và README của **Starting Kit v9**.

# Terminology

## Dataset / Evaluation Geometry

RealPDE Track 1 Sim2Real là bài toán:

$$
\boxed{20\text{ past frames}\rightarrow20\text{ future frames}}
$$

Evaluator input:

$$
\boxed{(N,T_{in},H,W,C)=(N,20,32,64,3)}
$$

với channel order:

$$
\boxed{[u,v,p]}
$$

Trong đó:

* `u`: velocity component theo x-direction.
* `v`: velocity component theo y-direction.
* `p`: pressure.
* Real PIV không đo pressure nên `p = 0`.
* Raw real PIV data có resolution `64×128`, nhưng evaluation downsample ×2 và score tại `32×64`. 

Output:

$$
\boxed{(N,T_{out},H,W,C)=(N,20,32,64,3)}
$$

---

# 5 Evaluation Metrics

Có 5 subscores, mỗi score nằm trong khoảng `0–100`:

1. Rel-L2 Score
2. TKE Score
3. MVPE Score
4. Time Score
5. SPS Score

Leaderboard combine 5 scores này thành:

$$
\boxed{FinalScore=f(RelL2,TKE,MVPE,Time,SPS)}
$$

nhưng **công thức combine final score không được publish trong Starting Kit v9**. 

---

## 1. Rel-L2 Score — Field Prediction Accuracy

Rel-L2 đo predicted `u,v` có gần ground-truth `u,v` trên toàn bộ future spatiotemporal field hay không.

**Rel-L2 không phải MSE.**

Scorer:

```python id="l5p2h1"
p = pred[..., :c].reshape(pred.shape[0], -1)
t = target[..., :c].reshape(target.shape[0], -1)

denom = np.linalg.norm(t, axis=1)
error = np.linalg.norm(p - t, axis=1) / denom
```

Với mỗi sample:

$$
\boxed{
e_{\text{RelL2}}
=
\frac{\|\hat X-X\|_2}
{\|X\|_2}
}
$$

Tức scorer flatten toàn bộ:

$$
20\times32\times64\times2
$$

giá trị `u,v` thành một vector rồi so prediction với GT.

Sau đó average error qua samples.

Error được convert thành score:

$$
\boxed{
S_{\text{RelL2}}
=
\frac{100}
{1+0.5e_{\text{RelL2}}}
}
$$

Error càng nhỏ → score càng gần 100. 

### Intuition

Rel-L2 hỏi:

> Model có predict đúng `u,v` tại đúng spatial location và đúng future timestep không?

Vì vậy Rel-L2 rất sensitive với:

* magnitude error;
* spatial error;
* temporal/phase error.

Nếu vortex đúng shape nhưng xuất hiện sai thời điểm hoặc sai vị trí → Rel-L2 vẫn bị phạt.

---

# 2. TKE Score — Temporal Fluctuation Energy

TKE = **Turbulent Kinetic Energy**.

Fluctuation = dao động quanh giá trị trung bình (mean score).

Tại 1 (x, y) cố định, velocity u thay đổi theo thời gian.

u(t)=[0.04, 0.06, 0.05, 0.07, 0.03]

Giá trị trung bình theo thời gian:

u_mean=0.05

nên u' = [-0.01, +0.01, 0, +0.02, −0.02]

=> Velocity = mean flow + fluctuation



Metric này đo model có reproduce đúng mức **temporal fluctuation** của velocity hay không.

Tại mỗi spatial location `(i,j)`, trong 20 future frames:

$$
u_1(i,j),u_2(i,j),...,u_{20}(i,j)
$$

tính temporal mean:

$$
\bar u(i,j)
=
\frac1{20}\sum_tu_t(i,j)
$$

Fluctuation:

$$
\boxed{
u'_t(i,j)
=
u_t(i,j)-\bar u(i,j)
}
$$

Tương tự:

$$
v'_t(i,j)
=
v_t(i,j)-\bar v(i,j)
$$

Scorer thực hiện:

```python id="ftd0dw"
u_prime = np.mean(
    (u - np.mean(u, axis=1, keepdims=True)) ** 2,
    axis=1
)
```

Ở đây `axis=1` là time dimension.

Sau đó:

$$
\boxed{
TKE(i,j)
=
\frac12
\left[
\overline{u'^2}(i,j)
+
\overline{v'^2}(i,j)
\right]
}
$$

Scorer tạo:

$$
TKE_{\text{pred}}(i,j)
$$

và:

$$
TKE_{\text{GT}}(i,j)
$$

rồi so hai TKE maps bằng Relative L2.

Cuối cùng:

$$
\boxed{
S_{\text{TKE}}
=
\frac{100}
{1+0.5e_{\text{TKE}}}
}
$$



### Intuition

Giả sử GT:

```text id="qnnqge"
0.04 → 0.06 → 0.04 → 0.06 → ...
```

mean ≈ `0.05`, nhưng có oscillation.

Model:

```text id="r9idpg"
0.05 → 0.05 → 0.05 → 0.05 → ...
```

Mean cũng đúng:

$$
\bar u=0.05
$$

nhưng model không fluctuation:

$$
u'=0
$$

→ TKE prediction sai.

### Important

TKE không quan tâm chính xác temporal phase theo cùng cách Rel-L2.

Nó chủ yếu hỏi:

> Trong 20 future frames, tại mỗi spatial location, model có tạo ra đúng **mức temporal fluctuation energy** không?

Vì vậy model quá smooth có thể Rel-L2 tương đối tốt nhưng TKE score thấp.

---

# 3. MVPE Score — Mean Velocity Profile Error

MVPE đo **time-averaged velocity profile tại một số hard-coded probe locations**.

Nó không treat mọi pixel giống nhau.

V9 code:

```python id="a08q6s"
d = 16
center_x = 10
center_y = 32
n_probe = 9
sub_s_real = 2
```

Ở evaluation resolution `32×64`:

$$
probe_y=
[8,10,12,14,16,18,20,22,24]
$$

và:

$$
probe_x=
[13,21,29,37]
$$

Do đó có:

$$
\boxed{4\times9=36}
$$

spatial probe locations.

Conceptually:

```text id="m6qf86"
                  j / x

          13      21      29      37
           |       |       |       |

i = 8      •       •       •       •
i = 10     •       •       •       •
i = 12     •       •       •       •
i = 14     •       •       •       •
i = 16     •       •       •       •
i = 18     •       •       •       •
i = 20     •       •       •       •
i = 22     •       •       •       •
i = 24     •       •       •       •
```

Tại mỗi probe x, scorer lấy:

```python id="apx3mc"
pred[:, :, probe_y, probe_x, :2].mean(axis=1)
```

Tức lấy:

* `u,v`;
* tại 9 y positions;
* rồi average qua 20 future timesteps.

Sau đó predicted mean profile được so với GT mean profile bằng Relative L2.

Score:

$$
\boxed{
S_{\text{MVPE}}
=
\frac{100}
{1+0.5e_{\text{MVPE}}}
}
$$



### Intuition

MVPE hỏi:

> Average flow profile model tạo ra tại những probe locations quan trọng có giống real flow hay không?

### Modeling implication

MVPE không quan tâm mọi spatial location như nhau.

Do probe locations được hard-code, ta có thể:

* plot chính xác 36 probes lên flow field;
* xác định chúng nằm ở wake/airfoil/freestream nào;
* monitor validation error riêng tại các probes;
* potentially dùng probe-aware loss nếu nó cải thiện leaderboard mà không phá các metrics khác.

---

# 4. Time Score — Inference Runtime

Time Score đo model inference nhanh đến đâu so với numerical solver.

Reference numerical solver runtime:

$$
\boxed{
T_{\text{numerical}}=0.72896\text{ s}
}
$$

Tính:

$$
\boxed{
r=
\frac{T_{\text{neural}}}
{T_{\text{numerical}}}
}
$$

Sau đó:

$$
\boxed{
S_{\text{Time}}
=
\frac{100}
{1+\sqrt r}
}
$$



### Example

Nếu:

$$
T_{\text{neural}}=T_{\text{numerical}}
$$

thì:

$$
r=1
$$

và:

$$
S_{\text{Time}}=50
$$

Nếu neural model nhanh hơn numerical solver 100×:

$$
r=0.01
$$

thì:

$$
S_{\text{Time}}
=
\frac{100}{1+0.1}
\approx90.91
$$

### Implication

Ensemble nhiều giant models có thể:

$$
Accuracy\uparrow
$$

nhưng:

$$
Runtime\uparrow
\Rightarrow TimeScore\downarrow
$$

nên phải optimize trade-off giữa accuracy và compute.

---

# 5. SPS Score — Safe Prediction Score

SPS = **Safe Prediction Score**.

Nó đo cả:

1. prediction accuracy;
2. uncertainty calibration.

Thay vì chỉ output point prediction:

$$
\hat y=0.050
$$

model có thể output uncertainty interval:

$$
[lower,upper]
$$

Ví dụ:

$$
\hat y=0.050
$$

$$
[lower,upper]=[0.045,0.055]
$$

GT:

$$
y=0.052
$$

nằm trong interval → được score.

---

## SPS: Default Bounds

Nếu submission không cung cấp `lower/upper`, scorer tự tạo:

$$
\boxed{
lower
=
pred-0.05|pred|
}
$$

$$
\boxed{
upper
=
pred+0.05|pred|
}
$$

Tức total interval width:

$$
0.1|pred|
$$



Ví dụ:

$$
pred=0.06
$$

thì:

$$
0.05|pred|=0.003
$$

nên:

$$
[lower,upper]
=
[0.057,0.063]
$$

---

## SPS: Coverage

Scorer kiểm tra:

$$
\boxed{
lower\le GT\le upper
}
$$

Nếu GT nằm ngoài interval → element đó contribution = `0`.

Nếu interval cực rộng thì dễ cover GT, nhưng bị width penalty.

---

## SPS: Interval Width Penalty

Scorer tính:

$$
NIL=
\frac{upper-lower}{\sigma}
$$

với:

$$
\boxed{
\sigma=0.0563870259
}
$$

Sau đó dùng:

$$
\boxed{
e^{-NIL}
}
$$

Interval càng rộng:

$$
NIL\uparrow
\Rightarrow
e^{-NIL}\downarrow
$$

→ SPS giảm.

Vì vậy không thể cheat bằng interval cực rộng. 

---

## SPS: Accuracy Component

SPS không chỉ nhìn uncertainty interval.

Nó còn lấy ba errors:

$$
e_{\text{RelL2}},
e_{\text{TKE}},
e_{\text{MVPE}}
$$

và normalize:

$$
\boxed{
p_m=
\frac{e_m}
{0.5+e_m}
}
$$

Accuracy factor:

$$
\boxed{1-p_m}
$$

Error càng nhỏ → accuracy factor càng gần 1.

Mỗi SPS branch conceptually thưởng:

$$
\text{accuracy}
\times
\text{tight interval}
\times
\text{correct coverage}
$$

Cuối cùng:

$$
\boxed{
SPS_{\text{raw}}
=
0.5SPS_{\text{RelL2}}
+
0.3SPS_{\text{TKE}}
+
0.2SPS_{\text{MVPE}}
}
$$

Tức SPS gián tiếp weight:

* `50%` Rel-L2 accuracy;
* `30%` TKE accuracy;
* `20%` MVPE accuracy;

cộng thêm uncertainty calibration. 

V9 convert thành score tuyến tính:

$$
\boxed{
SPS_{\text{score}}
=
100\times SPS_{\text{raw}}
}
$$

với value clipped vào `[0,1]`. 

---

## SPS Mask

Trong SPS:

```python id="18ygne"
scored = target != 0.0
```

Theo comment của scorer, targets:

* outside PIV field of view;
* inside airfoil body;

không được tính vào SPS interval average/coverage.

Tuy nhiên accuracy factors Rel-L2/TKE/MVPE vẫn được tính trên whole measured field theo implementation của scorer. 

---

# Final Score

Sau khi evaluation:

```text id="em6z9p"
Rel-L2 Score ─┐
TKE Score ────┤
MVPE Score ───┤
Time Score ───┼──→ Secret combination → Final Score
SPS Score ────┘
```

Starting Kit v9 **không publish công thức combine**.

Local `scoring.py` chỉ trả:

```python id="sdqd8z"
{
    "rel_l2_score": ...,
    "tke_score": ...,
    "mvpe_score": ...,
    "time_score": ...,
    "sps_score": ...
}
```

không có formula cho `final_score`. 

---

# Submit What?

Không submit hidden-test predictions trực tiếp.

Submit một **ZIP chứa model/code**, trong đó entry point là:

```text id="xyuwcv"
submission.py
```

`submission.py` phải có:

```python id="crl4ke"
def predict(input_array, metadata=None):
```



---

## Input

Evaluator gọi:

```python id="6h6mzt"
predict(input_array, metadata={})
```

Trong scored calls:

```text id="vea21i"
metadata = {}
```

Input shape:

$$
\boxed{
(N,20,32,64,3)
}
$$

với:

```text id="jwm25a"
channel 0 = u
channel 1 = v
channel 2 = p
```

Real PIV:

$$
p=0
$$



---

# Output

Model phải predict next 20 frames:

$$
\boxed{
prediction.shape=(N,20,32,64,3)
}
$$

Normal submission:

```python id="e3s0rp"
return prediction
```

---

## Output With SPS Bounds

Nếu muốn tự optimize uncertainty:

```python id="7t4yr1"
return {
    "prediction": prediction,
    "lower": lower,
    "upper": upper
}
```

Trong đó:

$$
prediction.shape
=
lower.shape
=
upper.shape
$$

và phải:

$$
\boxed{lower\le upper}
$$

tại mọi element.

Tất cả arrays phải finite.

Nếu bounds invalid, scorer có thể trả zero cho toàn bộ subscores. 

Nếu không có uncertainty model tốt, có thể chỉ:

```python id="g4i4fl"
return prediction
```

và scorer tự dùng default ±5% interval. 

---

# Example Submission ZIP

Nếu dùng official FNO baseline:

```text id="1zw44i"
submission.zip
│
├── submission.py
├── sim_real_fno_fp16.pth
├── load_baseline.py
├── rpde_baselines/
└── _vendor/
```

Các file phải nằm ở **root của ZIP** như trên. 

Nếu dùng model tự viết:

```text id="oflkhx"
submission.zip
│
├── submission.py
├── model.py
├── checkpoint.pth
└── required_dependencies/
```

Submission size cap:

$$
\boxed{256\text{ MB}}
$$

Official models:

* CNO ≈ 32 MB
* Transolver ≈ 50 MB
* FNO fp32 ≈ 403 MB → quá lớn
* FNO fp16-packed ≈ 201 MB → fit submission cap. 

---

# Important Submission Constraints

### Evaluation resolution

Raw PIV:

$$
64\times128
$$

Scored resolution:

$$
\boxed{32\times64}
$$

Do not assume evaluator sends `64×128`. 

### Pressure

Real PIV không measured pressure:

$$
\boxed{p=0}
$$

Official example explicitly forces:

```python id="mhfnd5"
prediction[..., 2] = 0.0
```



### Metadata

Do not depend on metadata:

```python id="tczag2"
metadata = {}
```

on scored calls. 

### Dependencies

Evaluation image có PyTorch + NumPy nhưng **không có**:

```text id="1d1tnp"
scipy
h5py
einops
matplotlib
pandas
```

Nếu model cần package ngoài environment, phải vendor dependency phù hợp trong submission. 

### Normalization

Official FNO submission normalize raw input:

$$
x_{norm}
=
\frac{x-\mu_{in}}
{\sigma_{in}}
$$

chạy model trong normalized space, rồi denormalize:

$$
y
=
y_{norm}\sigma_{target}
+
\mu_{target}
$$

Official example cảnh báo skipping normalization là một nguyên nhân phổ biến khiến checkpoint load đúng nhưng score rất thấp. 

---

# Metric Mental Model

```text id="4hr7y2"
                     MODEL
                       │
              predict next 20 frames
                       │
          ┌────────────┼────────────┐
          │            │            │
          ▼            ▼            ▼
      Exact field   Fluctuation   Mean flow
       matching       energy       profiles
          │            │            │
          ▼            ▼            ▼
       Rel-L2         TKE          MVPE

                 + Runtime
                     │
                     ▼
                    Time

           + Uncertainty bounds
                     │
                     ▼
                    SPS

                     │
                     ▼
             Five 0–100 scores
                     │
                     ▼
           SECRET FINAL FORMULA
                     │
                     ▼
                Final Score
```

### Short interpretation

* **Rel-L2:** *Did I predict the correct future `u,v` field?*
* **TKE:** *Did I reproduce the correct amount of temporal fluctuation?*
* **MVPE:** *Did I reproduce the correct time-averaged velocity profiles at the predefined probes?*
* **Time:** *How fast is my model?*
* **SPS:** *Is my prediction accurate, and does my uncertainty interval reliably cover the truth without being unnecessarily wide?*
* **Final Score:** secret combination of the five subscores.
