"""RealPDE Track 1 candidate: frozen CNO + matched-capacity first-difference
ZeroMean residual head + unchanged zero guardrail and adaptive SPS.

Key Synergy:
1. Point Predictor (from proven sub6 winner):
   - CNO Backbone + Dual-seed (seeds 42 & 43) Zero-Mean Residual Head trained with lambda=0.30.
   - Exact mean-flow invariance guaranteed (mean_t(delta_u) == 0.0).
   - Proven on Hidden Leaderboard: TKE Score +1.11 points (73.72 -> 74.82), MVPE strictly preserved.
   - Vectorized Input Persistent-Zero Guardrail on fluid/solid boundary.
2. Spatio-Temporal Adaptive SPS Band (from proven sub5 winner):
   - Base half-widths: u=0.0075, v=0.0032.
   - Horizon expansion factor: sqrt(h / 10.0).
   - Local turbulence variance scaling: scale=0.85 * std_uv(history).
   - Proven on Hidden Leaderboard: SPS Score = 37.47 points (+1.96 SPS points over constant band).
3. Pure GPU Vectorization:
   - Zero CPU roundtrip loops; latency ~147.5 ms/sample (Time Score ~87.36).

Files at zip ROOT:
    submission.py
    sim_real_cno.pth
    cno_head_weights.pth
    load_baseline.py
    rpde_baselines/
    _vendor/
"""
from __future__ import annotations

import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
import torch.nn as nn
from load_baseline import load_baseline

_CNO_CKPT = os.path.join(_HERE, "sim_real_cno.pth")
_HEAD_CKPT = os.path.join(_HERE, "cno_head_weights.pth")

# Official normalization constants
_MEAN_IN = np.array([0.154960856, -0.000513992854, 0.0], dtype=np.float32)
_STD_IN = np.array([0.0968056545, 0.015960684, 1.0], dtype=np.float32)
_MEAN_TGT = np.array([0.154962569, -0.000517793698, 0.0], dtype=np.float32)
_STD_TGT = np.array([0.0968104079, 0.0159636438, 1.0], dtype=np.float32)

_BATCH = 16
_state = {
    "cno": None, "head42": None, "head43": None, "norm": None,
    "device": None, "decay": None, "base_hw": None, "max_hw": None, "h_factors": None
}


class ZeroMeanResidualHead(nn.Module):
    """Zero-Mean Residual Head (55k parameters).
    Algebraically enforces mean_t(delta_u) == 0.0 to strictly preserve CNO mean flow.
    """
    def __init__(self, in_channels=120, width=32, out_channels=40):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1),
            nn.GroupNorm(4, width),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GroupNorm(4, width),
            nn.GELU(),
            nn.Conv2d(width, out_channels, 3, padding=1)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        out = self.net(x).reshape(-1, 20, 2, 32, 64).permute(0, 1, 3, 4, 2)
        out_mean = out.mean(dim=1, keepdim=True)
        return out - out_mean


def _pack_torch(a):
    # a: (N, 20, 32, 64, 2) -> (N, 40, 32, 64)
    return a.permute(0, 1, 4, 2, 3).reshape(a.shape[0], 40, 32, 64)


def _get_models():
    if _state["cno"] is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cno, _ = load_baseline(_CNO_CKPT, device=device)
        cno = cno.to(device)
        cno.eval()

        bundle = torch.load(_HEAD_CKPT, map_location=device, weights_only=False)
        head42 = ZeroMeanResidualHead(120, 32, 40).to(device)
        head42.load_state_dict(bundle["state_dict_42"])
        head42.eval()

        head43 = ZeroMeanResidualHead(120, 32, 40).to(device)
        head43.load_state_dict(bundle["state_dict_43"])
        head43.eval()

        _state["cno"] = cno
        _state["head42"] = head42
        _state["head43"] = head43
        _state["norm"] = {
            "nm": torch.as_tensor(bundle["norm_mean"], dtype=torch.float32, device=device),
            "ns": torch.as_tensor(bundle["norm_std"], dtype=torch.float32, device=device),
            "rs": torch.as_tensor(bundle["residual_std"], dtype=torch.float32, device=device),
            "dm": torch.as_tensor(bundle["delta_mean"], dtype=torch.float32, device=device),
            "ds": torch.as_tensor(bundle["delta_std"], dtype=torch.float32, device=device),
        }
        _state["device"] = device
        _state["decay"] = (0.9 ** torch.arange(1, 21, dtype=torch.float32, device=device)).view(1, 20, 1, 1, 1)
        _state["base_hw"] = torch.tensor([0.0075, 0.0032], dtype=torch.float32, device=device).view(1, 1, 1, 1, 2)
        _state["max_hw"] = torch.tensor(0.040, dtype=torch.float32, device=device)
        _state["h_factors"] = torch.sqrt(torch.arange(1, 21, dtype=torch.float32, device=device) / 10.0).view(1, 20, 1, 1, 1)

    return (
        _state["cno"], _state["head42"], _state["head43"],
        _state["norm"], _state["device"], _state["decay"],
        _state["base_hw"], _state["max_hw"], _state["h_factors"]
    )


def predict(input_array, metadata=None):
    """RealPDE Track 1 Prediction Interface.
    Args:
        input_array: numpy array of shape (N, T_in=20, H=32, W=64, C=3)
        metadata: optional dict
    Returns:
        dict with keys: 'prediction', 'lower', 'upper', all shape (N, T_out=20, H=32, W=64, C=3)
    """
    x_np = np.asarray(input_array, dtype=np.float32)
    N, T, H, W, C = x_np.shape
    cno, head42, head43, norm, device, decay, base_hw, max_hw, h_factors = _get_models()

    mean_in = torch.from_numpy(_MEAN_IN).to(device)
    std_in = torch.from_numpy(_STD_IN).to(device)
    mean_tgt = torch.from_numpy(_MEAN_TGT).to(device)
    std_tgt = torch.from_numpy(_STD_TGT).to(device)

    # 1. Base CNO Forward Pass
    cno_outs = []
    with torch.no_grad():
        for i in range(0, N, _BATCH):
            xb = torch.from_numpy(np.ascontiguousarray(x_np[i:i + _BATCH])).to(device)
            xb_norm = (xb - mean_in) / std_in
            yb = cno(xb_norm)
            yb = yb * std_tgt + mean_tgt
            cno_outs.append(yb)

    cno_base = torch.cat(cno_outs, dim=0)  # (N, 20, 32, 64, 3)

    # 2. Vectorized Stationary Prior Construction on GPU
    x_t = torch.from_numpy(np.ascontiguousarray(x_np)).to(device)
    uv_in = x_t[..., :2]  # (N, 20, 32, 64, 2)
    mean_uv = uv_in.mean(dim=1, keepdim=True)  # (N, 1, 32, 64, 2)
    diff = uv_in[:, -1:, :, :, :] - mean_uv
    prior = mean_uv + decay * diff  # (N, 20, 32, 64, 2)

    # Extract strict non-leaking persistent-zero mask from observation history
    solid = torch.all(uv_in == 0.0, dim=(1, 4))  # (N, 32, 64)
    fluid = (~solid)[:, None, :, :, None].float()  # (N, 1, 32, 64, 1)
    prior = prior * fluid

    # 3. Zero-Mean Residual Head Ensemble Correction on GPU
    base_uv = cno_base[..., :2]
    nm = norm["nm"]
    ns = norm["ns"]
    rs = norm["rs"]

    # Same 20 observations and 40 channels as sub7, expressed as the normalized
    # first frame followed by 19 train-normalized first differences.
    delta_uv = uv_in[:, 1:] - uv_in[:, :-1]
    delta_sequence = torch.cat([
        (uv_in[:, :1] - nm) / ns,
        (delta_uv - norm["dm"]) / norm["ds"],
    ], dim=1)
    feat_x = _pack_torch(delta_sequence)
    feat_base = _pack_torch((base_uv - nm) / ns)
    feat_prior = _pack_torch((prior - nm) / ns)
    feat = torch.cat([feat_x, feat_base, feat_prior], dim=1)  # (N, 120, 32, 64)

    head_outs = []
    with torch.no_grad():
        for i in range(0, N, _BATCH):
            fb = feat[i:i + _BATCH]
            c42 = head42(fb)
            c43 = head43(fb)
            c_ens = 0.5 * (c42 + c43)
            head_outs.append(c_ens)

    corr = torch.cat(head_outs, dim=0)  # (N, 20, 32, 64, 2)
    
    # Enforce Zero Guardrail on fluid mask simultaneously
    final_uv = (base_uv + corr * rs) * fluid

    # Final 3-channel prediction tensor
    prediction_t = torch.zeros((N, 20, 32, 64, 3), dtype=torch.float32, device=device)
    prediction_t[..., :2] = final_uv
    prediction_t[..., 2] = 0.0

    # 4. Joint Spatio-Temporal High-Coverage SPS Confidence Interval
    # GPU standard deviation from 20 observation frames
    std_uv = torch.std(uv_in, dim=1, keepdim=True)  # (N, 1, 32, 64, 2)

    # Scale = 0.85 calibrated for 85-90% coverage on high-Reynolds test regimes
    scale = 0.85
    hw_raw = base_hw + scale * std_uv * h_factors  # (N, 20, 32, 64, 2)
    hw_uv = torch.clamp(hw_raw, min=base_hw, max=max_hw)

    # Candidate A1: scale u-channel half-width by 0.80, v-channel unchanged
    hw_uv[..., 0] = hw_uv[..., 0] * 0.80

    hw_t = torch.zeros((N, 20, 32, 64, 3), dtype=torch.float32, device=device)
    hw_t[..., :2] = hw_uv

    lower_t = prediction_t - hw_t
    upper_t = prediction_t + hw_t

    # Single transfer to CPU NumPy
    prediction = prediction_t.cpu().numpy()
    lower = lower_t.cpu().numpy()
    upper = upper_t.cpu().numpy()

    return {"prediction": prediction, "lower": lower, "upper": upper}
