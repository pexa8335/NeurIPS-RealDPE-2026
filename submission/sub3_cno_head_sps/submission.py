"""RealPDE Track 1 Submission 3: CNO Backbone + Ensemble Residual Head + Calibrated SPS.

Architecture:
1. Point Predictor: Organizer Sim+Real CNO Point Predictor combined with a
   lightweight 55k-parameter Residual Head trained on physical TKE loss (lambda=0.1)
   across dual random seeds (seed 42 and seed 43).
2. Conditioning: Residual Head is conditioned on 20-frame observation history,
   CNO base forecast, and stationary damped fluctuation prior.
3. SPS Uncertainty: Calibrated confidence interval with Q=0.75 * RMS_RESIDUAL,
   achieving optimal 82-84% empirical coverage and minimizing pinball penalty.

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

# Normalization constants
_MEAN_IN = np.array([0.154960856, -0.000513992854, 0.0], dtype=np.float32)
_STD_IN = np.array([0.0968056545, 0.015960684, 1.0], dtype=np.float32)
_MEAN_TGT = np.array([0.154962569, -0.000517793698, 0.0], dtype=np.float32)
_STD_TGT = np.array([0.0968104079, 0.0159636438, 1.0], dtype=np.float32)

RMS_RESIDUAL = 0.010925
Q = 0.75  # Calibrated for optimal SPS score (82-84% coverage)

_BATCH = 16
_state = {"cno": None, "head42": None, "head43": None, "norm": None, "device": None}


class ResidualHead(nn.Module):
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

    def forward(self, x):
        return self.net(x).reshape(-1, 20, 2, 32, 64).permute(0, 1, 3, 4, 2)


def _pack(a):
    # a: (N, 20, 32, 64, 2) -> (N, 40, 32, 64)
    return a.transpose(0, 1, 4, 2, 3).reshape(len(a), 40, 32, 64)


def _get_models():
    if _state["cno"] is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cno, _ = load_baseline(_CNO_CKPT, device=device)
        cno = cno.to(device)
        cno.eval()

        bundle = torch.load(_HEAD_CKPT, map_location=device, weights_only=False)
        head42 = ResidualHead(120, 32, 40).to(device)
        head42.load_state_dict(bundle["state_dict_42"])
        head42.eval()

        head43 = ResidualHead(120, 32, 40).to(device)
        head43.load_state_dict(bundle["state_dict_43"])
        head43.eval()

        _state["cno"] = cno
        _state["head42"] = head42
        _state["head43"] = head43
        _state["norm"] = {
            "nm": torch.as_tensor(bundle["norm_mean"], dtype=torch.float32, device=device),
            "ns": torch.as_tensor(bundle["norm_std"], dtype=torch.float32, device=device),
            "rs": torch.as_tensor(bundle["residual_std"], dtype=torch.float32, device=device),
        }
        _state["device"] = device
    return _state["cno"], _state["head42"], _state["head43"], _state["norm"], _state["device"]


def predict(input_array, metadata=None):
    """Track 1 Prediction API.
    Args:
        input_array: numpy array of shape (N, T_in=20, H=32, W=64, C=3)
        metadata: optional dict
    Returns:
        dict with keys: 'prediction', 'lower', 'upper', all shape (N, T_out=20, H=32, W=64, C=3)
    """
    x = np.asarray(input_array, dtype=np.float32)
    N, T, H, W, C = x.shape
    cno, head42, head43, norm, device = _get_models()

    mean_in = torch.from_numpy(_MEAN_IN).to(device)
    std_in = torch.from_numpy(_STD_IN).to(device)
    mean_tgt = torch.from_numpy(_MEAN_TGT).to(device)
    std_tgt = torch.from_numpy(_STD_TGT).to(device)

    # 1. Base CNO Forward Pass
    cno_outs = []
    with torch.no_grad():
        for i in range(0, N, _BATCH):
            xb = torch.from_numpy(np.ascontiguousarray(x[i:i + _BATCH])).to(device)
            xb_norm = (xb - mean_in) / std_in
            yb = cno(xb_norm)
            yb = yb * std_tgt + mean_tgt
            cno_outs.append(yb.cpu().numpy())

    cno_base = np.concatenate(cno_outs, axis=0).astype(np.float32)

    # 2. Stationary Prior Construction
    uv_in = x[..., :2]
    mean_uv = uv_in.mean(axis=1) # (N, 32, 64, 2)
    prior_seq = []
    for h in range(1, 21):
        frame = mean_uv + (0.9 ** h) * (uv_in[:, -1] - mean_uv)
        prior_seq.append(frame)
    prior = np.stack(prior_seq, axis=1) # (N, 20, 32, 64, 2)

    # Solid mask from observation history
    solid = np.all(uv_in == 0.0, axis=(1, 4)) # (N, 32, 64)
    fluid = (~solid)[:, None, :, :, None].astype(np.float32) # (N, 1, 32, 64, 1)
    prior *= fluid

    # 3. Residual Head Ensemble Correction
    base_uv = cno_base[..., :2]
    nm = norm["nm"].cpu().numpy()
    ns = norm["ns"].cpu().numpy()
    rs = norm["rs"].cpu().numpy()

    feat_x = _pack((uv_in - nm) / ns)
    feat_base = _pack((base_uv - nm) / ns)
    feat_prior = _pack((prior - nm) / ns)
    feat = np.concatenate([feat_x, feat_base, feat_prior], axis=1) # (N, 120, 32, 64)

    head_outs = []
    with torch.no_grad():
        for i in range(0, N, _BATCH):
            fb = torch.from_numpy(np.ascontiguousarray(feat[i:i + _BATCH])).to(device)
            c42 = head42(fb).cpu().numpy()
            c43 = head43(fb).cpu().numpy()
            c_ens = 0.5 * (c42 + c43)
            head_outs.append(c_ens)

    corr = np.concatenate(head_outs, axis=0) # (N, 20, 32, 64, 2)
    final_uv = base_uv + corr * rs * fluid

    # Final 3-channel prediction
    prediction = np.zeros((N, 20, 32, 64, 3), dtype=np.float32)
    prediction[..., :2] = final_uv
    prediction[..., 2] = 0.0

    # Calibrated uncertainty bounds
    half = np.float32(Q * RMS_RESIDUAL)
    lower = prediction - half
    upper = prediction + half

    return {"prediction": prediction, "lower": lower, "upper": upper}
