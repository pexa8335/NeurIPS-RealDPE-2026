"""RealPDE Track 1 Sim2Real submission: FNO baseline + calibrated SPS intervals.

Wraps the organizers' sim_real_fine-tuned FNO checkpoint (fp16-packed) behind
the Track 1 predict() API, and returns per-element lower/upper bounds for
sps_score.

Bounds: prediction +/- Q * RMS_RESIDUAL, a constant half-width calibrated on
train_real windows with the official scorer (see runs/sps/fno_const.npz).
The constant band beats the scorer's default +/-5%*|pred| band, which collapses
to zero width where the prediction is near zero. Q is set ~1.3x above the
in-sample optimum because the hidden evaluation windows are out-of-sample for
this checkpoint (larger, heavier-tailed residuals).

Archive layout (all at the zip ROOT):
    submission.py            <- this file
    sim_real_fno_fp16.pth    <- checkpoint (fp16-packed, 193 MB)
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

import torch  # noqa: E402

from load_baseline import load_baseline  # noqa: E402  (prepends _vendor/ for einops)

_CKPT_PATH = os.path.join(_HERE, "sim_real_fno_fp16.pth")

# GaussianNormalizer statistics fitted on train_real 32x64 windows, channels
# [u, v, p] (identical to the organizers' FNO example submission).
_MEAN_IN = np.array([0.154960856, -0.000513992854, 0.0], dtype=np.float32)
_STD_IN = np.array([0.0968056545, 0.015960684, 1.0], dtype=np.float32)
_MEAN_TGT = np.array([0.154962569, -0.000517793698, 0.0], dtype=np.float32)
_STD_TGT = np.array([0.0968104079, 0.0159636438, 1.0], dtype=np.float32)

# Calibrated interval: half-width = Q * RMS_RESIDUAL (see module docstring).
RMS_RESIDUAL = 0.008530
Q = 1.15

_BATCH = 16

_state = {"model": None, "device": None}


def _get_model():
    if _state["model"] is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _meta = load_baseline(_CKPT_PATH, device=device)
        model = model.to(device)
        model.eval()
        _state["model"] = model
        _state["device"] = device
    return _state["model"], _state["device"]


def predict(input_array, metadata=None):
    """Track 1 API: (N, T_in, H, W, C) raw fields -> dict with prediction and
    lower/upper bounds, all (N, T_out, H, W, C)."""
    x = np.asarray(input_array, dtype=np.float32)
    model, device = _get_model()

    mean_in = torch.from_numpy(_MEAN_IN).to(device)
    std_in = torch.from_numpy(_STD_IN).to(device)
    mean_tgt = torch.from_numpy(_MEAN_TGT).to(device)
    std_tgt = torch.from_numpy(_STD_TGT).to(device)

    outs = []
    with torch.no_grad():
        for i in range(0, x.shape[0], _BATCH):
            xb = torch.from_numpy(np.ascontiguousarray(x[i:i + _BATCH])).to(device)
            xb = (xb - mean_in) / std_in
            yb = model(xb)
            yb = yb * std_tgt + mean_tgt
            outs.append(yb.float().cpu().numpy())

    prediction = np.concatenate(outs, axis=0).astype(np.float32)
    prediction[..., 2] = 0.0  # p is unmeasured in real data

    half = np.float32(Q * RMS_RESIDUAL)
    lower = prediction - half
    upper = prediction + half
    return {"prediction": prediction, "lower": lower, "upper": upper}
