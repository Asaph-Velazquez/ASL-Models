"""capa 1: normalizacion (camino cuerpo | mano) + interpolacion + movimiento"""
import json
from pathlib import Path

import numpy as np


def load_layout(cache_dir):
    return json.loads((Path(cache_dir) / "layout.json").read_text())


def _block(arr, layout, name):
    a, b = layout["offsets"][name]
    return arr[:, a:b]


def interpolate_missing(arr, layout):
    """interpola frames sin deteccion por bloque"""
    arr = arr.copy()
    T = arr.shape[0]
    interp_frames = 0
    for i, name in enumerate(("lh", "rh", "pose")):
        a, b = layout["offsets"][name]
        m_a = layout["offsets"]["mask"][0] + i
        mask = arr[:, m_a] > 0.5
        if mask.all() or not mask.any():
            continue
        idx = np.arange(T)
        good = idx[mask]
        for c in range(a, b):
            arr[:, c] = np.interp(idx, good, arr[good, c])
        interp_frames += int((~mask).sum())
    pct = 100.0 * interp_frames / max(1, 3 * T)
    return arr, pct


def normalize_body(arr, layout, eps=1e-6):
    """camino cuerpo: centra en hombros y escala por su ancho"""
    arr = arr.copy()
    pose_idx = layout["pose_upper_indices"]
    a, _ = layout["offsets"]["pose"]
    try:
        i_l, i_r = pose_idx.index(11), pose_idx.index(12)
    except ValueError:  # sin hombros: centra por media del pose
        pose = _block(arr, layout, "pose").reshape(len(arr), -1, 3)
        center = pose.mean(axis=1, keepdims=False)
        scale = np.ones(len(arr))
        return _apply(arr, layout, center, scale, eps)
    l_sh = arr[:, a + 3 * i_l: a + 3 * i_l + 3]
    r_sh = arr[:, a + 3 * i_r: a + 3 * i_r + 3]
    center = (l_sh + r_sh) / 2.0
    scale = np.linalg.norm((l_sh - r_sh)[:, :2], axis=1)
    scale = np.where(scale < eps, np.median(scale[scale >= eps]) if (scale >= eps).any() else 1.0,
                     scale)
    return _apply(arr, layout, center, scale, eps)


def normalize_hand(arr, layout, eps=1e-6):
    """camino mano: centra en la muneca dominante y escala por el tamano de la mano"""
    arr = arr.copy()
    m_a = layout["offsets"]["mask"][0]
    lh_rate = arr[:, m_a].mean()
    rh_rate = arr[:, m_a + 1].mean()
    hand = "rh" if rh_rate >= lh_rate else "lh"
    a, _ = layout["offsets"][hand]
    wrist = arr[:, a: a + 3]
    mcp = arr[:, a + 3 * 9: a + 3 * 9 + 3]
    center = wrist
    scale = np.linalg.norm((mcp - wrist)[:, :2], axis=1)
    med = np.median(scale[scale >= eps]) if (scale >= eps).any() else 1.0
    scale = np.where(scale < eps, med, scale)
    return _apply(arr, layout, center, scale, eps)


def _apply(arr, layout, center, scale, eps):
    scale = np.maximum(scale, eps)[:, None]
    for name in ("lh", "rh", "pose"):
        a, b = layout["offsets"][name]
        block = arr[:, a:b].reshape(len(arr), -1, 3)
        block[:, :, 0] -= center[:, 0:1]
        block[:, :, 1] -= center[:, 1:2]
        block[:, :, 2] -= center[:, 2:3]
        block /= scale[:, :, None]
        arr[:, a:b] = block.reshape(len(arr), -1)
    return arr


def add_motion(pos):
    """pos+vel+acel, misma representacion que el generador sintetico"""
    vel = np.zeros_like(pos)
    vel[1:] = np.diff(pos, axis=0)
    acc = np.zeros_like(vel)
    acc[1:] = np.diff(vel, axis=0)
    return np.concatenate([pos, vel, acc], axis=1)


def normalize_clip(arr, layout, path, interpolate=True):
    """capa 1 completa para un clip [T, D]; path 'body' | 'hand'"""
    info = {"camino": path, "pct_interpolado": 0.0}
    if interpolate:
        arr, info["pct_interpolado"] = interpolate_missing(arr, layout)
    arr = normalize_body(arr, layout) if path == "body" else normalize_hand(arr, layout)
    feats = add_motion(arr.astype(np.float32))
    return feats, info


def path_for_class(functional_class, hand_relative_classes):
    return "hand" if functional_class in hand_relative_classes else "body"
