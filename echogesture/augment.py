"""Augmentations applied to raw landmarks (before feature computation)."""
from __future__ import annotations

import numpy as np

# MediaPipe pose left/right swap table
_POSE_SWAP = np.arange(33)
for a, b in [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10)] + [(i, i + 1) for i in range(11, 32, 2)]:
    _POSE_SWAP[a], _POSE_SWAP[b] = b, a


def _copy(clip: dict) -> dict:
    return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in clip.items()}


def mirror(clip: dict) -> dict:
    c = _copy(clip)
    c["hands_img"][..., 0] = 1.0 - c["hands_img"][..., 0]
    c["hands_world"][..., 0] *= -1.0
    c["hands_img"] = c["hands_img"][:, ::-1].copy()
    c["hands_world"] = c["hands_world"][:, ::-1].copy()
    c["hand_present"] = c["hand_present"][:, ::-1].copy()
    c["pose"][..., 0] = 1.0 - c["pose"][..., 0]
    c["pose"] = c["pose"][:, _POSE_SWAP].copy()
    return c


def _rot2d(xy, ang, aspect):
    cx, cy = 0.5 * aspect, 0.5
    x = xy[..., 0] * aspect - cx
    y = xy[..., 1] - cy
    co, si = np.cos(ang), np.sin(ang)
    xy[..., 0] = (co * x - si * y + cx) / aspect
    xy[..., 1] = si * x + co * y + cy


def rotate(clip: dict, roll: float = 0.0, yaw: float = 0.0, pitch: float = 0.0) -> dict:
    """Camera roll (in-plane, radians) plus out-of-plane yaw/pitch of the 3D hands."""
    c = _copy(clip)
    aspect = float(c["aspect"])
    _rot2d(c["hands_img"], roll, aspect)
    _rot2d(c["pose"], roll, aspect)
    cr, sr = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    c["hands_world"] = (c["hands_world"] @ (rz @ ry @ rx).T.astype(np.float32)).astype(np.float32)
    return c


def occlude(clip: dict, rng: np.random.Generator, span: float = 0.3, drop: float = 0.1) -> dict:
    """Hide one hand for a contiguous span and randomly lose detections on single frames."""
    c = _copy(clip)
    T = c["hand_present"].shape[0]
    if span > 0:
        length = max(1, int(T * span))
        start = int(rng.integers(0, max(1, T - length + 1)))
        c["hand_present"][start:start + length, int(rng.integers(0, 2))] = 0.0
    if drop > 0:
        c["hand_present"] *= (rng.random(c["hand_present"].shape) > drop).astype(np.float32)
    return c


def jitter(clip: dict, rng: np.random.Generator, img_sigma: float = 0.004, world_sigma: float = 0.003) -> dict:
    c = _copy(clip)
    c["hands_img"] += rng.normal(0, img_sigma, c["hands_img"].shape).astype(np.float32)
    c["pose"] += rng.normal(0, img_sigma, c["pose"].shape).astype(np.float32)
    c["hands_world"] += rng.normal(0, world_sigma, c["hands_world"].shape).astype(np.float32)
    return c


def train_augment(clip: dict, rng: np.random.Generator, allow_mirror: bool = True) -> dict:
    c = clip
    if allow_mirror and rng.random() < 0.5:
        c = mirror(c)
    c = rotate(c, roll=rng.normal(0, 0.12), yaw=rng.normal(0, 0.30), pitch=rng.normal(0, 0.15))
    if rng.random() < 0.5:
        c = occlude(c, rng, span=rng.uniform(0.1, 0.35) if rng.random() < 0.5 else 0.0,
                    drop=rng.uniform(0.0, 0.15))
    return jitter(c, rng)


def robustness_perturb(clip: dict, rng: np.random.Generator) -> dict:
    """Fixed 'hard conditions' used by the robustness metric on cached landmarks:
    angled signer + tilted camera + one hand occluded 30% of the clip + 15% lost detections."""
    sign = 1.0 if rng.random() < 0.5 else -1.0
    c = rotate(clip, roll=sign * np.deg2rad(12), yaw=sign * np.deg2rad(25), pitch=np.deg2rad(10))
    c = occlude(c, rng, span=0.3, drop=0.15)
    return jitter(c, rng, img_sigma=0.006, world_sigma=0.004)
