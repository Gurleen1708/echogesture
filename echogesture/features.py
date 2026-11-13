"""Raw landmarks -> pose-invariant feature vectors (pure NumPy, no learned parts).

Per hand (72 values):
  * 63: 3D hand shape in a canonical palm frame (wrist at origin, unit palm
        length, palm axes aligned) -> invariant to hand position, distance to
        camera and camera viewpoint.
  *  6: palm "up" and palm-normal directions in camera space (palm orientation
        carries meaning in sign language, so it is kept as a separate, small cue).
  *  2: wrist position relative to the shoulder centre, in shoulder widths
        -> invariant to where the signer stands and how far away they are.
  *  1: presence flag (all hand values are zero when the hand is not seen).
Body (10 values): nose, elbows and wrists in the same shoulder-normalised frame.
"""
from __future__ import annotations

import numpy as np

from . import config

EPS = 1e-6
POSE_JOINTS = (0, 13, 14, 15, 16)  # nose, elbows, wrists
L_SHOULDER, R_SHOULDER = 11, 12


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + EPS)


def _body_frame(pose: np.ndarray, pose_present: np.ndarray, aspect: float):
    """Shoulder centre + width per frame; frames without a body borrow the nearest valid one."""
    T = pose.shape[0]
    xy = pose[..., :2].astype(np.float32).copy()
    xy[..., 0] *= aspect
    width = np.linalg.norm(xy[:, L_SHOULDER] - xy[:, R_SHOULDER], axis=-1)
    valid = (pose_present > 0.5) & (width > 1e-3)
    if valid.any():
        idx = np.where(valid)[0]
        nearest = idx[np.abs(idx[None, :] - np.arange(T)[:, None]).argmin(axis=1)]
        xy = xy[nearest]
        width = width[nearest]
        center = (xy[:, L_SHOULDER] + xy[:, R_SHOULDER]) / 2.0
    else:  # no body at all: assume a centred signer filling ~1/3 of the frame
        center = np.tile(np.array([0.5 * aspect, 0.45], np.float32), (T, 1))
        width = np.full((T,), 0.3 * aspect, np.float32)
        xy = np.tile(center[:, None, :], (1, 33, 1))
    return xy, center, width, valid.any()


def _hand_block(world, img, present, center, width, aspect):
    T = world.shape[0]
    p = world - world[:, :1]                                   # wrist at origin
    scale = np.linalg.norm(p[:, 9], axis=-1, keepdims=True) + EPS   # wrist -> middle MCP
    u = _unit(p[:, 9])                                         # palm "up"
    n = _unit(np.cross(u, p[:, 17] - p[:, 5]))                 # palm normal
    v = np.cross(n, u)                                         # across the palm
    rot = np.stack([v, u, n], axis=1)                          # (T, 3, 3), rows = axes
    canon = np.einsum("tij,tkj->tki", rot, p) / scale[:, :, None]
    wrist = img[:, 0, :2].copy()
    wrist[:, 0] *= aspect
    pos = (wrist - center) / width[:, None]
    feat = np.concatenate([canon.reshape(T, 63), u, n, pos, np.ones((T, 1), np.float32)], axis=1)
    return feat * present[:, None]


def clip_features(clip: dict) -> np.ndarray:
    """Raw clip dict -> (T, FEAT_DIM) float32."""
    aspect = float(clip["aspect"])
    xy, center, width, has_body = _body_frame(clip["pose"], clip["pose_present"], aspect)
    blocks = [
        _hand_block(clip["hands_world"][:, s], clip["hands_img"][:, s],
                    clip["hand_present"][:, s], center, width, aspect)
        for s in (0, 1)
    ]
    body = (xy[:, POSE_JOINTS] - center[:, None, :]) / width[:, None, None]
    body = body.reshape(len(width), -1) * (1.0 if has_body else 0.0)
    feats = np.concatenate(blocks + [body], axis=1).astype(np.float32)
    assert feats.shape[1] == config.FEAT_DIM
    return np.clip(np.nan_to_num(feats), -config.FEAT_CLIP, config.FEAT_CLIP)


def sample_indices(n_src: int, n_out: int = config.SEQ_LEN, rng: np.random.Generator | None = None):
    """Evenly spaced frame indices; with `rng`, a random temporal crop + jitter (training)."""
    if rng is None:
        pos = np.linspace(0, n_src - 1, n_out)
    else:
        start = rng.uniform(0, 0.12) * (n_src - 1)
        end = (n_src - 1) - rng.uniform(0, 0.12) * (n_src - 1)
        pos = np.linspace(start, max(end, start), n_out)
        pos = pos + rng.uniform(-0.5, 0.5, n_out) * max((end - start) / n_out, 0.0)
    return np.clip(np.round(pos), 0, n_src - 1).astype(int)


def clip_to_input(clip: dict, rng: np.random.Generator | None = None) -> np.ndarray:
    """Raw clip -> (SEQ_LEN, FEAT_DIM) model input."""
    feats = clip_features(clip)
    return feats[sample_indices(feats.shape[0], config.SEQ_LEN, rng)]
