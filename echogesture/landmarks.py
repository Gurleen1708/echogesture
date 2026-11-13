"""MediaPipe (Tasks API) hand + pose landmark extraction. Runs fully on-device.

Raw per-frame output (the "raw clip" format used everywhere else):
    hands_img    (2, 21, 3)  normalised image coords, slot 0 = Left, slot 1 = Right
    hands_world  (2, 21, 3)  metric 3D hand landmarks (metres, hand-centred)
    hand_present (2,)        1.0 if that slot was detected in this frame
    pose         (33, 3)     normalised image coords
    pose_present ()          1.0 if a body was detected
A clip stacks these along a leading time axis and adds `aspect` (width / height).
"""
from __future__ import annotations

import cv2
import numpy as np

from . import config

_BUFFERS: dict[str, bytes] = {}


def _asset(path) -> bytes:
    key = str(path)
    if key not in _BUFFERS:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing MediaPipe model: {path}\n"
                "Run:  python scripts/download_mediapipe_models.py"
            )
        _BUFFERS[key] = path.read_bytes()
    return _BUFFERS[key]


def empty_frame() -> dict:
    return {
        "hands_img": np.zeros((2, 21, 3), np.float32),
        "hands_world": np.zeros((2, 21, 3), np.float32),
        "hand_present": np.zeros((2,), np.float32),
        "pose": np.zeros((33, 3), np.float32),
        "pose_present": np.float32(0.0),
    }


def stack_frames(frames: list[dict], aspect: float) -> dict:
    """List of per-frame dicts -> one raw clip dict with a time axis."""
    if not frames:
        frames = [empty_frame()]
    clip = {k: np.stack([f[k] for f in frames]).astype(np.float32) for k in frames[0]}
    clip["aspect"] = np.float32(aspect)
    return clip


class LandmarkExtractor:
    """Wraps MediaPipe HandLandmarker + PoseLandmarker in VIDEO mode.

    Create one instance per clip (timestamps must increase within an instance).
    Model files are cached in memory, so re-creating it is cheap.
    """

    def __init__(self, use_pose: bool = True, min_conf: float = 0.4):
        import mediapipe as mp

        self._mp = mp
        vision = mp.tasks.vision
        base = mp.tasks.BaseOptions
        self.hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=base(model_asset_buffer=_asset(config.HAND_TASK)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=min_conf,
                min_hand_presence_confidence=min_conf,
                min_tracking_confidence=min_conf,
            )
        )
        self.pose = None
        if use_pose:
            self.pose = vision.PoseLandmarker.create_from_options(
                vision.PoseLandmarkerOptions(
                    base_options=base(model_asset_buffer=_asset(config.POSE_TASK)),
                    running_mode=vision.RunningMode.VIDEO,
                    num_poses=1,
                    min_pose_detection_confidence=min_conf,
                    min_pose_presence_confidence=min_conf,
                    min_tracking_confidence=min_conf,
                )
            )
        self._last_ts = -1

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> dict:
        mp = self._mp
        ts = max(int(timestamp_ms), self._last_ts + 1)  # must be strictly increasing
        self._last_ts = ts
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        out = empty_frame()

        res = self.hands.detect_for_video(image, ts)
        best = [-1.0, -1.0]
        pending = []
        for i, lms in enumerate(res.hand_landmarks):
            cat = res.handedness[i][0]
            slot = 0 if cat.category_name.lower().startswith("l") else 1
            img = np.array([[p.x, p.y, p.z] for p in lms], np.float32)
            world = np.array([[p.x, p.y, p.z] for p in res.hand_world_landmarks[i]], np.float32)
            pending.append((float(cat.score), slot, img, world))
        # Highest-confidence hand gets its own slot; a second hand with the same
        # label falls into the other slot instead of being dropped.
        for score, slot, img, world in sorted(pending, key=lambda t: -t[0]):
            if best[slot] >= 0:
                slot = 1 - slot
                if best[slot] >= 0:
                    continue
            best[slot] = score
            out["hands_img"][slot] = img
            out["hands_world"][slot] = world
            out["hand_present"][slot] = 1.0

        if self.pose is not None:
            pres = self.pose.detect_for_video(image, ts)
            if pres.pose_landmarks:
                out["pose"] = np.array([[p.x, p.y, p.z] for p in pres.pose_landmarks[0]], np.float32)
                out["pose_present"] = np.float32(1.0)
        return out

    def close(self):
        for t in (self.hands, self.pose):
            if t is not None:
                t.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --- video helpers -----------------------------------------------------------

def degrade(frame: np.ndarray, mode: str, rng: np.random.Generator) -> np.ndarray:
    """Synthetic hard conditions used for the robustness metric."""
    if mode in ("lowlight", "all"):
        f = (frame.astype(np.float32) / 255.0) ** 2.2 * 0.45  # darker, crushed shadows
        f += rng.normal(0, 0.03, f.shape)                     # sensor noise
        frame = (np.clip(f, 0, 1) * 255).astype(np.uint8)
    if mode in ("occlusion", "all"):
        h, w = frame.shape[:2]
        bw, bh = int(w * 0.18), int(h * 0.22)
        x0 = int(rng.uniform(0.2, 0.62) * w)
        y0 = int(rng.uniform(0.3, 0.6) * h)
        frame = frame.copy()
        frame[y0:y0 + bh, x0:x0 + bw] = 30
    return frame


def read_clip_frames(video_path: str, max_frames: int = config.MAX_CLIP_FRAMES,
                     frame_start: int = 0, frame_end: int = -1):
    """Return (frames, timestamps_ms, aspect): up to `max_frames` evenly spaced frames."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if not (1.0 <= fps <= 240.0):
        fps = 25.0
    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx >= frame_start and (frame_end < 0 or idx <= frame_end):
            frames.append(frame)
        idx += 1
    cap.release()
    if not frames:
        raise IOError(f"No frames decoded from: {video_path}")
    n = len(frames)
    keep = np.unique(np.linspace(0, n - 1, min(n, max_frames)).round().astype(int))
    h, w = frames[0].shape[:2]
    return [frames[i] for i in keep], [int(i * 1000.0 / fps) for i in keep], w / float(h)


def extract_clip(video_path: str, degrade_mode: str | None = None, seed: int = 0,
                 frame_start: int = 0, frame_end: int = -1, timing: dict | None = None) -> dict:
    """Video file -> raw landmark clip."""
    import time

    frames, stamps, aspect = read_clip_frames(video_path, frame_start=frame_start, frame_end=frame_end)
    rng = np.random.default_rng(seed)
    per_frame = []
    out = []
    with LandmarkExtractor() as ex:
        for frame, ts in zip(frames, stamps):
            if degrade_mode:
                frame = degrade(frame, degrade_mode, rng)
            t0 = time.perf_counter()
            out.append(ex.process(frame, ts))
            per_frame.append((time.perf_counter() - t0) * 1000.0)
    if timing is not None:
        timing["landmark_ms_per_frame"] = float(np.median(per_frame))
        timing["frames"] = len(frames)
    return stack_frames(out, aspect)
