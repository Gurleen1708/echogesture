"""Central configuration. Everything that is likely to change lives here."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- paths -----------------------------------------------------------------
MODELS_DIR = ROOT / "models"
MEDIAPIPE_DIR = MODELS_DIR / "mediapipe"
HAND_TASK = MEDIAPIPE_DIR / "hand_landmarker.task"
POSE_TASK = MEDIAPIPE_DIR / "pose_landmarker_lite.task"
INT8_MODEL = MODELS_DIR / "echogesture_int8.onnx"
FP32_MODEL = MODELS_DIR / "echogesture_fp32.onnx"
VOCAB_FILE = MODELS_DIR / "vocab.txt"
META_FILE = MODELS_DIR / "model_meta.json"

# --- sequence / feature layout --------------------------------------------
SEQ_LEN = 32            # frames fed to the classifier
MAX_CLIP_FRAMES = 48    # at most this many frames of a clip go through MediaPipe
HAND_FEATS = 72         # 63 canonical 3D shape + 3 palm-up + 3 palm-normal + 2 position + 1 presence
POSE_FEATS = 10         # nose, elbows, wrists (x, y) in shoulder units
FEAT_DIM = 2 * HAND_FEATS + POSE_FEATS  # 154
FEAT_CLIP = 4.0         # features are clipped to [-4, 4] (keeps INT8 ranges tight)

# --- model -----------------------------------------------------------------
CHANNELS = 160
BLOCK_DILATIONS = (1, 2, 4, 1)
DROPOUT = 0.3

# --- submission targets (from the Track 2 brief) ---------------------------
TARGETS = {
    "top1_accuracy": (">=", 70.0, "%"),
    "latency_p95": ("<=", 50.0, "ms"),
    "model_size": ("<=", 25.0, "MB"),
    "accuracy_drop_robustness": ("<=", 5.0, "pt"),
    "avg_power": ("<=", 500.0, "mW"),
    "thermal_drift_10min": ("<=", 20.0, "%"),
}


def load_vocab(path: Path | str = VOCAB_FILE) -> list[str]:
    """One gloss per line, stored uppercase (the brief requires uppercase predictions)."""
    words = [w.strip().upper() for w in Path(path).read_text(encoding="utf-8").splitlines()]
    return [w for w in words if w]
