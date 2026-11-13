"""On-device inference: video -> landmarks -> features -> INT8 ONNX model -> gloss."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import config
from .features import clip_to_input


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class Classifier:
    """The quantised classifier on ONNX Runtime (CPU, no network)."""

    def __init__(self, model_path=config.INT8_MODEL, vocab_path=config.VOCAB_FILE, threads: int = 1):
        import onnxruntime as ort

        if not Path(model_path).exists():
            raise FileNotFoundError(f"{model_path} not found. Train and run scripts/export_quantize.py first.")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.vocab = config.load_vocab(vocab_path)
        self.model_mb = Path(model_path).stat().st_size / 1e6
        self.runtime = f"ONNX Runtime {ort.__version__} INT8 CPU"
        warm = np.zeros((1, config.SEQ_LEN, config.FEAT_DIM), np.float32)
        for _ in range(5):
            self.session.run(None, {self.input_name: warm})

    def logits(self, x: np.ndarray) -> np.ndarray:
        """x: (SEQ_LEN, FEAT_DIM) or (1, SEQ_LEN, FEAT_DIM)."""
        if x.ndim == 2:
            x = x[None]
        return self.session.run(None, {self.input_name: x.astype(np.float32)})[0][0]

    def classify(self, x: np.ndarray):
        t0 = time.perf_counter()
        probs = softmax(self.logits(x))
        return probs, (time.perf_counter() - t0) * 1000.0


class Predictor:
    def __init__(self, classifier: Classifier | None = None, extractor=None):
        self.classifier = classifier or Classifier()
        if extractor is None:
            from .landmarks import extract_clip
            extractor = extract_clip
        self.extract = extractor
        self.last_timing: dict = {}

    def predict(self, video_path: str) -> dict:
        timing: dict = {}
        clip = self.extract(str(video_path), timing=timing)
        probs, ms = self.classifier.classify(clip_to_input(clip))
        timing["model_ms"] = ms
        self.last_timing = timing
        order = np.argsort(-probs)[:5]
        vocab = self.classifier.vocab
        return {
            "clip_id": Path(video_path).stem,
            "prediction": vocab[int(order[0])],
            "top5": [vocab[int(i)] for i in order],
            "confidence": round(float(probs[order[0]]), 4),
            "latency_ms": round(float(ms), 2),
        }


_PREDICTOR: Predictor | None = None


def get_predictor() -> Predictor:
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = Predictor()
    return _PREDICTOR


def to_jsonl_line(result: dict) -> str:
    return json.dumps(result, ensure_ascii=False)
