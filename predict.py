"""Evaluator entry point.

    from predict import predict
    predict("clips/test_0412.mp4")
    -> {"clip_id": "test_0412", "prediction": "HAPPY", "top5": [...], "confidence": 0.81, "latency_ms": 23.4}

Runs fully on-device: MediaPipe landmarks + an INT8 ONNX classifier, no network calls.
`latency_ms` is the classifier inference time; landmark time is reported by scripts/benchmark.py.
"""
from __future__ import annotations

import json
import sys

from echogesture.predictor import get_predictor


def predict(video_path: str) -> dict:
    """Return the predicted gloss/word for one video clip. Must run fully on-device."""
    return get_predictor().predict(video_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python predict.py <video_path>")
    print(json.dumps(predict(sys.argv[1])))
