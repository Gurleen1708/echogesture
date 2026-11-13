"""Quick disposable model for UI testing only.

Trains for 5 epochs on random features and leaves the ONNX + vocab in models/.
Predictions will be nonsense — this is just so the live demo can run end to end
and you can check that the camera, hand tracking, and HUD all work.

Delete models/echogesture_int8.onnx before running real training.

    python scripts/make_stub_model.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                       QuantType, quantize_static)

from echogesture import config
from echogesture.model import SignTCN

VOCAB = [
    "HELLO", "THANK_YOU", "YES", "NO", "PLEASE", "LEARN", "BOOK", "DRINK",
    "EAT", "HELP", "TIME", "COMPUTER", "FRIEND", "FAMILY", "WORK", "SCHOOL",
    "HOME", "GOOD", "BAD", "LOVE",
]


class Reader(CalibrationDataReader):
    def __init__(self, samples):
        self.it = iter([{"features": s[None].astype(np.float32)} for s in samples])

    def get_next(self):
        return next(self.it, None)


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = SignTCN(len(VOCAB))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    print("training stub model (5 epochs on random data) ...")
    for epoch in range(5):
        x = torch.from_numpy(rng.normal(0, 0.5, (256, config.SEQ_LEN, config.FEAT_DIM)).astype(np.float32))
        y = torch.randint(0, len(VOCAB), (256,))
        loss = loss_fn(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        print(f"  epoch {epoch + 1}  loss {loss.item():.3f}")

    print("exporting to ONNX ...")
    model.eval()
    dummy = torch.zeros(1, config.SEQ_LEN, config.FEAT_DIM)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    kw = dict(input_names=["features"], output_names=["logits"], opset_version=17)
    try:
        torch.onnx.export(model, dummy, str(config.FP32_MODEL), dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(model, dummy, str(config.FP32_MODEL), **kw)

    print("quantizing to INT8 ...")
    calib = [rng.normal(0, 0.5, (config.SEQ_LEN, config.FEAT_DIM)).astype(np.float32) for _ in range(100)]
    quantize_static(str(config.FP32_MODEL), str(config.INT8_MODEL), Reader(calib),
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)

    config.VOCAB_FILE.write_text("\n".join(VOCAB) + "\n", encoding="utf-8")
    config.META_FILE.write_text(json.dumps({
        "classes": len(VOCAB), "seq_len": config.SEQ_LEN, "feat_dim": config.FEAT_DIM,
        "int8_mb": round(config.INT8_MODEL.stat().st_size / 1e6, 3),
        "stub": True, "note": "random weights — replace with real training before submission",
    }, indent=2))
    print(f"\nstub model saved: {config.INT8_MODEL.name}  ({config.INT8_MODEL.stat().st_size / 1e6:.2f} MB)")
    print(f"vocab:            {config.VOCAB_FILE.name}  ({len(VOCAB)} words)")
    print("\nnow run:  python scripts\\live_demo.py --mirror")


if __name__ == "__main__":
    main()
