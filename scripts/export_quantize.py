"""Conversion script: PyTorch checkpoint -> ONNX FP32 -> ONNX INT8 (static, post-training).

    python scripts/export_quantize.py --manifest data/manifest.csv --cache data/cache
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

from echogesture import config
from echogesture.data import load_split
from echogesture.features import clip_to_input
from echogesture.model import SignTCN


class Reader(CalibrationDataReader):
    def __init__(self, samples):
        self.it = iter([{"features": s[None].astype(np.float32)} for s in samples])

    def get_next(self):
        return next(self.it, None)


def accuracy(path, x, y):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    pred = [int(sess.run(None, {"features": s[None]})[0][0].argmax()) for s in x]
    return float((np.array(pred) == y).mean() * 100)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--checkpoint", default=str(config.MODELS_DIR / "checkpoint.pt"))
    ap.add_argument("--calib", type=int, default=300, help="number of training clips used for calibration")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    vocab = ckpt["vocab"]
    model = SignTCN(len(vocab))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    dummy = torch.zeros(1, config.SEQ_LEN, config.FEAT_DIM)
    kwargs = dict(input_names=["features"], output_names=["logits"], opset_version=17)
    try:
        torch.onnx.export(model, dummy, str(config.FP32_MODEL), dynamo=False, **kwargs)
    except TypeError:  # older PyTorch without the 'dynamo' argument
        torch.onnx.export(model, dummy, str(config.FP32_MODEL), **kwargs)

    train_clips, _, _ = load_split(args.manifest, args.cache, vocab, ("train",))
    rng = np.random.default_rng(0)
    pick = rng.permutation(len(train_clips))[: args.calib]
    calib = [clip_to_input(train_clips[i]) for i in pick]
    quantize_static(str(config.FP32_MODEL), str(config.INT8_MODEL), Reader(calib),
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)

    config.VOCAB_FILE.write_text("\n".join(vocab) + "\n", encoding="utf-8")
    meta = {"classes": len(vocab), "seq_len": config.SEQ_LEN, "feat_dim": config.FEAT_DIM,
            "fp32_mb": round(config.FP32_MODEL.stat().st_size / 1e6, 3),
            "int8_mb": round(config.INT8_MODEL.stat().st_size / 1e6, 3)}
    for name, splits in (("val", ("val",)), ("test", ("test",))):
        clips, y, _ = load_split(args.manifest, args.cache, vocab, splits)
        if clips:
            x = np.stack([clip_to_input(c) for c in clips])
            meta[f"{name}_top1_fp32"] = round(accuracy(config.FP32_MODEL, x, y), 2)
            meta[f"{name}_top1_int8"] = round(accuracy(config.INT8_MODEL, x, y), 2)
    config.META_FILE.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
