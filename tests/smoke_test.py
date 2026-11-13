"""End-to-end check on synthetic landmarks (no dataset, no MediaPipe models needed).

    python tests/smoke_test.py

Builds a fake 20-word dataset, then runs train -> export/quantize -> benchmark -> predict()
-> predictions.jsonl -> HUD render in a temp folder. Proves the plumbing; says nothing about accuracy.
"""
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from echogesture import config
from echogesture.data import save_clip

rng = np.random.default_rng(0)
WORDS = [f"WORD{i:02d}" for i in range(20)]


def fake_clip(proto, rng):
    T = int(rng.integers(30, 60))
    t = np.linspace(0, 1, T)[:, None, None]
    clip = {"hands_img": np.zeros((T, 2, 21, 3), np.float32), "hands_world": np.zeros((T, 2, 21, 3), np.float32),
            "hand_present": np.ones((T, 2), np.float32), "pose": np.zeros((T, 33, 3), np.float32),
            "pose_present": np.ones((T,), np.float32), "aspect": np.float32(16 / 9)}
    for s in (0, 1):
        shape = proto["a"][s] * (1 - t) + proto["b"][s] * t
        clip["hands_world"][:, s] = shape + rng.normal(0, 0.004, shape.shape)
        centre = proto["pa"][s] * (1 - t[:, 0]) + proto["pb"][s] * t[:, 0]
        clip["hands_img"][:, s, :, :2] = centre[:, None, :] + shape[..., :2] * 0.8
    clip["pose"][:, :, :2] = 0.5 + rng.normal(0, 0.002, (T, 33, 2))
    clip["pose"][:, 11, :2] = [0.58, 0.40]
    clip["pose"][:, 12, :2] = [0.42, 0.40]
    return clip


def run(*cmd):
    print("\n$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([sys.executable, *map(str, cmd)], check=True, cwd=ROOT)


if __name__ == "__main__":
    tmp = Path(tempfile.mkdtemp(prefix="echogesture_smoke_"))
    backup = {p: p.read_bytes() for p in (config.VOCAB_FILE, config.INT8_MODEL, config.FP32_MODEL, config.META_FILE,
                                         config.MODELS_DIR / "checkpoint.pt") if p.exists()}
    try:
        config.VOCAB_FILE.write_text("\n".join(WORDS) + "\n")
        protos = [{"a": rng.normal(0, 0.04, (2, 21, 3)), "b": rng.normal(0, 0.04, (2, 21, 3)),
                   "pa": rng.uniform(0.3, 0.7, (2, 2)), "pb": rng.uniform(0.3, 0.7, (2, 2))} for _ in WORDS]
        rows = []
        for w, proto in zip(WORDS, protos):
            for k in range(16):
                cid = f"{w}_{k:02d}"
                save_clip(tmp / "cache" / f"{cid}.npz", fake_clip(proto, rng))
                rows.append([cid, "", w, "train" if k < 10 else "val" if k < 13 else "test"])
        manifest = tmp / "manifest.csv"
        with open(manifest, "w", newline="") as f:
            csv.writer(f).writerows([["clip_id", "path", "label", "split"], *rows])

        run("scripts/train.py", "--manifest", manifest, "--cache", tmp / "cache", "--epochs", 25)
        run("scripts/export_quantize.py", "--manifest", manifest, "--cache", tmp / "cache")
        run("scripts/benchmark.py", "--manifest", manifest, "--cache", tmp / "cache", "--runs", 300,
            "--out", tmp / "results.csv")

        # predict() on a real (blank) video file with a stub landmark stage
        from echogesture.data import load_clip
        from echogesture.live import Stats, StreamDecoder, draw_hud
        from echogesture.predictor import Classifier, Predictor, to_jsonl_line

        video = tmp / "WORD03_13.mp4"
        vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 25, (320, 180))
        for _ in range(30):
            vw.write(np.zeros((180, 320, 3), np.uint8))
        vw.release()
        clf = Classifier()
        stub = lambda path, timing=None: load_clip(tmp / "cache" / f"{Path(path).stem}.npz")
        res = Predictor(classifier=clf, extractor=stub).predict(str(video))
        print("\npredict() ->", to_jsonl_line(res))
        assert set(res) == {"clip_id", "prediction", "top5", "confidence", "latency_ms"}
        assert res["clip_id"] == "WORD03_13" and res["prediction"].isupper() and len(res["top5"]) == 5
        assert res["prediction"] == "WORD03", "smoke model should get this synthetic clip right"

        # streaming decoder + HUD
        clip = load_clip(tmp / "cache" / "WORD07_14.npz")
        dec = StreamDecoder(clf, aspect=16 / 9)
        stats = Stats()
        words, canvas = [], None
        for t in range(clip["hand_present"].shape[0]):
            raw = {k: clip[k][t] for k in ("hands_img", "hands_world", "hand_present", "pose", "pose_present")}
            w = dec.push(raw)
            if w:
                words.append(w)
            stats.add(18.0, dec.model_ms, 19.0 + dec.model_ms)
            canvas = draw_hud(np.full((720, 1280, 3), 60, np.uint8), raw, dec, words, stats,
                              "INT8 ONNX  ·  0.69 MB  ·  20 words  ·  open weights", expected="WORD07")
        cv2.imwrite(str(tmp / "hud.png"), canvas)
        print("stream decoder words:", words, "| HUD frame:", tmp / "hud.png")
        assert words == ["WORD07"], words

        # demo reel on two blank videos, with the landmark stage replayed from the cache
        import runpy

        import echogesture.landmarks as lm

        order = ["WORD03_13", "WORD07_14"]  # sorted order of the footage files

        class FakeExtractor:
            made = 0

            def __init__(self, *a, **k):
                self.clip = load_clip(tmp / "cache" / f"{order[FakeExtractor.made % len(order)]}.npz")
                FakeExtractor.made += 1
                self.t = 0

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                pass

            def process(self, frame, ts):
                t = min(self.t, self.clip["hand_present"].shape[0] - 1)
                self.t += 1
                return {k: self.clip[k][t] for k in ("hands_img", "hands_world", "hand_present", "pose", "pose_present")}

        footage = tmp / "footage"
        footage.mkdir()
        for name in order:
            vw = cv2.VideoWriter(str(footage / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 25, (320, 180))
            for _ in range(45):
                vw.write(np.zeros((180, 320, 3), np.uint8))
            vw.release()
        orig_extractor, orig_argv = lm.LandmarkExtractor, sys.argv
        try:
            lm.LandmarkExtractor = FakeExtractor
            sys.argv = ["make_demo_reel.py", "--clips", str(footage), "--out", str(tmp / "reel.mp4"),
                        "--size", "640x360", "--fps", "10"]
            runpy.run_path(str(ROOT / "scripts" / "make_demo_reel.py"), run_name="__main__")
        finally:
            lm.LandmarkExtractor, sys.argv = orig_extractor, orig_argv
        reel_summary = json.loads((ROOT / "results" / "demo_reel_summary.json").read_text())
        print("demo reel:", tmp / "reel.mp4", "| accuracy", reel_summary["accuracy"])
        assert (tmp / "reel.mp4").stat().st_size > 10_000
        assert reel_summary["accuracy"] == 100.0, reel_summary

        meta = json.loads(config.META_FILE.read_text())
        assert meta["int8_mb"] < 20
        print("\nSMOKE TEST PASSED  (artefacts in", tmp, ")")
    finally:  # leave the repo's real model files untouched
        for p in (config.VOCAB_FILE, config.INT8_MODEL, config.FP32_MODEL, config.META_FILE,
                  config.MODELS_DIR / "checkpoint.pt", config.MODELS_DIR / "checkpoint.history.json"):
            if p in backup:
                p.write_bytes(backup[p])
            elif p.exists():
                p.unlink()
