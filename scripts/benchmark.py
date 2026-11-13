"""Measure the Track 2 metrics and write results.csv (+ results/benchmark_details.json).

    python scripts/benchmark.py --manifest data/manifest.csv --cache data/cache --device "HP Victus 15, Core i5"
    python scripts/benchmark.py ... --hard-cache data/cache_hard      # robustness from degraded videos
    python scripts/benchmark.py ... --thermal-minutes 10              # sustained-load latency drift
    python scripts/benchmark.py ... --sample-video some_clip.mp4      # also time the landmark stage
"""
import argparse
import csv
import json
import platform
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import onnxruntime as ort

from echogesture import config
from echogesture.augment import robustness_perturb
from echogesture.data import load_split
from echogesture.features import clip_to_input
from echogesture.predictor import Classifier


def top1(clf, clips, y, perturb_seed=None):
    rng = np.random.default_rng(perturb_seed) if perturb_seed is not None else None
    hits = 0
    for c, label in zip(clips, y):
        if rng is not None:
            c = robustness_perturb(c, rng)
        hits += int(clf.logits(clip_to_input(c)).argmax()) == int(label)
    return 100.0 * hits / max(1, len(y))


def passed(metric, value):
    if value is None:
        return ""
    op, target, _ = config.TARGETS[metric]
    return "PASS" if (value >= target if op == ">=" else value <= target) else "FAIL"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--splits", nargs="*", default=["test"])
    ap.add_argument("--hard-cache", help="landmarks extracted with --degrade (same clips)")
    ap.add_argument("--device", default=platform.processor() or platform.machine())
    ap.add_argument("--runs", type=int, default=1000)
    ap.add_argument("--thermal-minutes", type=float, default=0)
    ap.add_argument("--avg-power-mw", type=float, help="fill in if you measured power on a phone")
    ap.add_argument("--sample-video")
    ap.add_argument("--out", default="results.csv")
    args = ap.parse_args()

    clf = Classifier()
    runtime = f"ONNX Runtime {ort.__version__} INT8, CPU, 1 thread"
    details = {}

    clips, y, ids = load_split(args.manifest, args.cache, clf.vocab, tuple(args.splits))
    acc = drop = None
    if clips:
        acc = top1(clf, clips, y)
        if args.hard_cache:
            hclips, hy, _ = load_split(args.manifest, args.hard_cache, clf.vocab, tuple(args.splits))
            hard = top1(clf, hclips, hy)
            details["robustness_source"] = f"degraded videos ({len(hclips)} clips)"
        else:
            hard = top1(clf, clips, y, perturb_seed=0)
            details["robustness_source"] = "landmark-level perturbation (angle, occlusion, lost detections)"
        drop = acc - hard
        details.update(test_clips=len(clips), top1_clean=acc, top1_hard=hard)
    else:
        print("No labelled clips for the requested split: accuracy rows left empty.")

    x = np.random.default_rng(0).normal(0, 0.5, (1, config.SEQ_LEN, config.FEAT_DIM)).astype(np.float32)
    lat = [clf.classify(x)[1] for _ in range(args.runs)]
    p95 = float(np.percentile(lat, 95))
    details.update(latency_ms_median=float(np.median(lat)), latency_ms_p95=p95)

    size_mb = config.INT8_MODEL.stat().st_size / 1e6

    drift = None
    if args.thermal_minutes > 0:
        print(f"sustained load for {args.thermal_minutes} min ...", flush=True)
        end = time.time() + args.thermal_minutes * 60
        first, last, window = [], [], min(60.0, args.thermal_minutes * 60 / 4)
        start = time.time()
        while time.time() < end:
            ms = clf.classify(x)[1]
            now = time.time()
            if now - start < window:
                first.append(ms)
            elif end - now < window:
                last.append(ms)
        drift = 100.0 * (np.median(last) - np.median(first)) / np.median(first)
        details.update(thermal_first_ms=float(np.median(first)), thermal_last_ms=float(np.median(last)))

    if args.sample_video:
        from echogesture.predictor import Predictor
        p = Predictor(classifier=clf)
        p.predict(args.sample_video)  # warm up
        details["sample_video_timing"] = p.last_timing
        t0 = time.time()
        runs = 0
        while time.time() - t0 < 6:  # ~6 s sustained end-to-end loop
            p.predict(args.sample_video)
            runs += 1
        elapsed = time.time() - t0
        frames = runs * p.last_timing.get("frames", 32)
        details["end_to_end_fps"] = round(frames / elapsed, 1)
        details["end_to_end_ms_per_clip"] = round(elapsed * 1000 / runs, 1)

    rows = [("top1_accuracy", acc), ("latency_p95", p95), ("model_size", size_mb),
            ("accuracy_drop_robustness", drop), ("avg_power", args.avg_power_mw), ("thermal_drift_10min", drift)]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "target", "measured", "unit", "device", "runtime", "pass"])
        for metric, value in rows:
            op, target, unit = config.TARGETS[metric]
            w.writerow([metric, f"{op}{target:g}", "" if value is None else f"{value:.2f}", unit,
                        args.device, runtime, passed(metric, value)])
    out_dir = config.ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "benchmark_details.json").write_text(json.dumps(details, indent=2))
    print(Path(args.out).read_text())
    print(json.dumps(details, indent=2))
