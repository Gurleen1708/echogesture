"""Build the 2-3 minute demo video automatically from a folder of recorded clips.

    python scripts/make_demo_reel.py --clips E:\\demo_footage --out demo_reel.mp4

Clip naming: the word being signed is read from the filename up to the first '_' or '-',
e.g.  HELLO_01.mp4, THANK_YOU-take2.mp4 (use THANKYOU or THANK_YOU exactly as in vocab.txt).
Clips with no recognisable word still play; they just get no expected/verdict overlay.

Output: title card -> each clip with the live HUD (expected word shown, verdict frozen for 1.5 s)
-> summary card (accuracy, mean end-to-end ms, FPS, model size). Also writes
results/demo_reel_summary.json. No video editor needed.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from echogesture.landmarks import LandmarkExtractor
from echogesture.live import INK, MINT, RED, WHITE, Stats, StreamDecoder, _rrect, _text, _tw, draw_hud, title_card
from echogesture.predictor import Classifier

VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")


def expected_word(path: Path, vocab: set[str]) -> str | None:
    stem = path.stem.upper()
    for sep in ("-", "_"):
        head = stem.split(sep)[0]
        if head in vocab:
            return head
    # allow THANK_YOU style multi-token words: longest vocab prefix of the stem
    best = [v for v in vocab if stem.startswith(v)]
    return max(best, key=len) if best else None


def verdict_overlay(frame, expected, predicted, s):
    h, w = frame.shape[:2]
    ok = expected is not None and predicted == expected
    if expected is None:
        msg, col = f"PREDICTED  {predicted or '-'}", MINT
    elif ok:
        msg, col = f"CORRECT  ·  {predicted}", MINT
    else:
        msg, col = f"PREDICTED {predicted or '-'}   EXPECTED {expected}", RED
    tw = _tw(msg, 0.9 * s, 2) + 40 * s
    _rrect(frame, (w - tw) / 2, h * 0.42, (w + tw) / 2, h * 0.42 + 64 * s, col, r=int(14 * s), alpha=0.92)
    _text(frame, msg, (w - tw) / 2 + 20 * s, h * 0.42 + 44 * s, 0.9 * s, INK, 2)
    return frame


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True, help="folder of recorded clips")
    ap.add_argument("--out", default="demo_reel.mp4")
    ap.add_argument("--size", default="1280x720", help="output resolution WxH")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--title", default="EchoGesture")
    ap.add_argument("--subtitle", default="On-device sign language recognition  ·  Team Mudra  ·  Thapar Institute")
    ap.add_argument("--hold", type=float, default=1.5, help="seconds to freeze the verdict after each clip")
    args = ap.parse_args()

    W, H = (int(v) for v in args.size.lower().split("x"))
    clips = sorted(p for p in Path(args.clips).iterdir() if p.suffix.lower() in VIDEO_EXT)
    if not clips:
        sys.exit(f"no video files in {args.clips}")

    clf = Classifier()
    vocab = set(clf.vocab)
    model_label = f"INT8 ONNX  ·  {clf.model_mb:.2f} MB  ·  {len(clf.vocab)} words  ·  open weights"
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))

    def emit(img, seconds):
        for _ in range(int(round(seconds * args.fps))):
            writer.write(img)

    emit(title_card(W, H, args.title, args.subtitle, [
        "camera  ->  MediaPipe hand + pose landmarks  ->  pose-invariant 3D features",
        "->  INT8 temporal conv net (ONNX Runtime, CPU)  ->  word  ->  caption / speech",
        "zero cloud  ·  bystander video never leaves the device",
    ]), 4.0)

    results = []
    overall = Stats(n=10 ** 6)
    for ci, clip in enumerate(clips, 1):
        exp = expected_word(clip, vocab)
        cap = cv2.VideoCapture(str(clip))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if not (1 <= src_fps <= 240):
            src_fps = 25.0
        decoder = StreamDecoder(clf, aspect=W / H, threshold=args.threshold)
        stats = Stats()
        words: list[str] = []
        last = None
        n = 0
        print(f"[{ci}/{len(clips)}] {clip.name}  expected={exp}", flush=True)
        with LandmarkExtractor() as ex:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.resize(frame, (W, H))
                t0 = time.perf_counter()
                raw = ex.process(frame, int(n * 1000 / src_fps))
                lm_ms = (time.perf_counter() - t0) * 1000
                word = decoder.push(raw)
                if word:
                    words.append(word)
                total = (time.perf_counter() - t0) * 1000
                stats.add(lm_ms, decoder.model_ms, total)
                overall.add(lm_ms, decoder.model_ms, total)
                last = draw_hud(frame, raw, decoder, words, stats, model_label, expected=exp,
                                title=f"{args.title}  ·  clip {ci}/{len(clips)}")
                writer.write(last)
                n += 1
        cap.release()
        predicted = words[-1] if words else (decoder.top3[0][0] if decoder.top3 else None)
        if last is not None:
            emit(verdict_overlay(last.copy(), exp, predicted, max(0.5, H / 720.0)), args.hold)
        results.append({"clip": clip.name, "expected": exp, "predicted": predicted, "words": words,
                        "correct": bool(exp and predicted == exp), **stats.summary()})

    scored = [r for r in results if r["expected"]]
    correct = sum(r["correct"] for r in scored)
    acc = 100.0 * correct / len(scored) if scored else float("nan")
    o = overall.summary()
    emit(title_card(W, H, "Results", f"{correct}/{len(scored)} clips correct"
                    + (f"  ·  {acc:.0f}% top-1" if scored else ""), [
        f"end-to-end {o['end_to_end_ms']:.1f} ms / frame  ·  {o['fps']:.1f} fps  ·  classifier {o['model_ms']:.2f} ms",
        f"landmarks {o['landmark_ms']:.1f} ms  ·  model {clf.model_mb:.2f} MB INT8  ·  {len(clf.vocab)} words",
        "fully on-device  ·  open weights + conversion script in the public repo",
    ]), 5.0)
    writer.release()

    Path("results").mkdir(exist_ok=True)
    summary = {"clips": results, "accuracy": acc, "overall": o, "model_mb": clf.model_mb, "classes": len(clf.vocab)}
    Path("results/demo_reel_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}   accuracy {acc:.0f}% on {len(scored)} labelled clips   {o}")
