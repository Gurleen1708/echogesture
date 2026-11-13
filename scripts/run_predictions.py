"""Run predict() over the test manifest and write predictions.jsonl (one JSON object per clip).

    python scripts/run_predictions.py --manifest test_manifest.csv --videos-dir test_clips --out predictions.jsonl
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from echogesture.data import read_manifest
from echogesture.predictor import get_predictor, to_jsonl_line

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", help="CSV with a clip id column (and optionally path / label)")
    ap.add_argument("--videos-dir", help="folder with <clip_id>.mp4 (required if the manifest has no path column)")
    ap.add_argument("--splits", nargs="*")
    ap.add_argument("--out", default="predictions.jsonl")
    args = ap.parse_args()

    if args.manifest:
        rows = read_manifest(args.manifest)
        if args.splits:
            rows = [r for r in rows if r["split"] in args.splits]
    else:
        rows = [{"clip_id": p.stem, "path": str(p), "label": None}
                for p in sorted(Path(args.videos_dir).iterdir()) if p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv", ".webm")]

    predictor = get_predictor()
    vocab = predictor.classifier.vocab
    hits = hits5 = scored = failed = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows, 1):
            path = r["path"]
            if not path or not Path(path).exists():
                cands = list(Path(args.videos_dir or ".").glob(r["clip_id"] + ".*"))
                path = str(cands[0]) if cands else None
            try:
                res = predictor.predict(path)
            except Exception as exc:  # every clip id must still be present in the file
                failed += 1
                print(f"  FAILED {r['clip_id']}: {exc}")
                res = {"clip_id": r["clip_id"], "prediction": vocab[0], "top5": vocab[:5],
                       "confidence": 0.0, "latency_ms": 0.0}
            res["clip_id"] = r["clip_id"]  # must match the manifest exactly
            f.write(to_jsonl_line(res) + "\n")
            if r.get("label"):
                scored += 1
                hits += res["prediction"] == r["label"]
                hits5 += r["label"] in res["top5"]
            if i % 25 == 0 or i == len(rows):
                print(f"[{i}/{len(rows)}]", flush=True)
    print(f"wrote {len(rows)} lines -> {args.out}  (failed clips: {failed})")
    if scored:
        print(f"top-1 {100 * hits / scored:.2f}%   top-5 {100 * hits5 / scored:.2f}%   on {scored} labelled clips")
