"""Run MediaPipe once over every clip in a manifest and cache the raw landmarks (resumable).

    python scripts/extract_landmarks.py --manifest data/manifest.csv --cache data/cache
    python scripts/extract_landmarks.py --manifest data/manifest.csv --cache data/cache_hard --splits test --degrade all
"""
import argparse
import zlib
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from echogesture.data import read_manifest, save_clip
from echogesture.landmarks import extract_clip

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--videos-dir", help="used when the manifest has no path column: <videos-dir>/<clip_id>.mp4")
    ap.add_argument("--splits", nargs="*", help="only these splits (default: all)")
    ap.add_argument("--degrade", choices=["lowlight", "occlusion", "all"],
                    help="darken / occlude the frames first (builds the robustness test set)")
    args = ap.parse_args()

    rows = read_manifest(args.manifest)
    if args.splits:
        rows = [r for r in rows if r["split"] in args.splits]
    cache = Path(args.cache)
    done = failed = 0
    for i, r in enumerate(rows, 1):
        out = cache / f"{r['clip_id']}.npz"
        if out.exists():
            continue
        path = r["path"] or str(Path(args.videos_dir or ".") / f"{r['clip_id']}.mp4")
        try:
            clip = extract_clip(path, degrade_mode=args.degrade, seed=zlib.crc32(r["clip_id"].encode()),
                                frame_start=r["frame_start"], frame_end=r["frame_end"])
            save_clip(out, clip)
            done += 1
        except Exception as exc:  # unreadable / missing video: skip, keep going
            failed += 1
            print(f"  skip {r['clip_id']}: {exc}")
        if i % 50 == 0 or i == len(rows):
            print(f"[{i}/{len(rows)}] new: {done}  failed: {failed}", flush=True)
