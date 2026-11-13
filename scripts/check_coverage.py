"""Show per-word download coverage vs WLASL. No network, no training — just counts.

    python scripts\check_coverage.py --wlasl-json E:\wlasl\WLASL_v0.3.json --videos-dir E:\wlasl\raw_videos
    python scripts\check_coverage.py --wlasl-json E:\wlasl\WLASL_v0.3.json --videos-dir E:\wlasl\raw_videos --top-n 100
"""
import argparse
import json
import sys
from pathlib import Path

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wlasl-json", required=True)
    ap.add_argument("--videos-dir", required=True)
    ap.add_argument("--top-n", type=int, default=100)
    ap.add_argument("--min-clips", type=int, default=6, help="word is 'ready' when it has this many clips")
    args = ap.parse_args()

    entries = json.loads(Path(args.wlasl_json).read_text(encoding="utf-8"))[: args.top_n]
    on_disk = {p.stem for p in Path(args.videos_dir).iterdir() if p.suffix.lower() in (".mp4", ".mkv", ".webm", ".swf", ".mov")}

    ready = partial = empty = 0
    total_clips = 0
    total_needed = sum(len(e["instances"]) for e in entries)
    lines = []
    for e in entries:
        got = sum(1 for inst in e["instances"] if str(inst["video_id"]) in on_disk)
        total_clips += got
        if got >= args.min_clips:
            ready += 1
            status = "OK   "
        elif got > 0:
            partial += 1
            status = "part "
        else:
            empty += 1
            status = "miss "
        lines.append((got, status, e["gloss"], len(e["instances"])))

    lines.sort(key=lambda r: -r[0])
    print(f"\n{args.top_n} WLASL words  |  min clips/word for 'ready' = {args.min_clips}")
    print(f"ready  {ready:3}   partial {partial:3}   empty {empty:3}")
    print(f"clips  {total_clips}/{total_needed}  ({100 * total_clips / total_needed:.0f} %)\n")
    for got, status, gloss, total in lines:
        bar = "#" * min(got, 40)
        print(f"  {status} {gloss:15} {got:3}/{total:3}  {bar}")

    out = Path("results") / "coverage.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(f"{g[2]}\t{g[0]}" for g in lines), encoding="utf-8")
    print(f"\nsaved {out}")
