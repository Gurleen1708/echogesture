"""Build a training manifest from WLASL.

    python scripts/prepare_wlasl.py --wlasl-json D:/wlasl/WLASL_v0.3.json --videos-dir D:/wlasl/videos --top-n 100
    python scripts/prepare_wlasl.py --wlasl-json ... --videos-dir ... --vocab organisers_vocab.txt

Writes data/manifest.csv (clip_id, path, label, split, frame_start, frame_end) and models/vocab.txt.
Use --vocab with the organisers' fixed vocabulary list as soon as you have it.
"""
import argparse
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from echogesture import config

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wlasl-json", required=True)
    ap.add_argument("--videos-dir", required=True)
    ap.add_argument("--top-n", type=int, default=100, help="first N glosses of the WLASL file (WLASL100/300/1000/2000)")
    ap.add_argument("--vocab", help="text file, one word per line; overrides --top-n")
    ap.add_argument("--use-frame-range", action="store_true",
                    help="trim with frame_start/frame_end (only for untrimmed raw downloads)")
    ap.add_argument("--out", default=str(config.ROOT / "data" / "manifest.csv"))
    args = ap.parse_args()

    entries = json.loads(Path(args.wlasl_json).read_text(encoding="utf-8"))
    if args.vocab:
        wanted = config.load_vocab(args.vocab)
    else:
        wanted = [e["gloss"].upper() for e in entries[: args.top_n]]
    wanted_set = set(wanted)
    videos = Path(args.videos_dir)

    rows, missing, per_split = [], 0, {}
    found_words = set()
    for e in entries:
        gloss = e["gloss"].upper()
        if gloss not in wanted_set:
            continue
        for inst in e["instances"]:
            p = videos / f"{inst['video_id']}.mp4"
            if not p.exists():
                missing += 1
                continue
            split = inst.get("split", "train")
            per_split[split] = per_split.get(split, 0) + 1
            found_words.add(gloss)
            rows.append([inst["video_id"], str(p), gloss, split,
                         max(0, inst.get("frame_start", 1) - 1) if args.use_frame_range else 0,
                         (inst.get("frame_end", -1) - 1 if inst.get("frame_end", -1) > 0 else -1)
                         if args.use_frame_range else -1])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "path", "label", "split", "frame_start", "frame_end"])
        w.writerows(rows)
    config.VOCAB_FILE.write_text("\n".join(wanted) + "\n", encoding="utf-8")
    print(f"{len(rows)} clips -> {out}   per split: {per_split}   videos missing on disk: {missing}")
    print(f"{len(wanted)} words -> {config.VOCAB_FILE}")
    absent = [w for w in wanted if w not in found_words]
    if absent:
        print(f"WARNING: {len(absent)} vocabulary words have no training clips: {absent[:15]}")
