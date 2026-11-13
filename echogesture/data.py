"""Manifest + landmark-cache loading."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ID_COLS = ("clip_id", "video_id", "id", "clip")
PATH_COLS = ("path", "video_path", "video", "file", "filename")
LABEL_COLS = ("label", "gloss", "word", "class")


def _pick(row: dict, names) -> str | None:
    low = {k.lower().strip(): v for k, v in row.items() if k}
    for n in names:
        if low.get(n):
            return low[n].strip()
    return None


def read_manifest(path: str) -> list[dict]:
    """Flexible CSV reader: needs a clip id column; path / label / split are optional."""
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            cid = _pick(row, ID_COLS)
            vpath = _pick(row, PATH_COLS)
            if cid is None and vpath:
                cid = Path(vpath).stem
            if cid is None:
                continue
            label = _pick(row, LABEL_COLS)
            rows.append({
                "clip_id": cid,
                "path": vpath,
                "label": label.upper() if label else None,
                "split": (_pick(row, ("split",)) or "").lower(),
                "frame_start": int(_pick(row, ("frame_start",)) or 0),
                "frame_end": int(_pick(row, ("frame_end",)) or -1),
            })
    return rows


def save_clip(path: Path, clip: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **clip)


def load_clip(path: Path) -> dict:
    with np.load(path) as z:
        clip = {k: z[k] for k in z.files}
    clip["aspect"] = np.float32(clip["aspect"])
    return clip


def load_split(manifest: str, cache_dir: str, vocab: list[str], splits: tuple[str, ...]):
    """Return (clips, labels, clip_ids) for rows of the given splits that have a cached clip."""
    index = {w: i for i, w in enumerate(vocab)}
    clips, labels, ids = [], [], []
    for row in read_manifest(manifest):
        if row["split"] not in splits or row["label"] not in index:
            continue
        p = Path(cache_dir) / f"{row['clip_id']}.npz"
        if not p.exists():
            continue
        clips.append(load_clip(p))
        labels.append(index[row["label"]])
        ids.append(row["clip_id"])
    return clips, np.array(labels, dtype=np.int64), ids
