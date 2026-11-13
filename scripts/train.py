"""Train the classifier on cached landmarks.

    python scripts/train.py --manifest data/manifest.csv --cache data/cache --epochs 200
"""
import argparse
import json
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn

from echogesture import config
from echogesture.augment import train_augment
from echogesture.data import load_split
from echogesture.features import clip_to_input
from echogesture.model import SignTCN


def evaluate(model, x, y, device):
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(x[i:i + 256].to(device)).cpu() for i in range(0, len(x), 256)])
    top5 = logits.topk(min(5, logits.shape[1]), dim=1).indices
    yt = torch.from_numpy(y)
    return (top5[:, 0] == yt).float().mean().item() * 100, (top5 == yt[:, None]).any(1).float().mean().item() * 100


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--train-splits", nargs="*", default=["train"])
    ap.add_argument("--val-splits", nargs="*", default=["val"])
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(config.MODELS_DIR / "checkpoint.pt"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    vocab = config.load_vocab()
    train_clips, train_y, _ = load_split(args.manifest, args.cache, vocab, tuple(args.train_splits))
    val_clips, val_y, _ = load_split(args.manifest, args.cache, vocab, tuple(args.val_splits))
    if not train_clips:
        sys.exit("No training clips found. Did extract_landmarks.py run, and does vocab.txt match the manifest labels?")
    if not val_clips:
        print("WARNING: no validation clips; the last epoch will be saved (no model selection).")
    val_x = torch.from_numpy(np.stack([clip_to_input(c) for c in val_clips])) if val_clips else None
    print(f"{len(vocab)} classes | train {len(train_clips)} | val {len(val_clips)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SignTCN(len(vocab)).to(device)
    print(f"device {device} | {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps = args.epochs * int(np.ceil(len(train_clips) / args.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    best, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        order = rng.permutation(len(train_clips))
        total = 0.0
        for i in range(0, len(order), args.batch):
            idx = order[i:i + args.batch]
            if len(idx) < 2:
                continue
            xb = np.stack([clip_to_input(train_augment(train_clips[j], rng, not args.no_mirror), rng) for j in idx])
            xb, yb = torch.from_numpy(xb).to(device), torch.from_numpy(train_y[idx]).to(device)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        top1, top5 = evaluate(model, val_x, val_y, device) if val_clips else (float(epoch), 0.0)
        history.append({"epoch": epoch, "loss": total / len(order), "val_top1": top1, "val_top5": top5})
        if top1 >= best:
            best = top1
            torch.save({"state_dict": model.state_dict(), "vocab": vocab, "val_top1": top1,
                        "feat_dim": config.FEAT_DIM, "seq_len": config.SEQ_LEN}, args.out)
        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"epoch {epoch:3d} | loss {total / len(order):.3f} | val top1 {top1:5.1f} top5 {top5:5.1f} "
                  f"| best {best:5.1f} | {time.time() - t0:.1f}s", flush=True)
    Path(args.out).with_suffix(".history.json").write_text(json.dumps(history, indent=1))
    print(f"saved {args.out}")
