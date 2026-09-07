"""
Train a bilinear MLP on modular addition with a real train/test split.

    f(a, b) = (a + b) mod P,  P = 113

The complete input space is 113^2 = 12,769 pairs, so it is enumerated exactly
and split. Both train AND test metrics are recorded every --log-every epochs
(default 100), because the train/test gap is the whole point of this task:
modular addition is the canonical grokking benchmark, and train accuracy alone
says nothing.

Examples
--------
    python train.py --name grok     --weight-decay 1.0
    python train.py --name memorize --weight-decay 0.0
    python train.py --name relu --arch relu --hidden-dim 427
"""

from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from bilinear import P_DEFAULT, all_pairs, build_model, split_indices

RESULTS_DIR = "results"
CKPT_DIR = "checkpoints"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="grok", help="run name; used for output filenames")
    ap.add_argument("--arch", default="bilinear", choices=["bilinear", "relu"])
    ap.add_argument("--p", type=int, default=P_DEFAULT, help="modulus (should be prime)")
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--train-frac", type=float, default=0.3,
                    help="fraction of the 12,769 pairs used for training")
    ap.add_argument("--epochs", type=int, default=25000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1.0,
                    help="AdamW decoupled weight decay")
    ap.add_argument("--init-scale", type=float, default=1.0)
    ap.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.98))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


@torch.no_grad()
def evaluate(model, X, y, criterion):
    logits = model(X)
    loss = criterion(logits, y).item()
    acc = (logits.argmax(dim=1) == y).float().mean().item()
    return loss, acc


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    device = torch.device(args.device)
    X, y, _, _ = all_pairs(args.p)
    tr_idx, te_idx = split_indices(len(y), args.train_frac, args.split_seed)

    X, y = X.to(device), y.to(device)
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xte, yte = X[te_idx], y[te_idx]

    model = build_model(args.arch, args.p, args.hidden_dim, args.init_scale).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, betas=tuple(args.betas))
    criterion = nn.CrossEntropyLoss()

    print(f"run={args.name} arch={args.arch} device={device}")
    print(f"input space: {args.p}^2 = {len(y)} pairs "
          f"| train {len(ytr)} ({args.train_frac:.0%}) | test {len(yte)}")
    print(f"model: {2*args.p} -> {args.hidden_dim} -> {args.p} "
          f"| {model.n_params():,} parameters")
    print(f"epochs={args.epochs} lr={args.lr} weight_decay={args.weight_decay}")
    print("-" * 72)

    hist = {"epoch": [], "train_loss": [], "test_loss": [],
            "train_acc": [], "test_acc": []}
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = criterion(model(Xtr), ytr)
        loss.backward()
        opt.step()

        if epoch % args.log_every == 0 or epoch == 1:
            model.eval()
            trl, tra = evaluate(model, Xtr, ytr, criterion)
            tel, tea = evaluate(model, Xte, yte, criterion)
            hist["epoch"].append(epoch)
            hist["train_loss"].append(trl)
            hist["test_loss"].append(tel)
            hist["train_acc"].append(tra)
            hist["test_acc"].append(tea)
            if epoch % (args.log_every * 50) == 0 or epoch == 1:
                print(f"epoch {epoch:>7} | train loss {trl:.4f} acc {tra:.4f} "
                      f"| test loss {tel:.4f} acc {tea:.4f}")

    elapsed = time.time() - t0
    model.eval()
    trl, tra = evaluate(model, Xtr, ytr, criterion)
    tel, tea = evaluate(model, Xte, yte, criterion)
    print("-" * 72)
    print(f"final | train acc {tra:.4f} | test acc {tea:.4f} | {elapsed:.1f}s")

    # epoch at which test accuracy first crossed 90%, if ever (grokking onset)
    grok_epoch = None
    for e, a in zip(hist["epoch"], hist["test_acc"]):
        if a >= 0.9:
            grok_epoch = e
            break
    if grok_epoch is not None:
        print(f"test accuracy first reached 90% at epoch {grok_epoch}")
    else:
        print("test accuracy never reached 90% (no generalization)")

    ckpt_path = os.path.join(CKPT_DIR, f"{args.name}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "arch": args.arch,
        "p": args.p,
        "hidden_dim": args.hidden_dim,
        "train_idx": tr_idx,
        "test_idx": te_idx,
        "history": hist,
        "final": {"train_loss": trl, "train_acc": tra,
                  "test_loss": tel, "test_acc": tea,
                  "grok_epoch": grok_epoch, "n_params": model.n_params()},
    }, ckpt_path)
    print(f"saved: {ckpt_path}")

    with open(os.path.join(RESULTS_DIR, f"{args.name}_history.json"), "w") as fh:
        json.dump({"args": vars(args), "history": hist,
                   "final": {"train_acc": tra, "test_acc": tea,
                             "train_loss": trl, "test_loss": tel,
                             "grok_epoch": grok_epoch,
                             "n_params": model.n_params()}}, fh, indent=2)

    plot_curves(hist, args, grok_epoch)
    return model


def plot_curves(hist, args, grok_epoch):
    ep = np.array(hist["epoch"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

    ax1.plot(ep, hist["train_loss"], label="train", lw=1.4)
    ax1.plot(ep, hist["test_loss"], label="test", lw=1.4)
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("epoch (log)"); ax1.set_ylabel("cross-entropy (log)")
    ax1.set_title("Loss"); ax1.legend(); ax1.grid(alpha=0.3, which="both")

    ax2.plot(ep, hist["train_acc"], label="train", lw=1.4)
    ax2.plot(ep, hist["test_acc"], label="test", lw=1.4)
    ax2.set_xscale("log"); ax2.set_ylim(-0.02, 1.02)
    ax2.set_xlabel("epoch (log)"); ax2.set_ylabel("accuracy")
    ax2.set_title("Accuracy"); ax2.legend(); ax2.grid(alpha=0.3, which="both")
    if grok_epoch:
        ax2.axvline(grok_epoch, color="k", ls=":", lw=1)
        ax2.annotate(f"test > 90%\n@ epoch {grok_epoch}", (grok_epoch, 0.5),
                     xytext=(-8, 0), textcoords="offset points",
                     fontsize=8, va="center", ha="right")

    fig.suptitle(f"{args.name}: {args.arch}, train_frac={args.train_frac}, "
                 f"wd={args.weight_decay}", fontsize=11)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f"{args.name}_training_curves.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    train(parse_args())
