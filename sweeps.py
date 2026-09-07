"""
Seed and hyperparameter sweeps.

Two claims elsewhere in this repo cannot be supported by a single run:

1. the headline weight-space numbers are not a single-seed accident, and
2. grokking here is not a knife-edge hyperparameter setting.

Each is cheap to check and expensive to merely assert, so this script runs both
and writes the raw results to results/seed_sweep.json and
results/hparam_sweep.json. The README quotes those files.

The seed sweep varies BOTH the weight initialisation and the train/test split,
so each row is an independent experiment rather than a re-initialisation on a
fixed split.

    python sweeps.py                 # both, ~15 min on a consumer GPU
    python sweeps.py --what seeds
    python sweeps.py --what hparams
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

from analyze import ablation_curve, interaction_structure, neuron_fourier
from bilinear import (P_DEFAULT, all_pairs, build_model, interaction_tensor,
                      split_indices)

RESULTS_DIR = "results"


def train_one(p, hidden_dim, lr, wd, train_frac, seed, split_seed, epochs,
              device, eval_every=200):
    """Train one model; return it plus test accuracy and grokking onset."""
    X, y, _, _ = all_pairs(p)
    tr, te = split_indices(len(y), train_frac, split_seed)
    X, y = X.to(device), y.to(device)
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

    torch.manual_seed(seed)
    model = build_model("bilinear", p, hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd,
                            betas=(0.9, 0.98))
    criterion = nn.CrossEntropyLoss()

    grok_epoch = None
    for ep in range(1, epochs + 1):
        opt.zero_grad(set_to_none=True)
        criterion(model(Xtr), ytr).backward()
        opt.step()
        if grok_epoch is None and ep % eval_every == 0:
            with torch.no_grad():
                if (model(Xte).argmax(1) == yte).float().mean().item() >= 0.9:
                    grok_epoch = ep

    with torch.no_grad():
        tra = (model(Xtr).argmax(1) == ytr).float().mean().item()
        tea = (model(Xte).argmax(1) == yte).float().mean().item()
    return model.cpu().eval(), tra, tea, grok_epoch


def seed_sweep(args, device):
    rows = []
    print(f"{'seed':>5} {'train':>7} {'test':>7} {'grok@':>7} {'purity':>7} "
          f"{'ab':>5} {'f90':>5} {'R^2':>7} {'suff':>5} {'del':>5}   time")
    for seed in args.seeds:
        t0 = time.time()
        model, tra, tea, grok = train_one(
            args.p, args.hidden_dim, args.lr, args.weight_decay,
            args.train_frac, seed, seed, args.epochs, device)

        B = interaction_tensor(model)
        nf = neuron_fourier(model, args.p)
        inter = interaction_structure(B, args.p)
        abl = ablation_curve(model, args.p, nf["_order"])

        rows.append({
            "seed": seed, "train_acc": tra, "test_acc": tea, "grok_epoch": grok,
            "purity": nf["purity_weighted"],
            "ab_frequency_match": nf["ab_frequency_match"],
            "n_freqs_for_90pct": nf["n_freqs_for_90pct"],
            "cross_block_r2": inter["cross_block_is_sum_of_a_and_b_r2"],
            "n_freqs_sufficient": abl["n_freqs_sufficient"],
            "n_freqs_deletable": abl["n_freqs_deletable"],
        })
        r = rows[-1]
        print(f"{seed:>5} {tra:>7.3f} {tea:>7.3f} {str(grok):>7} "
              f"{r['purity']:>7.3f} {r['ab_frequency_match']:>5.3f} "
              f"{r['n_freqs_for_90pct']:>5} {r['cross_block_r2']:>7.3f} "
              f"{str(r['n_freqs_sufficient']):>5} {r['n_freqs_deletable']:>5}   "
              f"{time.time() - t0:.0f}s", flush=True)

    def rng(key):
        v = [r[key] for r in rows if r[key] is not None]
        return {"min": min(v), "max": max(v)} if v else None

    summary = {k: rng(k) for k in
               ("test_acc", "grok_epoch", "purity", "ab_frequency_match",
                "n_freqs_for_90pct", "cross_block_r2", "n_freqs_sufficient",
                "n_freqs_deletable")}
    out = {"args": vars(args), "rows": rows, "range": summary}
    with open(os.path.join(RESULTS_DIR, "seed_sweep.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nranges across seeds:")
    for k, v in summary.items():
        print(f"  {k}: {v['min']} .. {v['max']}" if v else f"  {k}: n/a")
    return rows


def hparam_sweep(args, device):
    rows = []
    print(f"{'lr':>8} {'wd':>7} {'train':>7} {'test':>7} {'grok@':>7}   time")
    for lr in args.lrs:
        for wd in args.wds:
            t0 = time.time()
            _, tra, tea, grok = train_one(
                args.p, args.hidden_dim, lr, wd, args.train_frac,
                args.seed, 0, args.epochs, device)
            rows.append({"lr": lr, "weight_decay": wd, "train_acc": tra,
                         "test_acc": tea, "grok_epoch": grok})
            print(f"{lr:>8.0e} {wd:>7.1f} {tra:>7.3f} {tea:>7.3f} "
                  f"{str(grok):>7}   {time.time() - t0:.0f}s", flush=True)

    with open(os.path.join(RESULTS_DIR, "hparam_sweep.json"), "w") as fh:
        json.dump({"args": vars(args), "rows": rows}, fh, indent=2)
    return rows


def plot(seed_rows, hp_rows, args):
    n = (1 if seed_rows else 0) + (1 if hp_rows else 0)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 4.4), squeeze=False)
    axes = axes[0]
    i = 0

    if seed_rows:
        ax = axes[i]
        i += 1
        keys = [("purity", "single-freq purity"),
                ("ab_frequency_match", "a/b freq agreement"),
                ("test_acc", "test accuracy"),
                ("cross_block_r2", "$B_{ab}$ $R^2$ on $(a+b)$")]
        x = np.arange(len(keys))
        for j, r in enumerate(seed_rows):
            ax.plot(x, [r[k] for k, _ in keys], "o-", lw=1.4, alpha=0.8,
                    label=f"seed {r['seed']}")
        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, lab in keys], rotation=20, ha="right",
                           fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("value")
        ax.set_title("Headline metrics across independent runs")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    if hp_rows:
        ax = axes[i]
        lrs = sorted({r["lr"] for r in hp_rows})
        wds = sorted({r["weight_decay"] for r in hp_rows})
        M = np.full((len(lrs), len(wds)), np.nan)
        for r in hp_rows:
            M[lrs.index(r["lr"]), wds.index(r["weight_decay"])] = r["test_acc"]
        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(wds)))
        ax.set_xticklabels([f"{w:g}" for w in wds])
        ax.set_yticks(range(len(lrs)))
        ax.set_yticklabels([f"{l:.0e}" for l in lrs])
        ax.set_xlabel("weight decay")
        ax.set_ylabel("learning rate")
        ax.set_title("Final test accuracy")
        for a in range(len(lrs)):
            for b in range(len(wds)):
                ax.text(b, a, f"{M[a, b]:.3f}", ha="center", va="center",
                        fontsize=8,
                        color="w" if M[a, b] < 0.6 else "k")
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "sweeps.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--what", default="both", choices=["both", "seeds", "hparams"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 42])
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 3e-3])
    ap.add_argument("--wds", type=float, nargs="+", default=[0.1, 0.3, 1.0, 3.0, 10.0])
    ap.add_argument("--p", type=int, default=P_DEFAULT)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--train-frac", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=25000)
    ap.add_argument("--lr", type=float, default=1e-3, help="lr for the seed sweep")
    ap.add_argument("--weight-decay", type=float, default=1.0,
                    help="weight decay for the seed sweep")
    ap.add_argument("--seed", type=int, default=42, help="seed for the hparam grid")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device(args.device)

    seed_rows = hp_rows = None
    if args.what in ("both", "seeds"):
        print("=== seed sweep (init + split both vary) ===")
        seed_rows = seed_sweep(args, device)
    if args.what in ("both", "hparams"):
        print("\n=== hyperparameter grid ===")
        hp_rows = hparam_sweep(args, device)

    plot(seed_rows, hp_rows, args)


if __name__ == "__main__":
    main()
