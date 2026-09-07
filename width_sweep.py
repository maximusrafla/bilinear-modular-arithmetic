"""
How many Fourier frequencies does the model use, and what sets that number?

A one-layer transformer trained on modular addition converges on a handful of
"key frequencies" (Nanda et al.; Bussmann). This bilinear MLP does not: it uses
tens of them. The obvious hypothesis is architectural rather than anything to do
with the task. A transformer routes every token through one shared embedding
matrix of rank <= d_model, so neurons are forced to reuse a small frequency
basis. Here each hidden neuron owns its own length-P embedding (a row of W_l and
of W_r), so nothing couples neurons and nothing rewards reuse.

That predicts the frequency count should track hidden width until it saturates
at the (P-1)/2 available frequencies, rather than sitting at a fixed small
number. This script tests that.

    python width_sweep.py
    python width_sweep.py --widths 8 16 32 --epochs 10000
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


def frequency_stats(model, p: int):
    """Importance-weighted single-frequency purity and frequency count.

    Mirrors neuron_fourier() in analyze.py, condensed to the two numbers this
    sweep needs.
    """
    W_l = model.W_l.weight.detach().cpu().numpy()
    W_r = model.W_r.weight.detach().cpu().numpy()
    drive = np.linalg.norm(model.W_out.weight.detach().cpu().numpy(), axis=0)

    purities, spectra = [], []
    for Wb in (W_l[:, :p], W_l[:, p:], W_r[:, :p], W_r[:, p:]):
        E = Wb - Wb.mean(axis=1, keepdims=True)
        pw = np.abs(np.fft.rfft(E, axis=1)) ** 2
        pw[:, 0] = 0.0
        tot = np.maximum(pw.sum(axis=1), 1e-30)
        w = drive * np.linalg.norm(E, axis=1)
        w = w / max(w.sum(), 1e-30)
        purities.append(float(((pw.max(axis=1) / tot) * w).sum()))
        spectra.append(((pw / tot[:, None]) * w[:, None]).sum(axis=0))

    agg = np.mean(spectra, axis=0)
    agg = agg / max(agg.sum(), 1e-30)
    cum = np.cumsum(agg[np.argsort(-agg)])
    return (float(np.mean(purities)),
            int((cum < 0.90).sum() + 1),
            int((cum < 0.99).sum() + 1))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--widths", type=int, nargs="+",
                    default=[8, 16, 32, 48, 64, 128, 256, 512])
    ap.add_argument("--p", type=int, default=P_DEFAULT)
    ap.add_argument("--epochs", type=int, default=25000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1.0)
    ap.add_argument("--train-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device(args.device)
    p = args.p

    X, y, _, _ = all_pairs(p)
    tr, te = split_indices(len(y), args.train_frac, 0)
    X, y = X.to(device), y.to(device)
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    criterion = nn.CrossEntropyLoss()

    rows = []
    print(f"{'width':>6} {'params':>9} {'train':>7} {'test':>7} "
          f"{'purity':>7} {'freqs90':>8} {'freqs99':>8}   time")
    for H in args.widths:
        torch.manual_seed(args.seed)
        model = build_model("bilinear", p, H).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay, betas=(0.9, 0.98))
        t0 = time.time()
        for _ in range(args.epochs):
            opt.zero_grad(set_to_none=True)
            criterion(model(Xtr), ytr).backward()
            opt.step()

        with torch.no_grad():
            tra = (model(Xtr).argmax(1) == ytr).float().mean().item()
            tea = (model(Xte).argmax(1) == yte).float().mean().item()
        purity, f90, f99 = frequency_stats(model, p)
        n = sum(q.numel() for q in model.parameters())
        rows.append({"width": H, "params": n, "train_acc": tra, "test_acc": tea,
                     "purity": purity, "n_freqs_90": f90, "n_freqs_99": f99})
        print(f"{H:>6} {n:>9,} {tra:>7.3f} {tea:>7.3f} {purity:>7.3f} "
              f"{f90:>8} {f99:>8}   {time.time() - t0:.0f}s", flush=True)

    with open(os.path.join(RESULTS_DIR, "width_sweep.json"), "w") as fh:
        json.dump({"args": vars(args), "rows": rows}, fh, indent=2)

    plot(rows, p)


def plot(rows, p):
    W = [r["width"] for r in rows]
    f90 = [r["n_freqs_90"] for r in rows]
    pur = [r["purity"] for r in rows]
    te = [r["test_acc"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    ax.plot(W, f90, "o-", lw=1.8, color="C0", label="frequencies carrying 90% of power")
    ax.axhline((p - 1) // 2, color="k", ls=":", lw=1,
               label=f"all {(p - 1) // 2} available frequencies")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("hidden width $H$")
    ax.set_ylabel("number of frequencies used")
    ax.set_title("Frequency count tracks width, then saturates")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    ax.plot(W, pur, "o-", lw=1.8, color="C2", label="single-frequency purity")
    ax.plot(W, te, "s--", lw=1.6, color="C1", label="test accuracy")
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("hidden width $H$")
    ax.set_title("Neurons stay individually monochromatic at every width")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "width_sweep.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
