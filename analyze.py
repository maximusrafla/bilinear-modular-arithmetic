"""
Weight-based analysis of a trained bilinear MLP on modular addition.

Everything here is computed from weights. No activations, no probes, no
sampling -- which is the property that makes bilinear layers worth studying
(Pearce et al., arXiv:2410.08417).

Structure of the model as a quadratic form
-----------------------------------------
    f_k(x) = x^T B_k x,   B_k = sym(sum_h W_out[k,h] W_l[h,:]^T W_r[h,:])

With x = [onehot(a); onehot(b)], writing B_k in blocks

    B_k = [[ B_aa , B_ab ],
           [ B_ab^T, B_bb ]]

gives    f_k(a,b) = B_aa[a,a] + B_bb[b,b] + 2 B_ab[a,b].

Two consequences drive this file:

* Only the DIAGONAL of B_aa and B_bb is ever probed by a one-hot input. Their
  off-diagonals are unconstrained by the task, and in practice hold ~99% of
  those blocks' energy. Eigendecomposing the full 2P x 2P B_k therefore mixes
  the circuit with a large unconstrained subspace, and any rank statistic taken
  there is mostly measuring noise. The circuit lives in B_ab.

* B_ab[a,b] is the only term coupling a and b, so it is where "+ mod P" has to
  be implemented. For modular addition it must depend on a and b only through
  (a + b) mod P -- i.e. it must be a circulant.

What the old version of this repo got wrong
-------------------------------------------
* sigma_1/sigma_2 ~ 1.1 was read as "not low-rank". A Fourier circuit puts each
  frequency into a degenerate cos/sin pair, so a ratio near 1 is what clean
  low-rank structure predicts. It cannot distinguish the two hypotheses. Worse,
  the memorizing control here gives 1.26 and the generalizing model 1.06 -- the
  statistic runs backwards.
* Singular vectors were plotted across all 226 input positions at once, which
  straddles the a/b boundary at index 113 and manufactures a discontinuity that
  was then read as "periodic structure".
* Mode-1 vs mode-2 similarity was read as commutativity. Both modes index the
  same [a;b] vector; they are W_l's input space vs W_r's, so the comparison says
  nothing about swapping a and b. The real test is the relative size of
  f(a,b) - f(b,a), done in block_symmetry() below.

Usage:
    python analyze.py --name grok
    python analyze.py --name grok --compare memorize
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from bilinear import (accuracy, all_pairs, build_model, eigendecompose,
                      frequency_ablate, interaction_tensor,
                      participation_ratio)

RESULTS_DIR = "results"
CKPT_DIR = "checkpoints"


def load_run(name: str):
    path = os.path.join(CKPT_DIR, f"{name}.pt")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["arch"], ckpt["p"], ckpt["hidden_dim"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


# --------------------------------------------------------------------------- #
# 1. Neuron-level Fourier structure  (the primary result)
# --------------------------------------------------------------------------- #

def neuron_fourier(model, p: int) -> dict:
    """Is each hidden neuron a single Fourier frequency, and which ones?

    Neuron h reads the input through four length-P embeddings: rows of
    W_l[:, :P], W_l[:, P:], W_r[:, :P], W_r[:, P:]. The Fourier construction for
    modular addition predicts each is a pure sinusoid, so DFT each row and ask
    what share of its power sits at its own dominant frequency ("purity").

    Neurons are weighted by how much they actually drive the output
    (||W_out[:, h]|| times the embedding norm) so that dead neurons -- of which
    weight decay leaves many -- do not dilute the statistic.
    """
    W_l = model.W_l.weight.detach().cpu().numpy()
    W_r = model.W_r.weight.detach().cpu().numpy()
    W_out = model.W_out.weight.detach().cpu().numpy()
    drive = np.linalg.norm(W_out, axis=0)  # (H,)

    blocks = {"W_l/a": W_l[:, :p], "W_l/b": W_l[:, p:],
              "W_r/a": W_r[:, :p], "W_r/b": W_r[:, p:]}

    per_block, spectra, weights, domfreqs = {}, [], [], []
    for tag, Wb in blocks.items():
        E = Wb - Wb.mean(axis=1, keepdims=True)     # drop the DC offset
        pw = np.abs(np.fft.rfft(E, axis=1)) ** 2
        pw[:, 0] = 0.0
        tot = np.maximum(pw.sum(axis=1), 1e-30)

        purity = pw.max(axis=1) / tot               # (H,)
        w = drive * np.linalg.norm(E, axis=1)
        w = w / max(w.sum(), 1e-30)
        agg = ((pw / tot[:, None]) * w[:, None]).sum(axis=0)

        per_block[tag] = float((purity * w).sum())
        spectra.append(agg)
        weights.append(w)
        domfreqs.append(pw.argmax(axis=1))

    agg = np.mean(spectra, axis=0)
    agg = agg / max(agg.sum(), 1e-30)
    order = np.argsort(-agg)
    cum = np.cumsum(agg[order])

    # do the a-embedding and b-embedding of a neuron use the SAME frequency?
    # (required for a (a+b) circuit; a (a-b) or unrelated pair would not)
    match_l = (domfreqs[0] == domfreqs[1])
    match_r = (domfreqs[2] == domfreqs[3])
    w_l, w_r = weights[0], weights[2]

    return {
        "purity_weighted": float(np.mean(list(per_block.values()))),
        "purity_by_block": per_block,
        "ab_frequency_match": float(0.5 * ((match_l * w_l).sum() + (match_r * w_r).sum())),
        "n_freqs_for_90pct": int((cum < 0.90).sum() + 1),
        "n_freqs_for_99pct": int((cum < 0.99).sum() + 1),
        "n_freqs_available": int(p // 2),
        "top_frequencies": [int(w) for w in order[:10]],
        "top_frequency_power": [float(agg[w]) for w in order[:10]],
        "_agg": agg, "_order": order, "_w": weights[0], "_blocks": blocks,
    }


# --------------------------------------------------------------------------- #
# 2. The interaction block: is it "+ mod P"?
# --------------------------------------------------------------------------- #

def _anova(M: np.ndarray):
    """M[a,b] -> mu + alpha(a) + beta(b) + gamma(a,b), gamma with zero margins."""
    mu = M.mean()
    al = M.mean(axis=1) - mu
    be = M.mean(axis=0) - mu
    return mu, al, be, M - mu - al[:, None] - be[None, :]


def _sum_r2(G: np.ndarray, p: int):
    """R^2 of the best approximation to G[a,b] that depends only on (a+b) mod P."""
    idx = (np.arange(p)[:, None] + np.arange(p)[None, :]) % p
    m = np.bincount(idx.ravel(), weights=G.ravel(), minlength=p) / p
    resid = G - m[idx]
    return 1.0 - (resid ** 2).sum() / max((G ** 2).sum(), 1e-300), m


def interaction_structure(B: np.ndarray, p: int) -> dict:
    """Decompose the cross block B_ab and test it for (a+b) dependence."""
    B_aa, B_ab, B_bb = B[:, :p, :p], B[:, :p, p:], B[:, p:, p:]
    tot = (B ** 2).sum()

    # how much of the diagonal blocks a one-hot input can even see
    d_aa = np.einsum("kii->ki", B_aa)
    d_bb = np.einsum("kii->ki", B_bb)
    probed = (d_aa ** 2).sum() + (d_bb ** 2).sum()
    diag_energy = (B_aa ** 2).sum() + (B_bb ** 2).sum()

    num = den = 0.0
    sep = joint = 0.0
    spec = np.zeros(p // 2 + 1)
    sum_fns = np.zeros((B.shape[0], p))
    for k in range(B.shape[0]):
        _, al, be, ga = _anova(B_ab[k])
        sep += (al ** 2).sum() * p + (be ** 2).sum() * p
        joint += (ga ** 2).sum()
        r2, m = _sum_r2(ga, p)
        num += r2 * (ga ** 2).sum()
        den += (ga ** 2).sum()
        sum_fns[k] = m
        f = np.abs(np.fft.rfft(m)) ** 2
        f[0] = 0.0
        spec += f
    spec = spec / max(spec.sum(), 1e-30)

    return {
        "cross_block_energy_fraction": float(2 * (B_ab ** 2).sum() / tot),
        "diag_block_energy_fraction": float(diag_energy / tot),
        "diag_block_energy_actually_probed": float(probed / max(diag_energy, 1e-300)),
        "cross_block_separable_share": float(sep / max(sep + joint, 1e-300)),
        "cross_block_is_sum_of_a_and_b_r2": float(num / max(den, 1e-300)),
        "_sum_fns": sum_fns, "_B_ab": B_ab,
    }


def block_symmetry(B: np.ndarray, p: int, L: np.ndarray) -> dict:
    """Does the model actually treat a and b symmetrically?

    Measured on the function, not on the raw matrices. Comparing B_aa to B_bb
    directly -- the tempting weight-space test -- is dominated by the
    off-diagonal entries a one-hot input never reads, and reports large
    asymmetry for a model that is in fact perfectly commutative. Since
    f_k(a,b) = B_aa[a,a] + B_bb[b,b] + 2 B_ab[a,b], the honest test is the
    relative size of f(a,b) - f(b,a).
    """
    B_ab = B[:, :p, p:]
    centred = L - L.mean(axis=(0, 1), keepdims=True)
    asym = L - np.transpose(L, (1, 0, 2))

    # predictions: does the model give the same answer for (a,b) and (b,a)?
    pred = L.argmax(axis=2)
    agree = float((pred == pred.T).mean())

    # the coupling term on its own, normalised by its own size (not by ||B_k||,
    # which is dominated by the unprobed off-diagonals of B_aa and B_bb)
    off = np.linalg.norm((B_ab - np.transpose(B_ab, (0, 2, 1))).reshape(p, -1), axis=1)
    den = np.maximum(np.linalg.norm(B_ab.reshape(p, -1), axis=1), 1e-300)

    return {
        "prediction_commutativity": agree,
        "logit_asymmetry": float(
            np.linalg.norm(asym) / max(np.linalg.norm(centred), 1e-300)),
        "cross_block_asymmetry": float((off / den).mean()),
    }


def spectrum_stats(vals: np.ndarray, B: np.ndarray, p: int) -> dict:
    """Rank statistics, computed on the full B_k and on the cross block alone."""
    pr_full = participation_ratio(vals, axis=1)
    energy = vals ** 2
    cum = np.cumsum(energy, axis=1) / np.maximum(energy.sum(axis=1, keepdims=True), 1e-300)

    s = np.linalg.svd(B[:, :p, p:], compute_uv=False)
    pr_cross = participation_ratio(s, axis=1)
    cum_c = np.cumsum(s ** 2, axis=1) / np.maximum((s ** 2).sum(axis=1, keepdims=True), 1e-300)

    return {
        "full_B_participation_ratio": float(pr_full.mean()),
        "full_B_dim": int(vals.shape[1]),
        "cross_block_participation_ratio": float(pr_cross.mean()),
        "cross_block_dim": int(p),
        "cross_block_energy_in_top_10": float(cum_c[:, 9].mean()),
        # the statistic the old README relied on, reported only for contrast
        "lambda1_over_lambda2": float(
            (np.abs(vals[:, 0]) / np.maximum(np.abs(vals[:, 1]), 1e-300)).mean()),
        "_cum": cum, "_cum_cross": cum_c, "_pr": pr_full, "_pr_cross": pr_cross,
    }


# --------------------------------------------------------------------------- #
# 3. Causal test: frequency surgery on the weights
# --------------------------------------------------------------------------- #

def ablation_curve(model, p: int, order, ks=(1, 2, 3, 5, 8, 12, 20, 30, 38, 45, 50, 53, 56)) -> dict:
    """Keep only the top-K frequencies in the embeddings, then only delete them.

    If the model computes with a set of frequencies, keeping that set should
    preserve accuracy and deleting it should destroy accuracy. Both directions
    are needed: keeping-only can succeed trivially if the removed part is small.
    """
    X, y, _, _ = all_pairs(p)
    base = accuracy(model, X, y)
    keep_curve, drop_curve = [], []
    for K in ks:
        if K > len(order):
            break
        sel = [int(w) for w in order[:K]]
        keep_curve.append((int(K), accuracy(frequency_ablate(model, sel, p), X, y)))
        drop_curve.append((int(K), accuracy(frequency_ablate(model, sel, p, drop=True), X, y)))

    thresh = 0.99 * base
    sufficient = next((k for k, a in keep_curve if a >= thresh), None)
    survives = max([k for k, a in drop_curve if a >= thresh], default=0)
    return {"baseline_accuracy": float(base),
            "keep_top_k": keep_curve, "drop_top_k": drop_curve,
            # smallest frequency set that reproduces the model on its own
            "n_freqs_sufficient": sufficient,
            # largest number of the most-used frequencies that can be deleted
            # with no loss -- a direct measure of redundancy
            "n_freqs_deletable": survives}


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def fig_neurons(nf, p, name, n_show=3):
    fig = plt.figure(figsize=(14, 7.5))
    gs = fig.add_gridspec(3, 3, width_ratios=[1.15, 1.15, 1.5])

    blocks = list(nf["_blocks"].items())
    Wa = blocks[0][1]
    order_n = np.argsort(-nf["_w"])[:n_show]
    for r, h in enumerate(order_n):
        e = Wa[h] - Wa[h].mean()
        ax = fig.add_subplot(gs[r, 0])
        ax.plot(e, lw=1.3, color="C0")
        ax.set_ylabel(f"neuron {h}", fontsize=8)
        ax.grid(alpha=0.3)
        if r == 0:
            ax.set_title("$a$-embedding $W_l[h,\\;0{:}P]$", fontsize=10)
        if r == n_show - 1:
            ax.set_xlabel("$a$")

        ax = fig.add_subplot(gs[r, 1])
        pw = np.abs(np.fft.rfft(e)) ** 2
        pw[0] = 0
        ax.bar(np.arange(len(pw)), pw, color="C3", width=1.0)
        ax.grid(alpha=0.3, axis="y")
        if r == 0:
            ax.set_title("its DFT (single spike = one frequency)", fontsize=10)
        if r == n_show - 1:
            ax.set_xlabel("frequency $w$")

    ax = fig.add_subplot(gs[:, 2])
    agg = nf["_agg"]
    ax.bar(np.arange(1, len(agg)), agg[1:], color="C0")
    ax.set_xlabel("frequency $w$")
    ax.set_ylabel("share of embedding power")
    ax.set_title(f"Frequencies used across all neurons\n"
                 f"purity={nf['purity_weighted']:.3f}, "
                 f"{nf['n_freqs_for_90pct']} of {nf['n_freqs_available']} "
                 f"frequencies carry 90%", fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(f"{name}: neuron-level Fourier structure", fontsize=13)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f"{name}_neuron_frequencies.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


def fig_cross_block(inter, p, name, ks=(0, 1, 37)):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    B_ab = inter["_B_ab"]
    for j, k in enumerate(ks):
        M = B_ab[k]
        vmax = np.abs(M).max()
        ax = axes[0, j]
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_title(f"$B_{{ab}}$ for output $k={k}$", fontsize=10)
        ax.set_xlabel("$b$")
        ax.set_ylabel("$a$")
        fig.colorbar(im, ax=ax, fraction=0.046)

        ax = axes[1, j]
        ax.plot(inter["_sum_fns"][k], lw=1.3, color="C2")
        ax.axvline(k, color="k", ls=":", lw=1)
        ax.set_xlabel("$s = (a+b) \\; \\mathrm{mod} \\; P$")
        ax.set_title(f"collapsed onto $s$; dotted line at $s=k$", fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"{name}: the interaction block. Anti-diagonal stripes mean 'depends on $a+b$'. "
        f"$R^2$ = {inter['cross_block_is_sum_of_a_and_b_r2']:.3f}", fontsize=12)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f"{name}_cross_block.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


def fig_spectrum(stats, name, compare=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    ax.plot(np.arange(1, stats["_cum"].shape[1] + 1), stats["_cum"].mean(0),
            lw=1.7, label=f"{name}: full $B_k$ (226 dims)")
    ax.plot(np.arange(1, stats["_cum_cross"].shape[1] + 1), stats["_cum_cross"].mean(0),
            lw=1.7, ls="-.", label=f"{name}: cross block $B_{{ab}}$ (113)")
    if compare:
        ax.plot(np.arange(1, compare[1]["_cum_cross"].shape[1] + 1),
                compare[1]["_cum_cross"].mean(0), lw=1.5, ls="--",
                color="C3", label=f"{compare[0]}: cross block")
    ax.axhline(0.99, color="k", ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("number of components (log)")
    ax.set_ylabel("cumulative share of spectral energy")
    ax.set_title("Spectrum concentration")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    hi = 130.0
    bins = np.linspace(0, hi, 40)
    ax.hist(stats["_pr"], bins=bins, alpha=0.75, label=f"{name}: full $B_k$")
    ax.hist(stats["_pr_cross"], bins=bins, alpha=0.6, label=f"{name}: cross block")
    if compare:
        ax.hist(compare[1]["_pr_cross"], bins=bins, alpha=0.5, histtype="step",
                lw=1.8, color="C3", label=f"{compare[0]}: cross block")
    ax.set_xlabel("participation ratio $(\\sum\\lambda^2)^2/\\sum\\lambda^4$")
    ax.set_ylabel("# of output classes")
    ax.set_title("Effective rank per output")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f"{name}_spectrum.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


def fig_ablation(abl, name, compare=None):
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    k1 = [k for k, _ in abl["keep_top_k"]]
    a1 = [a for _, a in abl["keep_top_k"]]
    k2 = [k for k, _ in abl["drop_top_k"]]
    a2 = [a for _, a in abl["drop_top_k"]]
    ax.plot(k1, a1, "o-", lw=1.8, color="C0", label=f"{name}: keep top-$K$")
    ax.plot(k2, a2, "s-", lw=1.8, color="C3", label=f"{name}: delete top-$K$")
    if compare:
        ck = [k for k, _ in compare[1]["keep_top_k"]]
        ca = [a for _, a in compare[1]["keep_top_k"]]
        ax.plot(ck, ca, "o--", lw=1.4, color="C7", label=f"{compare[0]}: keep top-$K$")
    ax.axhline(abl["baseline_accuracy"], color="k", ls=":", lw=1,
               label="unmodified model")
    ax.set_xlabel("$K$ = number of frequencies edited")
    ax.set_ylabel("accuracy on all $P^2$ pairs")
    ax.set_title("Causal frequency surgery on the embeddings")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f"{name}_ablation.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


def fig_interaction(B, p, name, ks=(0, 1, 2, 37, 56, 112)):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, k in zip(axes.flat, ks):
        M = B[k]
        vmax = np.abs(M).max()
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.axhline(p - 0.5, color="k", lw=0.8)
        ax.axvline(p - 0.5, color="k", lw=0.8)
        ax.set_title(f"$B_{{k={k}}}$", fontsize=10)
        ax.set_xticks([p // 2, p + p // 2])
        ax.set_xticklabels(["$a$", "$b$"])
        ax.set_yticks([p // 2, p + p // 2])
        ax.set_yticklabels(["$a$", "$b$"])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{name}: full interaction matrices $B_k$ "
                 f"(all {p} outputs are real classes)", fontsize=12)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f"{name}_interaction_matrices.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


# --------------------------------------------------------------------------- #

def analyze(name: str, verbose: bool = True) -> dict:
    model, ckpt = load_run(name)
    p = ckpt["p"]
    if ckpt["arch"] != "bilinear":
        raise SystemExit(f"run {name!r} is arch={ckpt['arch']}; the weight "
                         "contraction is exact only for the bilinear architecture")

    B = interaction_tensor(model)
    vals, _ = eigendecompose(B)

    with torch.no_grad():
        X, _, _, _ = all_pairs(p)
        L = model(X).double().numpy().reshape(p, p, p)

    nf = neuron_fourier(model, p)
    inter = interaction_structure(B, p)
    sym = block_symmetry(B, p, L)
    stats = spectrum_stats(vals, B, p)
    abl = ablation_curve(model, p, nf["_order"])

    metrics = {
        "run": name,
        "train_acc": ckpt["final"]["train_acc"],
        "test_acc": ckpt["final"]["test_acc"],
        "grok_epoch": ckpt["final"].get("grok_epoch"),
        "n_params": ckpt["final"]["n_params"],
        "tensor_shape": list(B.shape),
        "neuron_fourier": {k: v for k, v in nf.items() if not k.startswith("_")},
        "interaction": {k: v for k, v in inter.items() if not k.startswith("_")},
        "symmetry": sym,
        "spectrum": {k: v for k, v in stats.items() if not k.startswith("_")},
        "ablation": abl,
    }

    if verbose:
        f, i, s = metrics["neuron_fourier"], metrics["interaction"], metrics["spectrum"]
        print(f"=== {name} ===")
        print(f"  train acc {metrics['train_acc']:.4f} | test acc {metrics['test_acc']:.4f} "
              f"| params {metrics['n_params']:,} | B: {tuple(B.shape)}")
        print("  -- neuron-level Fourier structure --")
        print(f"     single-frequency purity (importance-weighted): {f['purity_weighted']:.3f}")
        print(f"     a- and b-embedding use the same frequency:     {f['ab_frequency_match']:.3f}")
        print(f"     frequencies carrying 90% / 99% of power:       "
              f"{f['n_freqs_for_90pct']} / {f['n_freqs_for_99pct']} "
              f"of {f['n_freqs_available']}")
        print(f"     most-used frequencies: {f['top_frequencies'][:6]}")
        print("  -- interaction block B_ab --")
        print(f"     share of |B|^2 in the cross block:             "
              f"{i['cross_block_energy_fraction']:.3f}")
        print(f"     of the diagonal blocks, share a one-hot probes:"
              f" {i['diag_block_energy_actually_probed']:.4f}")
        print(f"     B_ab depends on (a+b) mod P, R^2:              "
              f"{i['cross_block_is_sum_of_a_and_b_r2']:.4f}")
        print(f"     predictions commute, f(a,b)==f(b,a):           "
              f"{sym['prediction_commutativity']:.4f}")
        print(f"     logit asymmetry (relative RMS):                "
              f"{sym['logit_asymmetry']:.4f}")
        print("  -- rank --")
        print(f"     participation ratio, full B_k:   {s['full_B_participation_ratio']:.1f} "
              f"of {s['full_B_dim']}   (contaminated by the unprobed subspace)")
        print(f"     participation ratio, cross block:{s['cross_block_participation_ratio']:6.1f} "
              f"of {s['cross_block_dim']}")
        print(f"     lambda1/lambda2 = {s['lambda1_over_lambda2']:.2f}  "
              "(the old repo's statistic; see module docstring)")
        print("  -- causal frequency surgery --")
        print(f"     baseline accuracy on all pairs: {abl['baseline_accuracy']:.4f}")
        print("     keep only top-K:  " +
              ", ".join(f"K={k}:{a:.3f}" for k, a in abl["keep_top_k"][:6]))
        print("     delete top-K:     " +
              ", ".join(f"K={k}:{a:.3f}" for k, a in abl["drop_top_k"][-6:]))
        print(f"     smallest sufficient frequency set: {abl['n_freqs_sufficient']}")
        print(f"     most-used frequencies deletable with no loss: "
              f"{abl['n_freqs_deletable']}")

    return dict(metrics=metrics, B=B, vals=vals, nf=nf, inter=inter,
                stats=stats, abl=abl, p=p)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="grok")
    ap.add_argument("--compare", default=None,
                    help="second run to overlay (e.g. a memorizing control)")
    ap.add_argument("--no-figs", action="store_true")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    r = analyze(args.name)

    c = None
    if args.compare:
        print()
        c = analyze(args.compare)
        with open(os.path.join(RESULTS_DIR, f"{args.compare}_metrics.json"), "w") as fh:
            json.dump(c["metrics"], fh, indent=2)

    with open(os.path.join(RESULTS_DIR, f"{args.name}_metrics.json"), "w") as fh:
        json.dump(r["metrics"], fh, indent=2)

    if not args.no_figs:
        print()
        fig_neurons(r["nf"], r["p"], args.name)
        fig_cross_block(r["inter"], r["p"], args.name)
        fig_spectrum(r["stats"], args.name,
                     compare=(args.compare, c["stats"]) if c else None)
        fig_ablation(r["abl"], args.name,
                     compare=(args.compare, c["abl"]) if c else None)
        fig_interaction(r["B"], r["p"], args.name)
        if c:
            fig_neurons(c["nf"], c["p"], args.compare)
            fig_cross_block(c["inter"], c["p"], args.compare)
            fig_interaction(c["B"], c["p"], args.compare)


if __name__ == "__main__":
    main()
