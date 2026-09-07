"""
Bilinear MLP on modular addition: model, data, and weight-space analysis
primitives.

A bilinear MLP (Pearce et al., arXiv:2410.08417) is a gated linear unit with
the element-wise nonlinearity removed:

    h(x) = (W_l x) * (W_r x)        (* = element-wise)
    f(x) = W_out h(x)

Because f is exactly quadratic in x, each output logit is a quadratic form

    f_k(x) = x^T Q_k x,    Q_k[i,j] = sum_h W_out[k,h] W_l[h,i] W_r[h,j]

and only the symmetric part B_k = (Q_k + Q_k^T)/2 affects the output. B_k can
be eigendecomposed exactly from the weights alone -- no activations, no probes,
no sampling error. That is the whole point of the architecture for
interpretability.

Task: f(a, b) = (a + b) mod P, with x = [onehot(a); onehot(b)] in R^{2P}.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

P_DEFAULT = 113


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class BilinearMLP(nn.Module):
    """f(x) = W_out((W_l x) * (W_r x)), no biases.

    Biases are omitted deliberately: they would add linear and constant terms
    to f and break the exact quadratic-form contraction used in analyze.py.
    """

    def __init__(self, p: int = P_DEFAULT, hidden_dim: int = 256, init_scale: float = 1.0):
        super().__init__()
        self.p = p
        self.hidden_dim = hidden_dim
        self.input_dim = 2 * p
        self.W_l = nn.Linear(self.input_dim, hidden_dim, bias=False)
        self.W_r = nn.Linear(self.input_dim, hidden_dim, bias=False)
        # NOTE: output dim is p (one logit per residue class), not 2p.
        self.W_out = nn.Linear(hidden_dim, p, bias=False)

        if init_scale != 1.0:
            with torch.no_grad():
                for m in (self.W_l, self.W_r, self.W_out):
                    m.weight.mul_(init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W_out(self.W_l(x) * self.W_r(x))

    def n_params(self) -> int:
        return sum(q.numel() for q in self.parameters())


class ReLUMLP(nn.Module):
    """ReLU control: f(x) = W_out relu(W_in x).

    A baseline for accuracy only -- it admits no exact weight-space
    contraction, which is the point of the comparison, and analyze.py
    refuses it. It has 3p*H parameters against the bilinear layer's 5p*H,
    so `--hidden-dim 427` gives 144,753, closely matching the 144,640 of the
    default bilinear model.
    """

    def __init__(self, p: int = P_DEFAULT, hidden_dim: int = 256):
        super().__init__()
        self.p = p
        self.hidden_dim = hidden_dim
        self.input_dim = 2 * p
        self.W_in = nn.Linear(self.input_dim, hidden_dim, bias=False)
        self.W_out = nn.Linear(hidden_dim, p, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W_out(torch.relu(self.W_in(x)))

    def n_params(self) -> int:
        return sum(q.numel() for q in self.parameters())


def build_model(arch: str, p: int, hidden_dim: int, init_scale: float = 1.0) -> nn.Module:
    if arch == "bilinear":
        return BilinearMLP(p, hidden_dim, init_scale)
    if arch == "relu":
        return ReLUMLP(p, hidden_dim)
    raise ValueError(f"unknown arch: {arch!r}")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def all_pairs(p: int = P_DEFAULT):
    """Enumerate the complete input space: all p*p ordered pairs (a, b).

    Returns X of shape (p*p, 2p) (concatenated one-hots) and y of shape (p*p,).
    The old version of this repo sampled 20k random pairs from a space of only
    12,769, i.e. sampled with heavy replacement and never held anything out.
    """
    a = np.repeat(np.arange(p), p)
    b = np.tile(np.arange(p), p)
    y = (a + b) % p

    X = np.zeros((p * p, 2 * p), dtype=np.float32)
    X[np.arange(p * p), a] = 1.0
    X[np.arange(p * p), p + b] = 1.0
    return torch.from_numpy(X), torch.from_numpy(y.astype(np.int64)), a, b


def split_indices(n: int, train_frac: float, seed: int):
    """Random train/test split over the enumerated pairs."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(round(train_frac * n))
    return (
        torch.from_numpy(np.sort(perm[:n_train]).copy()),
        torch.from_numpy(np.sort(perm[n_train:]).copy()),
    )


# --------------------------------------------------------------------------- #
# Weight-space contraction
# --------------------------------------------------------------------------- #

def interaction_tensor(model: BilinearMLP) -> np.ndarray:
    """Symmetric interaction tensor B of shape (p, 2p, 2p).

    B[k] is the symmetric matrix with f_k(x) = x^T B[k] x. This is the exact
    quadratic form, obtained purely from weights.
    """
    W_l = model.W_l.weight.detach().cpu().double().numpy()      # (H, 2P)
    W_r = model.W_r.weight.detach().cpu().double().numpy()      # (H, 2P)
    W_out = model.W_out.weight.detach().cpu().double().numpy()  # (P, H)

    Q = np.einsum("kh,hi,hj->kij", W_out, W_l, W_r, optimize=True)
    return 0.5 * (Q + np.transpose(Q, (0, 2, 1)))


def eigendecompose(B: np.ndarray):
    """Eigendecompose every output's interaction matrix.

    Returns eigenvalues (p, 2p) and eigenvectors (p, 2p, 2p) with
    eigvecs[k][:, i] the eigenvector for eigvals[k, i], sorted by |lambda|
    descending. Each logit is then f_k(x) = sum_i lambda_i (v_i . x)^2.
    """
    vals, vecs = np.linalg.eigh(B)  # ascending, real (B is symmetric)
    order = np.argsort(-np.abs(vals), axis=1)
    vals = np.take_along_axis(vals, order, axis=1)
    vecs = np.take_along_axis(vecs, order[:, None, :], axis=2)
    return vals, vecs


# --------------------------------------------------------------------------- #
# Spectrum statistics
# --------------------------------------------------------------------------- #

def participation_ratio(vals: np.ndarray, axis: int = -1) -> np.ndarray:
    """Participation ratio of a spectrum: (sum l^2)^2 / sum l^4.

    Equals r for a flat rank-r spectrum and 1 for a rank-1 spectrum. This is
    the statistic the old README should have used. The ratio sigma_1/sigma_2
    it reported instead is ~1 for ANY solution whose components come in
    degenerate pairs -- which is exactly what a clean Fourier circuit produces,
    so it cannot distinguish low-rank structure from its absence.
    """
    s2 = np.sum(vals ** 2, axis=axis)
    s4 = np.sum(vals ** 4, axis=axis)
    return np.where(s4 > 0, s2 ** 2 / np.maximum(s4, 1e-300), 0.0)


# --------------------------------------------------------------------------- #
# Frequency surgery on the weights
# --------------------------------------------------------------------------- #

def _filter_block(block: np.ndarray, keep: set[int], p: int, drop: bool) -> np.ndarray:
    """Keep (or drop) a set of Fourier frequencies in each row of an embedding block."""
    spec = np.fft.rfft(block, axis=1)
    mask = np.zeros(spec.shape[1], dtype=bool)
    for w in keep:
        if 0 <= w < spec.shape[1]:
            mask[w] = True

    # DC is a per-neuron offset, not a frequency, so it is never part of the
    # edit: it survives in both directions. Folding it into `mask` would mean
    # the drop path silently deleted it too.
    mask[0] = False
    if drop:
        sel = ~mask
    else:
        sel = mask.copy()
        sel[0] = True
    return np.fft.irfft(spec * sel, n=p, axis=1)


def frequency_ablate(model, keep_freqs, p: int, drop: bool = False):
    """Return a copy of `model` whose input embeddings keep only `keep_freqs`.

    Each hidden neuron reads the input through two length-P embeddings (one over
    `a`, one over `b`) in each of W_l and W_r. Filtering those rows in Fourier
    space is a direct causal edit of the weights: if the model really computes
    with frequencies {w}, keeping only those should preserve accuracy, and
    deleting only those should destroy it.
    """
    m = copy.deepcopy(model).cpu()
    keep = set(int(w) for w in keep_freqs)
    with torch.no_grad():
        for lin in (m.W_l, m.W_r):
            W = lin.weight.detach().numpy()
            new = np.concatenate([_filter_block(W[:, :p], keep, p, drop),
                                  _filter_block(W[:, p:], keep, p, drop)], axis=1)
            lin.weight.copy_(torch.from_numpy(new.astype(np.float32)))
    return m.eval()


@torch.no_grad()
def accuracy(model, X: torch.Tensor, y: torch.Tensor) -> float:
    return (model(X).argmax(dim=1) == y).float().mean().item()
