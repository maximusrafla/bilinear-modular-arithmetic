"""
Reproduce the main results and figures from scratch.

Trains the two contrasting runs on an identical 30/70 split -- one that
generalizes and one that only memorizes -- then runs the identical weight-space
analysis on both. The comparison is the point: it is what lets the analysis
attribute a finding to the task rather than to the training regime.

    python run_all.py                 # ~3 min on a consumer GPU
    python run_all.py --epochs 6000   # faster; grokking sets in around epoch 4k

The width experiment lives in width_sweep.py and is run separately.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

RUNS = [
    # name, extra args, what it is for
    ("grok", ["--weight-decay", "1.0"],
     "regularized: groks, generalizes to held-out pairs"),
    ("memorize", ["--weight-decay", "0.0"],
     "unregularized control: fits train, fails test"),
]


def sh(cmd):
    print(f"\n$ {' '.join(cmd)}\n" + "=" * 72)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25000)
    ap.add_argument("--train-frac", type=float, default=0.3)
    args = ap.parse_args()

    for name, extra, why in RUNS:
        print(f"\n### {name}: {why}")
        sh([sys.executable, "train.py", "--name", name,
            "--epochs", str(args.epochs),
            "--train-frac", str(args.train_frac), *extra])

    sh([sys.executable, "analyze.py", "--name", "grok", "--compare", "memorize"])
    print("\nDone. Figures and metrics are in results/.")


if __name__ == "__main__":
    main()
