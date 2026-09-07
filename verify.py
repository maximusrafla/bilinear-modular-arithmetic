"""
Check every quantitative claim in README.md against the committed JSON.

The tables in this repo are generated from results/*.json, but the prose that
summarises them is written by hand -- and that is exactly where the last two
errors lived. A hyperparameter cell said 0.990 because it came from a 30k-epoch
run while the shipped default is 25k, and the sentence above the table said
"eight of ten settings reach 100%" when seven do (eight is the number that ever
crossed 90%; one of those decays again by the end).

Both survived a check that compared table cells to JSON, because a sentence is
not a cell. So this script checks the sentences too: every number quoted in the
README prose is recomputed from the artifacts and compared.

    python verify.py          # exits nonzero if anything drifts
"""

from __future__ import annotations

import json
import os
import re
import sys

RESULTS = "results"
NUMBER_WORDS = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
                6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten"}


def load(name):
    with open(os.path.join(RESULTS, name), encoding="utf-8") as fh:
        return json.load(fh)


class Checker:
    def __init__(self, readme):
        self.readme = readme
        self.results = []

    def ok(self, label, cond):
        self.results.append((label, bool(cond)))

    def close(self, label, got, want, tol=5e-4):
        self.ok(f"{label}: README {got} vs artifact {want}", abs(got - want) <= tol)

    def find(self, label, pattern):
        """Extract a single captured number from the prose, or fail loudly."""
        m = re.search(pattern, self.readme)
        if not m:
            self.ok(f"{label}: pattern not found in README", False)
            return None
        return m.group(1)

    def row(self, prefix, ncells=None):
        """First table row starting with `prefix`.

        `ncells` disambiguates: several row labels (e.g. "| test accuracy")
        appear in more than one table, and the width table would otherwise
        shadow the two-column comparison table.
        """
        for line in self.readme.splitlines():
            if line.startswith(prefix):
                cells = [c.strip().strip("*") for c in line.split("|")[2:-1]]
                if ncells is None or len(cells) == ncells:
                    return cells
        return None


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    readme = open("README.md", encoding="utf-8").read()
    g = load("grok_metrics.json")
    m = load("memorize_metrics.json")
    ws = load("width_sweep.json")
    sd = load("seed_sweep.json")
    hp = load("hparam_sweep.json")
    c = Checker(readme)

    # ---- encoding / links -------------------------------------------------
    c.ok("README contains no replacement characters", readme.count(chr(0xFFFD)) == 0)
    for path in re.findall(r"\]\((results/[^)]+)\)", readme):
        c.ok(f"image exists: {path}", os.path.isfile(path))

    # ---- headline prose ---------------------------------------------------
    p = g["n_params"]
    n_pairs = 113 * 113
    n_test = n_pairs - round(0.3 * n_pairs)
    v = c.find("held-out pair count", r"\*\*100% test accuracy on\s*\n?([\d,]+) held-out pairs")
    if v:
        c.ok(f"held-out pairs {v} == {n_test}", int(v.replace(",", "")) == n_test)
    c.ok(f"grok test accuracy is 1.0", g["test_acc"] == 1.0)

    v = c.find("purity", r"single-frequency purity is \*\*([\d.]+)\*\*")
    if v:
        c.close("purity", float(v), g["neuron_fourier"]["purity_weighted"])

    v = c.find("a/b agreement", r"importance-weighted agreement \*\*([\d.]+)\*\*")
    if v:
        c.close("a/b agreement", float(v), g["neuron_fourier"]["ab_frequency_match"])

    v = c.find("frequencies used", r"spreads across \*\*(\d+) of the (\d+) available")
    mm = re.search(r"spreads across \*\*(\d+) of the (\d+) available", readme)
    if mm:
        c.ok(f"frequencies used {mm.group(1)}",
             int(mm.group(1)) == g["neuron_fourier"]["n_freqs_for_90pct"])
        c.ok(f"frequencies available {mm.group(2)}",
             int(mm.group(2)) == g["neuron_fourier"]["n_freqs_available"])

    v = c.find("deletable", r"deleting the (\d+) most-used")
    if v:
        c.ok(f"deletable {v}", int(v) == g["ablation"]["n_freqs_deletable"])

    v = c.find("sufficient", r"while the \*\*(\d+)\*\*\s*\n?most-used frequencies")
    if v:
        c.ok(f"sufficient {v}", int(v) == g["ablation"]["n_freqs_sufficient"])
    v = c.find("keep-top-K accuracy", r"already recover \*\*([\d.]+)%\*\*")
    if v:
        keep = dict(g["ablation"]["keep_top_k"])[g["ablation"]["n_freqs_sufficient"]]
        c.close("keep-top-K accuracy", float(v) / 100, keep, tol=1e-3)

    v = c.find("param count", r"\*\*([\d,]+) parameters\.\*\*")
    if v:
        c.ok(f"param count {v}", int(v.replace(",", "")) == p)

    v = c.find("grok epoch", r"first crossing 90% at epoch \*\*([\d,]+)\*\*")
    if v:
        c.ok(f"grok epoch {v}", int(v.replace(",", "")) == g["grok_epoch"])

    # ---- interaction block ------------------------------------------------
    mm = re.search(r"\(a\+b\) mod P` with \*\*R² = ([\d.]+)\*\* in\s*\n"
                   r"the grokked model, versus \*\*([\d.]+)\*\*", readme)
    if mm:
        c.close("B_ab R2 grok", float(mm.group(1)),
                g["interaction"]["cross_block_is_sum_of_a_and_b_r2"])
        c.close("B_ab R2 memorize", float(mm.group(2)),
                m["interaction"]["cross_block_is_sum_of_a_and_b_r2"])

    v = c.find("unprobed share", r"task and hold \*\*([\d.]+)%\*\* of those blocks")
    if v:
        c.close("unprobed share", float(v) / 100,
                1 - g["interaction"]["diag_block_energy_actually_probed"], tol=1e-3)

    mm = re.search(r"full `B_k` is \*\*(\d+) of (\d+)\*\* for the model that\s*\n"
                   r"generalizes and \*\*(\d+) of \d+\*\*", readme)
    if mm:
        c.ok("PR grok", abs(int(mm.group(1)) - g["spectrum"]["full_B_participation_ratio"]) < 1)
        c.ok("PR dim", int(mm.group(2)) == g["spectrum"]["full_B_dim"])
        c.ok("PR memorize",
             abs(int(mm.group(3)) - m["spectrum"]["full_B_participation_ratio"]) < 1)

    # ---- comparison table -------------------------------------------------
    table = {
        "| test accuracy": (g["test_acc"], m["test_acc"]),
        "| single-frequency purity": (g["neuron_fourier"]["purity_weighted"],
                                      m["neuron_fourier"]["purity_weighted"]),
        "| `a`/`b` embeddings share a frequency":
            (g["neuron_fourier"]["ab_frequency_match"],
             m["neuron_fourier"]["ab_frequency_match"]),
        "| `B_ab` explained by `(a+b) mod P`, R²":
            (g["interaction"]["cross_block_is_sum_of_a_and_b_r2"],
             m["interaction"]["cross_block_is_sum_of_a_and_b_r2"]),
        "| predictions commute": (g["symmetry"]["prediction_commutativity"],
                                  m["symmetry"]["prediction_commutativity"]),
        "| smallest sufficient frequency set":
            (g["ablation"]["n_freqs_sufficient"], m["ablation"]["n_freqs_sufficient"]),
        "| frequencies deletable with no loss":
            (g["ablation"]["n_freqs_deletable"], m["ablation"]["n_freqs_deletable"]),
        "| σ₁/σ₂ of the full `B_k`": (g["spectrum"]["lambda1_over_lambda2"],
                                      m["spectrum"]["lambda1_over_lambda2"]),
    }
    for prefix, (wg, wm) in table.items():
        cells = c.row(prefix, ncells=2)
        if cells is None or len(cells) < 2:
            c.ok(f"comparison row missing: {prefix}", False)
            continue
        for cell, want, who in ((cells[0], wg, "grok"), (cells[1], wm, "memorize")):
            try:
                c.close(f"{prefix.strip('| ')} [{who}]", float(cell), want, tol=1e-2)
            except ValueError:
                c.ok(f"{prefix} [{who}] not numeric: {cell!r}", False)

    # ---- width table ------------------------------------------------------
    cells = c.row("| frequencies used (90% of power)", ncells=len(ws["rows"]))
    c.ok("width freq row matches width_sweep.json",
         cells is not None and [int(x) for x in cells] == [r["n_freqs_90"] for r in ws["rows"]])
    cells = c.row("| hidden width", ncells=len(ws["rows"]))
    c.ok("width header matches width_sweep.json",
         cells is not None and [int(x) for x in cells] == [r["width"] for r in ws["rows"]])

    # ---- seed prose -------------------------------------------------------
    r = sd["range"]

    def rng_txt(key, fmt="{:.3f}"):
        lo, hi = r[key]["min"], r[key]["max"]
        f = (lambda x: fmt.format(x)) if isinstance(lo, float) else (lambda x: str(x))
        return f(lo) if lo == hi else f"{f(lo)}–{f(hi)}"

    for key, label in (("purity", "purity"), ("cross_block_r2", "seed R2"),
                       ("n_freqs_for_90pct", "seed freqs"),
                       ("n_freqs_sufficient", "seed sufficient"),
                       ("grok_epoch", "seed grok onset")):
        txt = rng_txt(key)
        c.ok(f"{label} range '{txt}' quoted in README", txt in readme)
    c.ok("seed test accuracy all 1.000",
         r["test_acc"]["min"] == r["test_acc"]["max"] == 1.0)
    c.ok("seed a/b agreement all 1.000",
         r["ab_frequency_match"]["min"] == r["ab_frequency_match"]["max"] == 1.0)
    c.ok("seed deletable constant and quoted",
         r["n_freqs_deletable"]["min"] == r["n_freqs_deletable"]["max"]
         and str(r["n_freqs_deletable"]["min"]) in readme)
    c.ok("README lists the swept seeds",
         all(str(x["seed"]) in readme for x in sd["rows"]))

    # ---- hyperparameter table AND the sentence above it -------------------
    lrs = sorted({x["lr"] for x in hp["rows"]})
    wds = sorted({x["weight_decay"] for x in hp["rows"]})
    acc = {(x["lr"], x["weight_decay"]): x["test_acc"] for x in hp["rows"]}
    for lr in lrs:
        cells = c.row(f"| **lr {lr:.0e}".replace("e-03", "e-3"), ncells=len(wds))
        if cells is None:
            c.ok(f"hparam row for lr={lr:.0e} missing", False)
            continue
        for w, cell in zip(wds, cells):
            c.close(f"hparam lr={lr:.0e} wd={w:g}", float(cell), acc[(lr, w)])

    n_at_100 = sum(1 for x in hp["rows"] if x["test_acc"] >= 0.999)
    n_ever = sum(1 for x in hp["rows"] if x["grok_epoch"] is not None)
    v = c.find("hparam success count", r"\*\*(\w+)\*\* of\s*\n?the ten settings")
    if v:
        c.ok(f"success count '{v}' == {n_at_100} ({NUMBER_WORDS[n_at_100]})",
             v.capitalize() == NUMBER_WORDS[n_at_100])
    c.ok(f"README distinguishes ever-generalised ({n_ever}) from final ({n_at_100})",
         f"({NUMBER_WORDS[n_ever].lower()})" in readme
         and f"({NUMBER_WORDS[n_at_100].lower()})" in readme)

    decay = [x for x in hp["rows"]
             if x["grok_epoch"] is not None and x["test_acc"] < 0.999]
    for x in decay:
        c.ok(f"transient-grok run lr={x['lr']:.0e} wd={x['weight_decay']:g} described",
             f"{x['grok_epoch']:,}" in readme and f"{x['test_acc']:.3f}" in readme)

    # ---- report -----------------------------------------------------------
    failed = [lab for lab, good in c.results if not good]
    for lab, good in c.results:
        print(("  OK   " if good else "  FAIL ") + lab)
    print(f"\n{len(c.results) - len(failed)}/{len(c.results)} checks passed")
    if failed:
        print("\nFAILURES:")
        for lab in failed:
            print("  - " + lab)
        return 1
    print("Every quantitative claim in README.md matches the committed artifacts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
