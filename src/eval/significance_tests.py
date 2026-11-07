#!/usr/bin/env python3
"""Significance tests for PPL-preference accuracy tables.

Each table cell is mean preference accuracy (PPL_gold < PPL_corrupt) over
training seeds. Per-seed, per-category accuracies come from the analysis cache
written by ``build_analysis_data.py`` (``eval_results/analysis_cache``), whose
item categories are joined from the test-set JSONs, so no re-evaluation is
needed when categories change. Both test sets (dat→acc, acc→dat) are analysed.

Categories (see ``corrupt_pairs.py``): ``core_pronoun`` / ``core_other`` (UD POS of
the NP head), and prepositional groups, either by the Turkish case of the
translation (dat→acc only) or by German preposition class (dative-only /
accusative-only / two-way / other). Family 2 only compares categories that
partition the items (never an aggregate with its own parts).

Two comparison families (different experimental units):

  1. Models (rows), same epoch + category
     Unit = seed (n=5). Unpaired Welch t-test. This is the right test for
     "does this training condition change German case preference?"

  2. Categories (columns), same epoch + model
     Unit = example (independent subsets). Two-proportion z-test on mean
     n_correct. This is the right test for "is two-way harder than dative-only?"

"No difference" is *not* the same as a non-significant t-test. TOST
equivalence tests (δ = 1 pp and 2 pp) are reported so a true null can be
supported rather than merely failed-to-reject.

Usage:
  python src/eval/significance_tests.py
  python src/eval/significance_tests.py --cache eval_results/analysis_cache \
      --out_json eval_results/significance_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "eval_results", "analysis_cache")

# Row label -> (L1 corpus, phase-2 condition) system in the analysis cache.
# Shifted-DE runs are not part of the v3 cache.
MODEL_SPECS: List[Tuple[str, Tuple[str, str]]] = [
    ("DE only", ("DE", "regular")),
    ("EN-DE", ("EN", "regular")),
    ("TR-DE", ("TR", "regular")),
    ("ZH-DE", ("ZH", "regular")),
    ("EN-DE (L1 shuffled)", ("EN", "shuffled")),
    ("TR-DE (L1 shuffled)", ("TR", "shuffled")),
    ("ZH-DE (L1 shuffled)", ("ZH", "shuffled")),
    ("EN-DE (reset embeddings)", ("EN", "reset")),
    ("TR-DE (reset embeddings)", ("TR", "reset")),
    ("ZH-DE (reset embeddings)", ("ZH", "reset")),
]

# Analysed test sets: name -> index in the cache.
TESTSETS = {"dat2acc": 0, "acc2dat": 1}

# Report category -> group key in the cache (ALL, aggregates and parts).
ALL_KEY = "ALL"
_CORE = {"core": "core", "core_pronoun": "cat:core_pronoun", "core_other": "cat:core_other"}
CATEGORY_KEYS: Dict[str, Dict[str, str]] = {
    "dat2acc": {
        "ALL": "ALL",
        **_CORE,
        "dative_only_prep": "de:dative_only_prep",
        "two_way_prep": "de:two_way_prep",
        "other_prep": "de:other_prep",
    },
    "acc2dat": {
        "ALL": "ALL",
        **_CORE,
        "accusative_only_prep": "de:accusative_only_prep",
        "two_way_prep": "de:two_way_prep",
        "other_prep": "de:other_prep",
    },
}
# Disjoint category sets: family 2 only compares categories within one set.
PARTITIONS: Dict[str, List[List[str]]] = {
    "dat2acc": [
        ["core_pronoun", "core_other", "dative_only_prep", "two_way_prep", "other_prep"],
    ],
    "acc2dat": [
        ["core_pronoun", "core_other", "accusative_only_prep", "two_way_prep",
         "other_prep"],
    ],
}
CORE_CATEGORIES = ["core_pronoun", "core_other"]

# Planned contrasts (not all pairwise) — these are the scientific questions.
PLANNED_VS_DE = [
    "EN-DE",
    "TR-DE",
    "ZH-DE",
    "EN-DE (L1 shuffled)",
    "TR-DE (L1 shuffled)",
    "ZH-DE (L1 shuffled)",
    "EN-DE (reset embeddings)",
    "TR-DE (reset embeddings)",
    "ZH-DE (reset embeddings)",
]
L1_REGULAR = ["EN-DE", "TR-DE", "ZH-DE"]
REGULAR_VS_CONTROL = [
    ("EN-DE", "EN-DE (L1 shuffled)"),
    ("TR-DE", "TR-DE (L1 shuffled)"),
    ("ZH-DE", "ZH-DE (L1 shuffled)"),
    ("EN-DE", "EN-DE (reset embeddings)"),
    ("TR-DE", "TR-DE (reset embeddings)"),
    ("ZH-DE", "ZH-DE (reset embeddings)"),
]


def benjamini_hochberg(pvals: List[float]) -> List[float]:
    n = len(pvals)
    if n == 0:
        return []
    order = np.argsort(pvals)
    q = np.empty(n, dtype=np.float64)
    prev = 1.0
    for rank_from_end, idx in enumerate(order[::-1]):
        rank = n - rank_from_end
        val = min(prev, pvals[idx] * n / rank)
        q[idx] = val
        prev = val
    return [float(min(1.0, x)) for x in q]


def welch_t(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n1, n2 = a.size, b.size
    if n1 < 2 or n2 < 2:
        return {
            "n_a": n1,
            "n_b": n2,
            "mean_a": float(np.mean(a)) if n1 else float("nan"),
            "mean_b": float(np.mean(b)) if n2 else float("nan"),
            "delta_pp": float("nan"),
            "t": float("nan"),
            "df": float("nan"),
            "p": float("nan"),
            "cohens_d": float("nan"),
        }
    m1, m2 = float(np.mean(a)), float(np.mean(b))
    v1, v2 = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    se = np.sqrt(v1 / n1 + v2 / n2)
    t = (m1 - m2) / se if se > 0 else 0.0
    num = (v1 / n1 + v2 / n2) ** 2
    den = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    df = num / den if den > 0 else float(n1 + n2 - 2)
    p = float(2 * stats.t.sf(abs(t), df))
    pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    d = (m1 - m2) / pooled if pooled > 0 else 0.0
    return {
        "n_a": n1,
        "n_b": n2,
        "mean_a": m1,
        "mean_b": m2,
        "delta_pp": 100.0 * (m1 - m2),
        "t": float(t),
        "df": float(df),
        "p": p,
        "cohens_d": float(d),
    }


def tost_welch(a: np.ndarray, b: np.ndarray, delta: float) -> Dict[str, float]:
    """Two one-sided tests for |mean(a)-mean(b)| < delta (accuracy units)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        return {"p_tost": float("nan"), "equivalent": False}
    m1, m2 = float(np.mean(a)), float(np.mean(b))
    v1 = float(np.var(a, ddof=1))
    v2 = float(np.var(b, ddof=1))
    n1, n2 = a.size, b.size
    se = np.sqrt(v1 / n1 + v2 / n2)
    num = (v1 / n1 + v2 / n2) ** 2
    den = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    df = num / den if den > 0 else float(n1 + n2 - 2)
    diff = m1 - m2
    # H0: diff <= -delta  vs  H1: diff > -delta
    t_lo = (diff + delta) / se if se > 0 else 0.0
    p_lo = float(stats.t.sf(t_lo, df))  # 1 - cdf
    # H0: diff >= +delta  vs  H1: diff < +delta
    t_hi = (diff - delta) / se if se > 0 else 0.0
    p_hi = float(stats.t.cdf(t_hi, df))
    p_tost = max(p_lo, p_hi)
    return {"p_tost": p_tost, "equivalent": bool(p_tost < 0.05)}


def two_prop_z(k1: float, n1: int, k2: float, n2: int) -> Dict[str, float]:
    if n1 <= 0 or n2 <= 0:
        return {"p": float("nan"), "z": float("nan"), "delta_pp": float("nan")}
    p1, p2 = k1 / n1, k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else 0.0
    p = float(2 * stats.norm.sf(abs(z)))
    return {
        "p": p,
        "z": float(z),
        "delta_pp": 100.0 * (p1 - p2),
        "p1": p1,
        "p2": p2,
        "n1": n1,
        "n2": n2,
    }


class Cache:
    """Per-seed accuracies and group sizes from ``build_analysis_data.py``."""

    def __init__(self, cache_dir: str) -> None:
        with open(os.path.join(cache_dir, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.grp = np.load(os.path.join(cache_dir, "bundle.npz"))["grp"]
        self.sys = {(x["l1"], x["cond"]): i for i, x in enumerate(self.meta["systems"])}
        self.group = {g["key"]: i for i, g in enumerate(self.meta["groups"])}
        self.metric = self.meta["group_metrics"].index("acc")

    def accuracy(self, system: Tuple[str, str], epoch: int, ts: int, key: str) -> Optional[np.ndarray]:
        if system not in self.sys or key not in self.group:
            return None
        x = self.grp[self.sys[system], epoch, ts, self.group[key], self.metric, :]
        x = x[~np.isnan(x)].astype(np.float64)
        return x if x.size else None

    def n_examples(self, ts: int, key: str) -> int:
        return int(self.meta["group_n"][ts][self.group[key]]) if key in self.group else 0


def collect_cells(cache: Cache, testset: str) -> Dict[Tuple[str, int, str], np.ndarray]:
    """(model_label, epoch, category) -> seed accuracies."""
    ts = TESTSETS[testset]
    cells: Dict[Tuple[str, int, str], np.ndarray] = {}
    for label, system in MODEL_SPECS:
        for epoch in cache.meta["epochs"]:
            for cat, key in CATEGORY_KEYS[testset].items():
                acc = cache.accuracy(system, epoch, ts, key)
                if acc is not None:
                    cells[(label, int(epoch), cat)] = acc
    return cells


def n_examples_map(cache: Cache, testset: str) -> Dict[str, int]:
    ts = TESTSETS[testset]
    return {
        cat: cache.n_examples(ts, key)
        for cat, key in CATEGORY_KEYS[testset].items()
        if cache.n_examples(ts, key) > 0
    }


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < 1e-4:
        return "<1e-4"
    return f"{p:.4f}"


def analyse(args: argparse.Namespace, cache: Cache, testset: str) -> Dict[str, Any]:
    """Run both comparison families on one test set; print the report, return its JSON."""
    print("\n" + "#" * 88)
    print(f"# TEST SET: {testset}")
    print("#" * 88)
    cells = collect_cells(cache, testset)
    n_ex = n_examples_map(cache, testset)
    categories = list(n_ex)
    epochs = [0, 1, 3, 5]
    models = [m[0] for m in MODEL_SPECS]

    # Reconstruct PDF-style tables (percent).
    tables: Dict[str, Any] = {}
    for epoch in epochs:
        rows = []
        for label in models:
            row: Dict[str, Any] = {"model": label, "epoch": epoch}
            for cat in categories:
                acc = cells.get((label, epoch, cat))
                if acc is None:
                    row[cat] = None
                    row[f"{cat}_std"] = None
                    row[f"{cat}_n"] = 0
                else:
                    row[cat] = float(np.mean(acc) * 100)
                    row[f"{cat}_std"] = float(np.std(acc, ddof=1) * 100) if acc.size > 1 else 0.0
                    row[f"{cat}_n"] = int(acc.size)
            rows.append(row)
        tables[str(epoch)] = rows

    # --- Family 1: model vs model, planned contrasts, seed-level Welch ---
    planned_rows: List[Dict[str, Any]] = []
    for epoch in [1, 3, 5]:  # epoch 0 DE-only has 1 seed
        for cat in categories:
            # vs DE only
            de = cells.get(("DE only", epoch, cat))
            if de is not None and de.size >= 2:
                for other in PLANNED_VS_DE:
                    oth = cells.get((other, epoch, cat))
                    if oth is None or oth.size < 2:
                        continue
                    t = welch_t(oth, de)
                    t1 = tost_welch(oth, de, 0.01)
                    t2 = tost_welch(oth, de, 0.02)
                    planned_rows.append(
                        {
                            "family": "vs_DE_only",
                            "epoch": epoch,
                            "category": cat,
                            "model_a": other,
                            "model_b": "DE only",
                            **t,
                            "p_tost_1pp": t1["p_tost"],
                            "eq_1pp": t1["equivalent"],
                            "p_tost_2pp": t2["p_tost"],
                            "eq_2pp": t2["equivalent"],
                        }
                    )
            # L1 regular vs each other
            for a, b in combinations(L1_REGULAR, 2):
                aa = cells.get((a, epoch, cat))
                bb = cells.get((b, epoch, cat))
                if aa is None or bb is None or aa.size < 2 or bb.size < 2:
                    continue
                t = welch_t(aa, bb)
                t1 = tost_welch(aa, bb, 0.01)
                t2 = tost_welch(aa, bb, 0.02)
                planned_rows.append(
                    {
                        "family": "L1_vs_L1",
                        "epoch": epoch,
                        "category": cat,
                        "model_a": a,
                        "model_b": b,
                        **t,
                        "p_tost_1pp": t1["p_tost"],
                        "eq_1pp": t1["equivalent"],
                        "p_tost_2pp": t2["p_tost"],
                        "eq_2pp": t2["equivalent"],
                    }
                )
            # regular vs controls
            for a, b in REGULAR_VS_CONTROL:
                aa = cells.get((a, epoch, cat))
                bb = cells.get((b, epoch, cat))
                if aa is None or bb is None or aa.size < 2 or bb.size < 2:
                    continue
                t = welch_t(aa, bb)
                t1 = tost_welch(aa, bb, 0.01)
                t2 = tost_welch(aa, bb, 0.02)
                planned_rows.append(
                    {
                        "family": "regular_vs_control",
                        "epoch": epoch,
                        "category": cat,
                        "model_a": a,
                        "model_b": b,
                        **t,
                        "p_tost_1pp": t1["p_tost"],
                        "eq_1pp": t1["equivalent"],
                        "p_tost_2pp": t2["p_tost"],
                        "eq_2pp": t2["equivalent"],
                    }
                )

    # FDR within each (family, epoch) — ALL category is the primary endpoint.
    for family in {r["family"] for r in planned_rows}:
        for epoch in [1, 3, 5]:
            idx = [
                i
                for i, r in enumerate(planned_rows)
                if r["family"] == family
                and r["epoch"] == epoch
                and r["category"] == "ALL"
                and np.isfinite(r["p"])
            ]
            qs = benjamini_hochberg([planned_rows[i]["p"] for i in idx])
            for i, q in zip(idx, qs):
                planned_rows[i]["q_fdr_ALL"] = q
            # also FDR across all categories (exploratory)
            idx_all = [
                i
                for i, r in enumerate(planned_rows)
                if r["family"] == family
                and r["epoch"] == epoch
                and np.isfinite(r["p"])
            ]
            qs = benjamini_hochberg([planned_rows[i]["p"] for i in idx_all])
            for i, q in zip(idx_all, qs):
                planned_rows[i]["q_fdr_all_cats"] = q

    # --- Family 2: category vs category within a model ---
    cat_rows: List[Dict[str, Any]] = []
    focus_models = ["DE only", "EN-DE", "TR-DE", "ZH-DE"]
    # Only categories within one partition are disjoint, hence independent.
    cat_pairs: List[Tuple[str, str]] = []
    for part in PARTITIONS[testset]:
        for pair in combinations([c for c in part if c in n_ex], 2):
            if pair not in cat_pairs:
                cat_pairs.append(pair)
    for epoch in [1, 3, 5]:
        for label in focus_models:
            for c1, c2 in cat_pairs:
                a = cells.get((label, epoch, c1))
                b = cells.get((label, epoch, c2))
                if a is None or b is None:
                    continue
                n1, n2 = n_ex[c1], n_ex[c2]
                k1 = float(np.mean(a) * n1)
                k2 = float(np.mean(b) * n2)
                z = two_prop_z(k1, n1, k2, n2)
                # seed-level Welch as a robustness check (same n_seeds)
                t = welch_t(a, b)
                cat_rows.append(
                    {
                        "epoch": epoch,
                        "model": label,
                        "cat_a": c1,
                        "cat_b": c2,
                        **z,
                        "seed_p": t["p"],
                        "seed_delta_pp": t["delta_pp"],
                    }
                )

    for epoch in [1, 3, 5]:
        for label in focus_models:
            idx = [
                i
                for i, r in enumerate(cat_rows)
                if r["epoch"] == epoch and r["model"] == label and np.isfinite(r["p"])
            ]
            qs = benjamini_hochberg([cat_rows[i]["p"] for i in idx])
            for i, q in zip(idx, qs):
                cat_rows[i]["q_fdr"] = q

    # --- Summaries ---
    def sig_count(rows: List[Dict[str, Any]], epoch: int, family: str, cat: str = "ALL") -> Dict[str, Any]:
        sub = [
            r
            for r in rows
            if r["epoch"] == epoch
            and r["family"] == family
            and r["category"] == cat
            and np.isfinite(r.get("p", np.nan))
        ]
        n = len(sub)
        n_sig = sum(1 for r in sub if r["p"] < args.alpha)
        n_q = sum(1 for r in sub if r.get("q_fdr_ALL", 1) < args.alpha)
        n_eq1 = sum(1 for r in sub if r.get("eq_1pp"))
        n_eq2 = sum(1 for r in sub if r.get("eq_2pp"))
        return {
            "n_tests": n,
            "n_p_lt_05": n_sig,
            "n_q_lt_05": n_q,
            "n_equivalent_1pp": n_eq1,
            "n_equivalent_2pp": n_eq2,
        }

    summary = {
        "n_examples": n_ex,
        "metric": "preference accuracy (PPL_gold < PPL_corrupt)",
        "test_models": "Welch t-test on 5 seed accuracies (unpaired)",
        "test_categories": "two-proportion z-test on mean n_correct (independent examples)",
        "alpha": args.alpha,
        "tost_deltas_pp": [1, 2],
        "counts": {
            str(ep): {
                fam: sig_count(planned_rows, ep, fam)
                for fam in ["vs_DE_only", "L1_vs_L1", "regular_vs_control"]
            }
            for ep in [1, 3, 5]
        },
    }

    # Print a human report focused on ALL, epochs 1/3/5.
    print("=" * 88)
    print("What a cell is")
    print("=" * 88)
    src, tgt = ("dative", "accusative") if testset == "dat2acc" else ("accusative", "dative")
    print(
        "Mean over seeds of preference accuracy: the model assigns lower PPL to the\n"
        f"gold {src} sentence than to its {tgt}-corrupted pair.\n"
        f"Test set sizes: {n_ex}"
    )
    print()

    print("=" * 88)
    print(f"Tables (accuracy %, mean over seeds)  |  {testset}")
    print("=" * 88)
    header = f"{'model':<28} " + " ".join(f"{c[:12]:>12}" for c in categories)
    for epoch in epochs:
        print(f"\n--- epoch {epoch} ---")
        print(header)
        for row in tables[str(epoch)]:
            vals = []
            for c in categories:
                v = row[c]
                vals.append(f"{v:12.2f}" if v is not None else f"{'—':>12}")
            print(f"{row['model']:<28} " + " ".join(vals))

    print("\n" + "=" * 88)
    print("FAMILY 1 — Models (seed-level Welch t, primary endpoint = ALL)")
    print("=" * 88)
    for epoch in [1, 3, 5]:
        print(f"\n### Epoch {epoch}  (ALL, n={n_ex['ALL']})")
        for fam, title in [
            ("vs_DE_only", "Each condition vs DE-only"),
            ("L1_vs_L1", "EN vs TR vs ZH (regular)"),
            ("regular_vs_control", "Regular vs shuffled / reset"),
        ]:
            print(f"\n  {title}")
            print(
                f"  {'A':<28} {'B':<28} {'Δpp':>7} {'p':>8} {'q_FDR':>8} "
                f"{'d':>6} {'eq1pp':>6} {'eq2pp':>6}"
            )
            sub = [
                r
                for r in planned_rows
                if r["epoch"] == epoch
                and r["family"] == fam
                and r["category"] == "ALL"
            ]
            sub.sort(key=lambda r: (r["p"] if np.isfinite(r["p"]) else 9))
            for r in sub:
                q = r.get("q_fdr_ALL", float("nan"))
                star = "*" if r["p"] < args.alpha else " "
                print(
                    f"  {r['model_a']:<28} {r['model_b']:<28} "
                    f"{r['delta_pp']:7.2f} {fmt_p(r['p']):>8}{star} {fmt_p(q):>8} "
                    f"{r['cohens_d']:6.2f} "
                    f"{'yes' if r['eq_1pp'] else 'no':>6} "
                    f"{'yes' if r['eq_2pp'] else 'no':>6}"
                )
            c = summary["counts"][str(epoch)][fam]
            print(
                f"  → {c['n_p_lt_05']}/{c['n_tests']} raw p<0.05; "
                f"{c['n_q_lt_05']}/{c['n_tests']} FDR q<0.05; "
                f"{c['n_equivalent_1pp']} equivalent at 1pp, "
                f"{c['n_equivalent_2pp']} at 2pp"
            )

    print("\n" + "=" * 88)
    print("FAMILY 2 — Categories (two-proportion z, same model)")
    print("=" * 88)
    print(
        "Only categories that partition the items are compared (they are different\n"
        "examples, hence independent); an aggregate is never compared with its parts."
    )
    for epoch in [5]:
        for label in focus_models:
            sub = [
                r
                for r in cat_rows
                if r["epoch"] == epoch and r["model"] == label
            ]
            n_sig = sum(1 for r in sub if r.get("q_fdr", 1) < args.alpha)
            print(f"\n  {label} epoch {epoch}: {n_sig}/{len(sub)} category pairs FDR-significant")
            # show the smallest and largest gaps
            sub_sorted = sorted(sub, key=lambda r: abs(r["delta_pp"]), reverse=True)
            print("  largest gaps:")
            for r in sub_sorted[:3]:
                print(
                    f"    {r['cat_a']:<16} vs {r['cat_b']:<16} "
                    f"Δ={r['delta_pp']:+6.1f}pp  p={fmt_p(r['p'])}  q={fmt_p(r.get('q_fdr', np.nan))}"
                )
            close = [r for r in sub if abs(r["delta_pp"]) < 2]
            if close:
                print("  gaps < 2 pp:")
                for r in close:
                    print(
                        f"    {r['cat_a']:<16} vs {r['cat_b']:<16} "
                        f"Δ={r['delta_pp']:+6.1f}pp  p={fmt_p(r['p'])}  q={fmt_p(r.get('q_fdr', np.nan))}"
                    )

    # Core (non-prepositional) argument categories vs DE-only.
    print("\n" + "=" * 88)
    print(f"Core-argument categories (n={'/'.join(str(n_ex[c]) for c in CORE_CATEGORIES)}) vs DE-only")
    print("=" * 88)
    for core_cat in CORE_CATEGORIES:
        print(f"\n -- category: {core_cat} --")
        for epoch in [1, 3, 5]:
            print(f"\n  epoch {epoch}")
            for other in L1_REGULAR:
                r = next(
                    (
                        x
                        for x in planned_rows
                        if x["epoch"] == epoch
                        and x["category"] == core_cat
                        and x["model_a"] == other
                        and x["family"] == "vs_DE_only"
                    ),
                    None,
                )
                if r:
                    print(
                        f"    {other:<8} {r['mean_a']*100:5.2f}% vs DE {r['mean_b']*100:5.2f}%  "
                        f"Δ={r['delta_pp']:+5.2f}pp  p={fmt_p(r['p'])}  d={r['cohens_d']:.2f}"
                    )

    return {
        "summary": summary,
        "tables_percent": tables,
        "planned_contrasts": planned_rows,
        "category_pairs": cat_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=DEFAULT_CACHE, help="build_analysis_data.py output dir")
    parser.add_argument(
        "--out_json",
        default=os.path.join(REPO_ROOT, "eval_results", "significance_summary.json"),
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--testset", choices=["both", *TESTSETS], default="both", help="test set(s) to analyse"
    )
    args = parser.parse_args()

    cache = Cache(args.cache)
    names = list(TESTSETS) if args.testset == "both" else [args.testset]
    payload = {name: analyse(args, cache, name) for name in names}

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
