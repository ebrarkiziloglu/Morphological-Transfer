"""
Aggregate the per-checkpoint eval outputs (``per_item.csv``) under
``eval_results/v3`` into one compact bundle for the analysis report.

Output (``--out``, default ``eval_results/analysis_cache/bundle.npz`` + ``meta.json``):

* ``meta.json``  -- systems, epochs, groups, metric names, per-item metadata.
* ``bundle.npz`` -- ``grp`` (group-level per-seed metrics) and the per-item
  seed-count arrays used by the item explorer.

A *system* is an (L1 corpus, phase-2 condition) pair, e.g. ``TRN|reset``.
A *group* is a subset of test items (category, pronoun / word-count / span-length
split, ...) defined once per item from the test-set metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

TESTSETS = ["dative_accusative_pairs", "accusative_dative_pairs"]
TESTSET_LABEL = {"dative_accusative_pairs": "dat→acc", "accusative_dative_pairs": "acc→dat"}
EPOCHS = [0, 1, 2, 3, 4, 5]
ITEM_EPOCHS = [1, 3, 5]
MAX_SEEDS = 5

# metric name -> description (order = last-but-one axis of ``grp``)
GROUP_METRICS = [
    "acc",        # sentence mean-NLL preference (the original metric)
    "acc_tot",    # total log-prob preference (= gold-length normalised)
    "acc_span",   # per-token surprisal from the edit start to the sentence end (span + everything after)
    "surp_g",     # token-weighted mean surprisal, gold sentence
    "surp_c",     # token-weighted mean surprisal, corrupted sentence
    "sspan_g",    # token-weighted mean surprisal on the gold span
    "sspan_c",    # token-weighted mean surprisal on the corrupted span
    "marg_gn",    # mean margin_goldnorm: (lp_gold - lp_corrupt) / n_tokens_gold, in nats
]
ITEM_METRICS = ["acc", "acc_tot", "acc_span"]

# Categories come from the test-set JSONs (joined by example_idx), not from the
# ``category`` column of per_item.csv, which holds whatever labels the test set had
# when the checkpoint was evaluated.
CORE_CATS = ["core_pronoun", "core_other"]
DE_PREP_CATS = {  # German preposition classes: dat→acc / acc→dat
    0: ["dative_only_prep", "two_way_prep", "other_prep"],
    1: ["accusative_only_prep", "two_way_prep", "other_prep"],
}
TESTSET_JSON = {  # test set -> JSON stem holding `category` and `is_pronoun`
    "dative_accusative_pairs": "dative_accusative_pairs",
    "accusative_dative_pairs": "accusative_dative_pairs",
}

L1_LABEL = {
    "DE": "German only", "EN": "English", "TR": "Turkish", "ZH": "Chinese",
    "TRS": "Stripped Case", "TRP": "Stripped Postp",
    "TRNI": "Stripped Nominal Infl", "TRN": "Stripped Nominal",
}
L1_ORDER = ["DE", "EN", "TR", "ZH", "TRS", "TRP", "TRNI", "TRN"]
VARIANT_DIRS = {
    "stripped": "TRS", "stripped_case": "TRS", "stripped_postp": "TRP",
    "stripped_nominal_infl": "TRNI", "stripped_nominal": "TRN",
}
COND_ORDER = ["regular", "reset", "shuffled"]


# ---------------------------------------------------------------------------
# Path -> system
# ---------------------------------------------------------------------------


def parse_checkpoint(rel: str) -> Dict[str, Any] | None:
    """``char-tr-de/phase_2/stripped_nominal/regular/de_nllb/reset_embeddings/ep3/seed_1``."""
    parts = rel.split("/")
    m_ep = [int(re.fullmatch(r"ep(\d+)", p).group(1)) for p in parts if re.fullmatch(r"ep(\d+)", p)]
    m_sd = [int(re.fullmatch(r"seed_(\d+)", p).group(1)) for p in parts if re.fullmatch(r"seed_(\d+)", p)]
    epoch = m_ep[0] if m_ep else 0
    seed = m_sd[0] if m_sd else 0
    lang = parts[0].split("-")[1].upper()  # char-tr-de -> TR
    if lang == "DE":
        # German-only baselines, one per phase-2 configuration. Phase 1 is German
        # (regular or word-shuffled corpus); phase_1/<corpus>/ep5 is the epoch-0
        # point of the matching phase_2 runs.
        if parts[1] == "phase_1":
            if parts[2] == "regular":  # shared by the regular and reset-embedding runs
                return dict(l1="DE", cond="regular", epoch=0, seed=seed, also_cond="reset")
            return dict(l1="DE", cond="shuffled", epoch=0, seed=seed) if parts[2] == "shuffled" else None
        if parts[1:5] == ["phase_2", "regular", "de_nllb", "regular"]:
            return dict(l1="DE", cond="regular", epoch=epoch, seed=seed)
        if parts[1:5] == ["phase_2", "regular", "de_nllb", "reset_embed"]:
            return dict(l1="DE", cond="reset", epoch=epoch, seed=seed)
        if parts[1:5] == ["phase_2", "shuffled", "de_nllb", "regular"]:
            return dict(l1="DE", cond="shuffled", epoch=epoch, seed=seed)
        return None
    if "phase_1" in parts:
        # L1-only model (before any German training) = epoch-0 point of the phase-2 runs:
        # char-<l1>-de/phase_1/[<variant>/]<regular|shuffled>/ep5/seed_N
        rest = [p for p in parts[parts.index("phase_1") + 1:]
                if not re.fullmatch(r"ep\d+|seed_\d+", p)]
        l1 = lang
        if rest and rest[0] in VARIANT_DIRS:
            l1 = VARIANT_DIRS[rest[0]]
            rest = rest[1:]
        if len(rest) != 1 or rest[0] not in ("regular", "shuffled"):
            return None
        if rest[0] == "shuffled":
            return dict(l1=l1, cond="shuffled", epoch=0, seed=seed)
        return dict(l1=l1, cond="regular", epoch=0, seed=seed, also_cond="reset")
    if "phase_2" not in parts:
        return None
    rest = [p for p in parts[parts.index("phase_2") + 1:]
            if not re.fullmatch(r"ep\d+|seed_\d+", p) and p != "de_nllb"]
    if lang == "DE":
        return dict(l1="DE", cond="regular", epoch=epoch, seed=seed)
    l1 = lang
    if rest and rest[0] in VARIANT_DIRS:
        l1 = VARIANT_DIRS[rest[0]]
        rest = rest[1:]
    corpus, p2 = rest[0], rest[-1]
    if corpus == "shuffled":
        cond = "shuffled"
    elif p2 == "reset_embeddings":
        cond = "reset"
    elif p2 == "regular":
        cond = "regular"
    else:  # shifted / reset_embeddings_shifted etc. are not part of this report
        return None
    return dict(l1=l1, cond=cond, epoch=epoch, seed=seed)


# ---------------------------------------------------------------------------
# Item metadata + groups
# ---------------------------------------------------------------------------


def span_bin(n: int) -> str:
    if n <= 3:
        return "sl:1-3"
    if n <= 6:
        return "sl:4-6"
    if n <= 10:
        return "sl:7-10"
    if n <= 20:
        return "sl:11-20"
    return "sl:21+"


def span_start(gold: str, bad: str) -> int:
    """Char offset where the (word-widened) edited region starts (see evaluate_ppl.changed_span)."""
    pre = len(os.path.commonprefix([gold, bad]))
    return gold.rfind(" ", 0, pre) + 1


def _load_json_records(testset_dir: str, stem: str, gold: pd.Series, corrupt: pd.Series) -> List[Dict[str, Any]]:
    """Test-set records (JSON order == example_idx), checked against the evaluated texts."""
    with open(os.path.join(testset_dir, f"{stem}.json"), encoding="utf-8") as f:
        recs = json.load(f)
    if len(recs) != len(gold):
        raise ValueError(f"{stem}: {len(recs)} test-set records vs {len(gold)} evaluated items")
    for i, (r, g, c) in enumerate(zip(recs, gold, corrupt)):
        if r["gold_text"] != g or r["corrupted"] != c:
            raise ValueError(f"{stem}: item {i} differs from the evaluated sentences; re-evaluate")
    return recs


def build_item_meta(
    ref: pd.DataFrame, testset_dir: str
) -> Tuple[pd.DataFrame, List[Tuple[str, str]], np.ndarray]:
    """Item table (both test sets stacked) + group definitions + membership matrix."""
    ref = ref.copy()
    ref["ts"] = ref["testset"].map({t: i for i, t in enumerate(TESTSETS)})
    ref = ref.sort_values(["ts", "example_idx"]).reset_index(drop=True)
    ref["category"] = "NaN"
    ref["pron"] = False  # NP head is UD PRON (from the test-set JSON)
    for t, ts_name in enumerate(TESTSETS):
        rows = ref["ts"] == t
        sub = ref[rows]
        recs = _load_json_records(testset_dir, TESTSET_JSON[ts_name], sub["gold_text"], sub["corrupt_text"])
        ref.loc[rows, "category"] = [str(r.get("category") or "NaN") for r in recs]
        ref.loc[rows, "pron"] = [bool(r["is_pronoun"]) for r in recs]
    gst = ref["group_span_text"].fillna("").astype(str)
    ref["n_words"] = gst.str.split().str.len().clip(lower=1)
    ref["span_start"] = [span_start(g, c) for g, c in zip(ref["gold_text"], ref["corrupt_text"])]
    ref["span_chars_g"] = ref["span_gold"].str.len()
    ref["span_chars_c"] = ref["span_corrupt"].str.len()
    ref["span_delta"] = ref["span_chars_c"] - ref["span_chars_g"]

    groups: List[Tuple[str, str]] = [("ALL", "All items")]
    masks: List[np.ndarray] = [np.ones(len(ref), bool)]
    ts = ref["ts"].to_numpy()
    cat = ref["category"]

    def add(key: str, label: str, mask: np.ndarray) -> None:
        if mask.sum() > 0:
            groups.append((key, label))
            masks.append(mask)

    core = cat.isin(CORE_CATS).to_numpy()
    add("core", "Core arguments (all)", core)
    for c in CORE_CATS:
        add(f"cat:{c}", c, (cat == c).to_numpy())
    add("prep", "Prepositional (all)", ((ts == 0) & cat.isin(DE_PREP_CATS[0]).to_numpy()) | ((ts == 1) & cat.isin(DE_PREP_CATS[1]).to_numpy()))
    for t, cats_t in DE_PREP_CATS.items():
        for c in cats_t:
            key = f"de:{c}"
            m = ((ts == t) & (cat == c).to_numpy())
            if key in [g[0] for g in groups]:
                masks[[g[0] for g in groups].index(key)] |= m
            else:
                add(key, c, m)
    add("pron", "Pronoun span (UD PRON)", ref["pron"].to_numpy())
    add("nonpron", "Non-pronoun span", ~ref["pron"].to_numpy())
    for k, lab in [(1, "1 word"), (2, "2 words"), (3, "3 words"), (4, "4 words")]:
        add(f"w:{k}", f"Span group = {lab}", (ref["n_words"] == k).to_numpy())
    add("w:5+", "Span group = 5+ words", (ref["n_words"] >= 5).to_numpy())
    sb = ref["span_chars_g"].map(span_bin)
    for k in ["sl:1-3", "sl:4-6", "sl:7-10", "sl:11-20", "sl:21+"]:
        add(k, f"Edited span {k[3:]} chars (gold)", (sb == k).to_numpy())
    add("dl:shorter", "Corrupt span shorter than gold", (ref["span_delta"] < 0).to_numpy())
    add("dl:equal", "Corrupt span same length as gold", (ref["span_delta"] == 0).to_numpy())
    add("dl:longer", "Corrupt span longer than gold", (ref["span_delta"] > 0).to_numpy())
    # core x word-count style splits for dat→acc handled by the category groups.
    return ref, groups, np.stack(masks, axis=1)


# ---------------------------------------------------------------------------
# Per-checkpoint worker
# ---------------------------------------------------------------------------

_USECOLS = [
    "testset", "example_idx", "correct", "correct_goldnorm",
    "n_tokens_gold", "n_tokens_corrupt", "logprob_gold", "logprob_corrupt",
    "span_n_tokens_gold", "span_n_tokens_corrupt",
    "span_logprob_gold", "span_logprob_corrupt", "margin_goldnorm",
    "after_n_tokens_gold", "after_n_tokens_corrupt",
    "after_logprob_gold", "after_logprob_corrupt",
]

_STATE: Dict[str, Any] = {}


def _init(member: np.ndarray, n_groups: int) -> None:
    _STATE["M"] = member.astype(np.float64)  # (n_items, n_testsets * n_groups)
    _STATE["cnt"] = _STATE["M"].sum(0)
    _STATE["n_groups"] = n_groups


def _wmean(M: np.ndarray, num: np.ndarray, den: np.ndarray) -> np.ndarray:
    d = M.T @ den
    return np.where(d > 0, (M.T @ num) / np.where(d > 0, d, 1), np.nan)


def process_checkpoint(args: Tuple[str, str]) -> Dict[str, Any]:
    rel, path = args
    df = pd.read_csv(path, usecols=_USECOLS, engine="pyarrow")
    df["ts"] = df["testset"].map({t: i for i, t in enumerate(TESTSETS)})
    df = df.sort_values(["ts", "example_idx"])
    M, cnt = _STATE["M"], _STATE["cnt"]

    correct = df["correct"].to_numpy(np.float64)
    corr_tot = df["correct_goldnorm"].to_numpy(np.float64)
    sg = df["span_logprob_gold"].to_numpy(np.float64)
    sc = df["span_logprob_corrupt"].to_numpy(np.float64)
    ng = df["span_n_tokens_gold"].to_numpy(np.float64)
    nc = df["span_n_tokens_corrupt"].to_numpy(np.float64)
    ok = np.isfinite(sg) & np.isfinite(sc) & (ng > 0) & (nc > 0)
    sg0, sc0 = np.where(ok, sg, 0.0), np.where(ok, sc, 0.0)

    # acc_span: per-token surprisal over everything that can differ, i.e. the
    # edited span plus the text after it (the prefix before the edit is
    # identical in both sentences by causality, but the continuation is not).
    ag = df["after_logprob_gold"].to_numpy(np.float64)
    ac = df["after_logprob_corrupt"].to_numpy(np.float64)
    ang = df["after_n_tokens_gold"].to_numpy(np.float64)
    anc = df["after_n_tokens_corrupt"].to_numpy(np.float64)
    tail_lg, tail_lc = sg + ag, sc + ac
    tail_ng, tail_nc = ng + ang, nc + anc
    ok_t = np.isfinite(tail_lg) & np.isfinite(tail_lc) & (tail_ng > 0) & (tail_nc > 0)
    tail_sg = -np.where(ok_t, tail_lg, 0.0) / np.maximum(tail_ng, 1)  # gold surprisal/char
    tail_sc = -np.where(ok_t, tail_lc, 0.0) / np.maximum(tail_nc, 1)
    # gold preferred if its surprisal is lower (positive = more surprising)
    span_c = np.where(ok_t, (tail_sg < tail_sc).astype(np.float64), np.nan)

    tg = df["n_tokens_gold"].to_numpy(np.float64)
    tc = df["n_tokens_corrupt"].to_numpy(np.float64)
    ones = np.ones_like(tg)
    okf = ok.astype(np.float64)

    cnt = np.where(cnt > 0, cnt, np.nan)
    out = np.full((M.shape[1], len(GROUP_METRICS)), np.nan, np.float64)
    out[:, 0] = (M.T @ correct) / cnt
    out[:, 1] = (M.T @ corr_tot) / cnt
    tp_ok = M.T @ ok_t.astype(np.float64)
    out[:, 2] = np.where(tp_ok > 0, (M.T @ np.where(ok_t, span_c, 0.0)) / np.where(tp_ok > 0, tp_ok, 1), np.nan)
    out[:, 3] = _wmean(M, -df["logprob_gold"].to_numpy(np.float64), tg)
    out[:, 4] = _wmean(M, -df["logprob_corrupt"].to_numpy(np.float64), tc)
    out[:, 5] = _wmean(M, -sg0, ng * okf)
    out[:, 6] = _wmean(M, -sc0, nc * okf)
    out[:, 7] = (M.T @ df["margin_goldnorm"].to_numpy(np.float64)) / cnt

    # per-item values for the explorer
    margin_span = np.where(ok_t, tail_sc - tail_sg, np.nan)
    return dict(
        rel=rel,
        group=out,
        item_acc=df["correct"].to_numpy(np.uint8),
        item_tot=df["correct_goldnorm"].to_numpy(np.uint8),
        item_span=np.where(ok_t, span_c, 255).astype(np.uint8),  # 255 = undefined
        item_margin=margin_span.astype(np.float32),
        n_items=len(df),
    )


def find_checkpoints(root: str) -> List[Tuple[str, str]]:
    found = []
    for dp, dn, fn in os.walk(root):
        if "per_item.csv" in fn and "overall.csv" in fn:
            rel = os.path.relpath(dp, root)
            found.append((rel, os.path.join(dp, "per_item.csv")))
    return sorted(found)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", default="eval_results/v3")
    ap.add_argument("--extra_roots", nargs="*", default=[],
                    help="further results roots (e.g. eval_results holding phase_1 evals); "
                         "processed after --results_root, so they override matching systems/epochs")
    ap.add_argument("--out_dir", default="eval_results/analysis_cache")
    ap.add_argument("--testset_dir", default="src/eval_dataset/testset")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ckpts = []
    for root_i, root in enumerate([args.results_root] + list(args.extra_roots)):
        for rel, path in find_checkpoints(root):
            parts = rel.split("/")
            # extra roots: only top-level char-*/phase_1 (epoch-0) checkpoints, so v3
            # phase_2 and unrelated folders (analysis_cache, v3/, ...) aren't touched
            if root_i > 0 and not (parts[0].startswith("char-") and "phase_1" in parts):
                continue
            info = parse_checkpoint(rel)
            if info is None:
                continue
            ckpts.append((f"{root_i}:{rel}", path, info))
    print(f"{len(ckpts)} checkpoints", flush=True)

    ref = pd.read_csv(
        ckpts[0][1], engine="pyarrow",
        usecols=["testset", "example_idx", "sent_id", "group_span_text", "preposition", "category",
                 "gold_text", "corrupt_text", "span_gold", "span_corrupt", "n_chars_gold",
                 "n_chars_corrupt"],
    )
    items, groups, member = build_item_meta(ref, args.testset_dir)
    n_items = len(items)
    print(f"{n_items} items, {len(groups)} groups", flush=True)

    # systems present
    sys_keys = sorted({(c[2]["l1"], c[2]["cond"]) for c in ckpts},
                      key=lambda t: (COND_ORDER.index(t[1]), L1_ORDER.index(t[0])))
    sys_idx = {k: i for i, k in enumerate(sys_keys)}
    n_sys = len(sys_keys)

    grp = np.full((n_sys, len(EPOCHS), len(TESTSETS), len(groups), len(GROUP_METRICS), MAX_SEEDS),
                  np.nan, np.float32)
    # Per-item seed-count arrays.  Undefined span counts tracked separately.
    shape = (n_sys, len(ITEM_EPOCHS), n_items)
    item_acc = np.zeros(shape, np.uint8)
    item_tot = np.zeros(shape, np.uint8)
    item_span = np.zeros(shape, np.uint8)
    item_span_n = np.zeros(shape, np.uint8)

    # Each testset gets its own block of group columns: (n_items, T * G).
    ts_arr = items["ts"].to_numpy()
    G = len(groups)
    member_all = np.hstack([member * (ts_arr == t)[:, None] for t in range(len(TESTSETS))])
    member_ts = [member * (ts_arr == t)[:, None] for t in range(len(TESTSETS))]

    info_by_rel = {rel: info for rel, _, info in ckpts}
    jobs = [(rel, path) for rel, path, _ in ckpts]
    margin_sum = np.zeros(shape, np.float64)
    seed_counts = np.zeros((n_sys, len(ITEM_EPOCHS)), np.int32)

    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(member_all, G)) as ex:
        for n, res in enumerate(ex.map(process_checkpoint, jobs, chunksize=4), 1):
            info = info_by_rel[res["rel"]]
            ei, seed = info["epoch"], info["seed"]
            if res["n_items"] != n_items or seed >= MAX_SEEDS:
                print("skip", res["rel"], res["n_items"], flush=True)
                continue
            g = res["group"].astype(np.float32).reshape(len(TESTSETS), G, len(GROUP_METRICS))
            for cond in [info["cond"]] + ([info["also_cond"]] if "also_cond" in info else []):
                si = sys_idx[(info["l1"], cond)]
                grp[si, ei, :, :, :, seed] = g
                if ei not in ITEM_EPOCHS:
                    continue
                k = ITEM_EPOCHS.index(ei)
                item_acc[si, k] += res["item_acc"]
                item_tot[si, k] += res["item_tot"]
                sp = res["item_span"]
                defined = sp != 255
                item_span[si, k] += np.where(defined, sp, 0).astype(np.uint8)
                item_span_n[si, k] += defined.astype(np.uint8)
                margin_sum[si, k] += np.nan_to_num(res["item_margin"])
                seed_counts[si, k] += 1
            if n % 50 == 0:
                print(f"  {n}/{len(jobs)}", flush=True)

    item_margin = (margin_sum / np.maximum(seed_counts, 1)[:, :, None]).astype(np.float32)

    meta = dict(
        testsets=TESTSETS,
        testset_labels=[TESTSET_LABEL[t] for t in TESTSETS],
        epochs=EPOCHS,
        item_epochs=ITEM_EPOCHS,
        group_metrics=GROUP_METRICS,
        item_metrics=ITEM_METRICS,
        groups=[dict(key=k, label=l) for k, l in groups],
        group_n=[[int((member_ts[t][:, gi]).sum()) for gi in range(len(groups))]
                 for t in range(len(TESTSETS))],
        systems=[dict(l1=l1, cond=c, label=L1_LABEL[l1]) for (l1, c) in sys_keys],
        seed_counts=seed_counts.tolist(),
        l1_label=L1_LABEL, l1_order=L1_ORDER, cond_order=COND_ORDER,
        items=dict(
            ts=items["ts"].tolist(),
            sent_id=items["sent_id"].fillna("").tolist(),
            category=items["category"].tolist(),
            prep=items["preposition"].fillna("").astype(str).tolist(),
            group_span=items["group_span_text"].fillna("").tolist(),
            n_words=items["n_words"].tolist(),
            pron=items["pron"].astype(int).tolist(),
            span_gold=items["span_gold"].fillna("").tolist(),
            span_corrupt=items["span_corrupt"].fillna("").tolist(),
            span_start=items["span_start"].tolist(),
            gold=items["gold_text"].tolist(),
            corrupt=items["corrupt_text"].tolist(),
        ),
    )
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    np.savez_compressed(
        os.path.join(args.out_dir, "bundle.npz"),
        grp=grp, item_acc=item_acc, item_tot=item_tot, item_span=item_span,
        item_span_n=item_span_n, item_margin=item_margin,
    )
    print("wrote", args.out_dir, flush=True)


if __name__ == "__main__":
    main()
