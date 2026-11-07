"""
Evaluate a pretrained causal LM on gold vs. corrupted sentence pairs (JSON).

For each example, compute perplexity of `gold_text` and the corrupted field.
The model "prefers" the sentence with lower perplexity. If PPL(gold) <= PPL(corrupt),
that counts as a correct preference (ties favor gold).

Supports:
  1) Single checkpoint mode (``--model_path``).
  2) Multi-seed / multi-epoch sweep (``--model_root``): evaluates checkpoints under
     ``{model_root}/{phase_subdir}/ep{k}/seed_{s}/`` (or seedless ``ep{k}/`` when
     that directory is itself a checkpoint, e.g. ep0). Loads the tokenizer once
     (``--tokenizer_path``, else ``model_root``, else the first checkpoint) and
     reuses it for every model. Writes per-seed and seed-aggregated CSV files,
     and (when present) per-``category`` breakdowns.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import sys
from collections import defaultdict
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def bootstrap_ci_mean(
    x: np.ndarray,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean of ``x``."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0 or n_boot <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = float(np.mean(x[idx]))
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def checkpoint_fingerprint(model_path: str) -> str:
    """Short SHA256 prefix of weight file so logs prove which checkpoint ran."""
    for name in ("model.safetensors", "pytorch_model.bin"):
        p = os.path.join(model_path, name)
        if os.path.isfile(p):
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            return f"{name} sha256={h.hexdigest()[:16]}"
    return "no model.safetensors/pytorch_model.bin found"


def normalize_category(raw: Any) -> str:
    """Map missing / null / NaN category labels to the string ``NaN``."""
    if raw is None:
        return "NaN"
    if isinstance(raw, float) and math.isnan(raw):
        return "NaN"
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return "NaN"
    return s


def load_json_dataset(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON root must be a list of objects")
    return data


def parse_int_list(value: str) -> List[int]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return [int(p) for p in parts]


def sentence_nll_and_ppl(
    model: Any,
    tokenizer: AutoTokenizer,
    text: str,
    device: torch.device,
    max_length: int,
    token_logprobs_out: Optional[List[Optional[np.ndarray]]] = None,
) -> Tuple[float, float, int]:
    """Return (mean NLL, perplexity, n_predicted_tokens) for a single string.

    If ``token_logprobs_out`` is given, one entry is appended: the per-token
    log-probs (length ``n_predicted_tokens``) when tokens map 1:1 to characters
    (entry ``j`` scores character ``j + 1``), else ``None``.
    """
    if not text or not text.strip():
        return float("nan"), float("nan"), 0

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    input_ids = enc["input_ids"].to(device)
    if input_ids.size(1) < 2:
        return float("nan"), float("nan"), 0

    with torch.no_grad():
        out = model(input_ids=input_ids, labels=input_ids)
    loss = out.loss
    if loss is None or torch.isnan(loss):
        return float("nan"), float("nan"), 0

    nll = loss.item()
    if token_logprobs_out is not None:
        if input_ids.size(1) == len(text):
            lp = torch.log_softmax(out.logits[0, :-1].float(), dim=-1)
            lp = lp.gather(1, input_ids[0, 1:, None])[:, 0]
            token_logprobs_out.append(lp.cpu().numpy().astype(np.float64))
        else:
            token_logprobs_out.append(None)
    # The causal LM loss averages over seq_len - 1 predicted tokens.
    return nll, math.exp(nll), int(input_ids.size(1) - 1)


def changed_span(gold: str, bad: str) -> Tuple[int, int, int]:
    """Locate the edited region between two sentences, widened to word edges.

    Returns ``(start, end_gold, end_bad)``: char offsets such that
    ``gold[:start] == bad[:start]`` and ``gold[end_gold:] == bad[end_bad:]``.
    """
    pre = len(os.path.commonprefix([gold, bad]))
    suf = len(os.path.commonprefix([gold[::-1], bad[::-1]]))
    suf = min(suf, len(gold) - pre, len(bad) - pre)
    start = gold.rfind(" ", 0, pre) + 1
    ends = []
    for t, e in ((gold, len(gold) - suf), (bad, len(bad) - suf)):
        sp = t.find(" ", e)
        ends.append(len(t) if sp < 0 else sp)
    return start, ends[0], ends[1]


def _span_lp(lp: Optional[np.ndarray], a: int, b: int) -> Tuple[float, int]:
    """Total log-prob of characters ``[a, b)`` (char 0 is never predicted)."""
    if lp is None:
        return float("nan"), 0
    a = max(a, 1)
    if b <= a:
        return 0.0, 0
    return float(lp[a - 1 : b - 1].sum()), b - a


def load_tokenizer(tokenizer_path: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        pad = getattr(tokenizer, "eos_token", None) or getattr(
            tokenizer, "cls_token", None
        )
        if pad is None:
            raise ValueError("Tokenizer needs pad, eos, or cls token")
        tokenizer.pad_token = pad
    return tokenizer


def load_model(model_path: str, device: torch.device) -> Any:
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()
    model.to(device)
    return model


def load_model_and_tokenizer(
    model_path: str,
    tokenizer_path: Optional[str],
    device: torch.device,
) -> Tuple[Any, AutoTokenizer]:
    tokenizer = load_tokenizer(tokenizer_path or model_path)
    model = load_model(model_path, device)
    return model, tokenizer


def unload_model(model: Any) -> None:
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _bucket_stats(
    correct: np.ndarray,
    margin_arr: np.ndarray,
    pg: np.ndarray,
    pc: np.ndarray,
    ntok_g: Optional[np.ndarray] = None,
    ntok_c: Optional[np.ndarray] = None,
    nll_g: Optional[np.ndarray] = None,
    nll_c: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    n = int(correct.size)
    stats = {
        "n_examples": n,
        "n_correct": int(np.sum(correct)) if n else 0,
        "accuracy": float(np.mean(correct)) if n else float("nan"),
        "mean_ppl_gold": float(np.mean(pg)) if n else float("nan"),
        "mean_ppl_corrupt": float(np.mean(pc)) if n else float("nan"),
        "mean_margin_nll": float(np.mean(margin_arr)) if n else float("nan"),
    }
    if ntok_g is not None and n:
        # Token-weighted: total NLL over all tokens / total token count.
        stats["mean_surprisal_gold"] = float(np.sum(nll_g * ntok_g) / np.sum(ntok_g))
        stats["mean_surprisal_corrupt"] = float(
            np.sum(nll_c * ntok_c) / np.sum(ntok_c)
        )
        stats["n_tokens_gold"] = int(np.sum(ntok_g))
        stats["n_tokens_corrupt"] = int(np.sum(ntok_c))
        # Gold-length normalization: total NLL of each sentence divided by the
        # gold sentence's token count, so both sides share one denominator.
        sum_g = nll_g * ntok_g
        sum_c = nll_c * ntok_c
        margin_gn = (sum_c - sum_g) / ntok_g
        stats["mean_surprisal_corrupt_goldnorm"] = float(
            np.sum(sum_c) / np.sum(ntok_g)
        )
        stats["mean_margin_goldnorm"] = float(np.mean(margin_gn))
        stats["accuracy_goldnorm"] = float(np.mean(margin_gn > 0))
    else:
        stats["mean_surprisal_corrupt_goldnorm"] = float("nan")
        stats["mean_margin_goldnorm"] = float("nan")
        stats["accuracy_goldnorm"] = float("nan")
        stats["mean_surprisal_gold"] = float("nan")
        stats["mean_surprisal_corrupt"] = float("nan")
        stats["n_tokens_gold"] = 0
        stats["n_tokens_corrupt"] = 0
    return stats


def metrics_by_category(
    categories: Sequence[str],
    y_pred: np.ndarray,
    margins: np.ndarray,
    ppl_gold: np.ndarray,
    ppl_corrupt: np.ndarray,
    nll_gold: Optional[np.ndarray] = None,
    nll_corrupt: Optional[np.ndarray] = None,
    ntok_gold: Optional[np.ndarray] = None,
    ntok_corrupt: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, float]]:
    """Aggregate accuracy / PPL / margin / per-token surprisal per category, plus ``ALL``."""
    have_tok = ntok_gold is not None
    if not have_tok:
        nll_gold = nll_corrupt = ntok_gold = ntok_corrupt = np.zeros(len(y_pred))
    buckets: DefaultDict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {
            "correct": [], "margin": [], "ppl_gold": [], "ppl_corrupt": [],
            "nll_g": [], "nll_c": [], "ntok_g": [], "ntok_c": [],
        }
    )
    for cat, pred, margin, pg, pc, ng, nc, tg, tc in zip(
        categories, y_pred, margins, ppl_gold, ppl_corrupt,
        nll_gold, nll_corrupt, ntok_gold, ntok_corrupt,
    ):
        buckets[cat]["correct"].append(float(pred == 1))
        buckets[cat]["margin"].append(float(margin))
        buckets[cat]["ppl_gold"].append(float(pg))
        buckets[cat]["ppl_corrupt"].append(float(pc))
        buckets[cat]["nll_g"].append(float(ng))
        buckets[cat]["nll_c"].append(float(nc))
        buckets[cat]["ntok_g"].append(float(tg))
        buckets[cat]["ntok_c"].append(float(tc))

    by_cat: Dict[str, Dict[str, float]] = {}

    # Overall = all categories combined (including NaN)
    y_pred_a = np.asarray(y_pred, dtype=np.float64)
    by_cat["ALL"] = _bucket_stats(
        (y_pred_a == 1).astype(np.float64),
        np.asarray(margins, dtype=np.float64),
        np.asarray(ppl_gold, dtype=np.float64),
        np.asarray(ppl_corrupt, dtype=np.float64),
        *(
            (
                np.asarray(ntok_gold, dtype=np.float64),
                np.asarray(ntok_corrupt, dtype=np.float64),
                np.asarray(nll_gold, dtype=np.float64),
                np.asarray(nll_corrupt, dtype=np.float64),
            )
            if have_tok
            else ()
        ),
    )

    for cat in sorted(buckets.keys(), key=lambda c: (c == "NaN", c)):
        by_cat[cat] = _bucket_stats(
            np.asarray(buckets[cat]["correct"], dtype=np.float64),
            np.asarray(buckets[cat]["margin"], dtype=np.float64),
            np.asarray(buckets[cat]["ppl_gold"], dtype=np.float64),
            np.asarray(buckets[cat]["ppl_corrupt"], dtype=np.float64),
            *(
                (
                    np.asarray(buckets[cat]["ntok_g"], dtype=np.float64),
                    np.asarray(buckets[cat]["ntok_c"], dtype=np.float64),
                    np.asarray(buckets[cat]["nll_g"], dtype=np.float64),
                    np.asarray(buckets[cat]["nll_c"], dtype=np.float64),
                )
                if have_tok
                else ()
            ),
        )
    return by_cat


def format_category_report(by_cat: Dict[str, Dict[str, float]]) -> str:
    if not by_cat:
        return ""
    header = (
        f"{'category':<16} {'n':>6} {'correct':>8} {'accuracy':>10} "
        f"{'mean_ppl_g':>12} {'mean_ΔNLL':>12} {'surpr_g/tok':>12}"
    )
    lines = [header, "-" * len(header)]

    def _sort_key(cat: str) -> Tuple[int, str]:
        if cat == "ALL":
            return (0, cat)
        if cat == "NaN":
            return (2, cat)
        return (1, cat)

    for cat in sorted(by_cat.keys(), key=_sort_key):
        stats = by_cat[cat]
        lines.append(
            f"{cat:<16} {int(stats['n_examples']):>6} "
            f"{int(stats['n_correct']):>8} {100 * stats['accuracy']:>9.2f}% "
            f"{stats['mean_ppl_gold']:>12.4f} {stats['mean_margin_nll']:>12.6f} "
            f"{stats['mean_surprisal_gold']:>12.6f}"
        )
    return "\n".join(lines)


def summarize_predictions(
    y_pred: np.ndarray,
    ppl_gold: np.ndarray,
    ppl_corrupt: np.ndarray,
    nll_gold: np.ndarray,
    nll_corrupt: np.ndarray,
    categories: Sequence[str],
    skipped: int,
    fingerprint: str,
    bootstrap_n: int = 0,
    bootstrap_seed: int = 42,
    bootstrap_alpha: float = 0.05,
    ntok_gold: Optional[np.ndarray] = None,
    ntok_corrupt: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    n = int(y_pred.size)
    if n == 0:
        raise RuntimeError(f"No valid examples (skipped={skipped}).")

    y_true = np.ones(n, dtype=np.int32)
    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["prefer_corrupt", "prefer_gold"],
        zero_division=0,
    )
    margin = nll_corrupt - nll_gold
    correct = (margin > 0).astype(np.float64)

    if bootstrap_n > 0:
        ci_margin_lo, ci_margin_hi = bootstrap_ci_mean(
            margin, bootstrap_n, bootstrap_seed, bootstrap_alpha
        )
        ci_acc_lo, ci_acc_hi = bootstrap_ci_mean(
            correct, bootstrap_n, bootstrap_seed + 1, bootstrap_alpha
        )
    else:
        ci_margin_lo = ci_margin_hi = ci_acc_lo = ci_acc_hi = float("nan")

    by_category = metrics_by_category(
        categories, y_pred, margin, ppl_gold, ppl_corrupt,
        nll_gold, nll_corrupt, ntok_gold, ntok_corrupt,
    )
    overall = by_category["ALL"]

    return {
        "checkpoint_fingerprint": fingerprint,
        "n_examples": n,
        "n_skipped": skipped,
        "accuracy": acc,
        "precision_prefer_gold": float(prec),
        "recall_prefer_gold": float(rec),
        "f1_prefer_gold": float(f1),
        "mean_ppl_gold": float(np.mean(ppl_gold)),
        "mean_ppl_corrupt": float(np.mean(ppl_corrupt)),
        "mean_surprisal_gold": overall["mean_surprisal_gold"],
        "mean_surprisal_corrupt": overall["mean_surprisal_corrupt"],
        "mean_surprisal_corrupt_goldnorm": overall["mean_surprisal_corrupt_goldnorm"],
        "mean_margin_goldnorm": overall["mean_margin_goldnorm"],
        "accuracy_goldnorm": overall["accuracy_goldnorm"],
        "n_tokens_gold": overall["n_tokens_gold"],
        "n_tokens_corrupt": overall["n_tokens_corrupt"],
        "margin_nll_mean": float(np.mean(margin)),
        "margin_nll_median": float(np.median(margin)),
        "margin_nll_std": float(np.std(margin, ddof=0)),
        "bootstrap_n": int(bootstrap_n),
        "bootstrap_alpha": float(bootstrap_alpha),
        "bootstrap_ci_mean_margin": [ci_margin_lo, ci_margin_hi],
        "bootstrap_ci_accuracy": [ci_acc_lo, ci_acc_hi],
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "by_category": by_category,
        "n_with_category": sum(1 for c in categories if c != "NaN"),
    }


def score_examples(
    model: Any,
    tokenizer: AutoTokenizer,
    examples: Sequence[Dict[str, Any]],
    gold_key: str,
    corrupt_key: str,
    device: torch.device,
    max_length: int,
    category_key: str = "category",
    return_token_counts: bool = False,
    items_out: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Any, ...]:
    """Score all examples; return preds, ppls, nlls, categories, skipped.

    With ``return_token_counts`` the (gold, corrupt) predicted-token-count arrays
    are appended to the returned tuple. If ``items_out`` is given, one dict per
    scored example (raw log-probs, lengths, margins) is appended to it.
    """
    y_pred: List[int] = []
    ppl_gold_list: List[float] = []
    ppl_bad_list: List[float] = []
    nll_gold_list: List[float] = []
    nll_bad_list: List[float] = []
    ntok_gold_list: List[int] = []
    ntok_bad_list: List[int] = []
    categories: List[str] = []
    skipped = 0

    for ex_idx, ex in enumerate(examples):
        gold = ex.get(gold_key)
        bad = ex.get(corrupt_key)
        if not isinstance(gold, str) or not isinstance(bad, str):
            skipped += 1
            continue
        lps: Optional[List[Optional[np.ndarray]]] = [] if items_out is not None else None
        nll_1, ppl_1, nt_1 = sentence_nll_and_ppl(
            model, tokenizer, gold, device, max_length, token_logprobs_out=lps
        )
        nll_2, ppl_2, nt_2 = sentence_nll_and_ppl(
            model, tokenizer, bad, device, max_length, token_logprobs_out=lps
        )
        if math.isnan(ppl_1) or math.isnan(ppl_2):
            skipped += 1
            continue

        ppl_gold_list.append(ppl_1)
        ppl_bad_list.append(ppl_2)
        nll_gold_list.append(nll_1)
        nll_bad_list.append(nll_2)
        ntok_gold_list.append(nt_1)
        ntok_bad_list.append(nt_2)
        # Prefer gold when corrupt has strictly higher NLL (ties count as prefer_corrupt).
        y_pred.append(1 if nll_2 > nll_1 else 0)
        categories.append(normalize_category(ex.get(category_key)))

        if items_out is not None:
            sum_g, sum_c = nll_1 * nt_1, nll_2 * nt_2
            # Edited region (word-widened) and the text after it. Text before
            # it is identical, so its log-prob is identical by causality.
            a, eg, ec = changed_span(gold, bad)
            sp_g, spn_g = _span_lp(lps[0], a, eg)
            sp_c, spn_c = _span_lp(lps[1], a, ec)
            af_g, afn_g = _span_lp(lps[0], eg, len(gold))
            af_c, afn_c = _span_lp(lps[1], ec, len(bad))
            # Edit start to sentence end (span + after): the whole region whose
            # log-prob can differ between the two sentences.
            tail_g, tail_c = sp_g + af_g, sp_c + af_c
            tailn_g, tailn_c = spn_g + afn_g, spn_c + afn_c
            tail_ok = tailn_g > 0 and tailn_c > 0 and not (
                math.isnan(tail_g) or math.isnan(tail_c)
            )
            items_out.append(
                {
                    "example_idx": ex_idx,
                    "sent_id": ex.get("sent_id", ""),
                    "group_span_text": ex.get("group_span_text", ""),
                    "preposition": ex.get("preposition", ""),
                    "category": categories[-1],
                    "gold_text": gold,
                    "corrupt_text": bad,
                    "n_chars_gold": len(gold),
                    "n_chars_corrupt": len(bad),
                    "n_tokens_gold": nt_1,
                    "n_tokens_corrupt": nt_2,
                    "logprob_gold": -sum_g,
                    "logprob_corrupt": -sum_c,
                    "surprisal_per_tok_gold": nll_1,
                    "surprisal_per_tok_corrupt": nll_2,
                    "ppl_gold": ppl_1,
                    "ppl_corrupt": ppl_2,
                    "margin_per_tok": nll_2 - nll_1,
                    "margin_goldnorm": (sum_c - sum_g) / nt_1,
                    "span_gold": gold[a:eg],
                    "span_corrupt": bad[a:ec],
                    "span_n_tokens_gold": spn_g,
                    "span_n_tokens_corrupt": spn_c,
                    "span_logprob_gold": sp_g,
                    "span_logprob_corrupt": sp_c,
                    "span_margin": sp_g - sp_c,
                    "after_n_tokens_gold": afn_g,
                    "after_n_tokens_corrupt": afn_c,
                    "after_logprob_gold": af_g,
                    "after_logprob_corrupt": af_c,
                    "tail_n_tokens_gold": tailn_g,
                    "tail_n_tokens_corrupt": tailn_c,
                    "tail_logprob_gold": tail_g,
                    "tail_logprob_corrupt": tail_c,
                    # per-token surprisal on span + after, lower wins; None = undefined
                    "correct_tail": (
                        int(-tail_g / tailn_g < -tail_c / tailn_c) if tail_ok else ""
                    ),
                    "correct": int(nll_2 > nll_1),
                    "correct_goldnorm": int(sum_c > sum_g),
                }
            )

    result = (
        np.asarray(y_pred, dtype=np.int32),
        np.asarray(ppl_gold_list, dtype=np.float64),
        np.asarray(ppl_bad_list, dtype=np.float64),
        np.asarray(nll_gold_list, dtype=np.float64),
        np.asarray(nll_bad_list, dtype=np.float64),
        categories,
        skipped,
    )
    if return_token_counts:
        result += (
            np.asarray(ntok_gold_list, dtype=np.float64),
            np.asarray(ntok_bad_list, dtype=np.float64),
        )
    return result


def evaluate(
    model_path: str,
    tokenizer_path: Optional[str],
    json_path: str,
    gold_key: str,
    corrupt_key: str,
    split_filter: Optional[str],
    max_length: int,
    device: Optional[torch.device] = None,
    bootstrap_n: int = 0,
    bootstrap_seed: int = 42,
    bootstrap_alpha: float = 0.05,
    category_key: str = "category",
    model: Any = None,
    tokenizer: Optional[AutoTokenizer] = None,
) -> Dict[str, Any]:
    """Evaluate one checkpoint on one JSON dataset."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    owns_model = model is None
    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path, device)

    examples = load_json_dataset(json_path)
    if split_filter:
        examples = [e for e in examples if e.get("split") == split_filter]
        logger.info(f"Filtered to split={split_filter!r}: {len(examples)} examples")

    y_pred, ppl_g, ppl_c, nll_g, nll_c, cats, skipped, nt_g, nt_c = score_examples(
        model,
        tokenizer,
        examples,
        gold_key,
        corrupt_key,
        device,
        max_length,
        category_key=category_key,
        return_token_counts=True,
    )
    fp = checkpoint_fingerprint(model_path)
    out = summarize_predictions(
        y_pred,
        ppl_g,
        ppl_c,
        nll_g,
        nll_c,
        cats,
        skipped,
        fp,
        bootstrap_n=bootstrap_n,
        bootstrap_seed=bootstrap_seed,
        bootstrap_alpha=bootstrap_alpha,
        ntok_gold=nt_g,
        ntok_corrupt=nt_c,
    )

    if owns_model:
        unload_model(model)
    return out


# ---------------------------------------------------------------------------
# Multi-seed / multi-epoch sweep + CSV
# ---------------------------------------------------------------------------


def _is_checkpoint_dir(path: str) -> bool:
    """True if *path* looks like a HuggingFace model directory."""
    if not os.path.isdir(path):
        return False
    return any(
        os.path.isfile(os.path.join(path, name))
        for name in (
            "config.json",
            "model.safetensors",
            "pytorch_model.bin",
        )
    )


def _has_tokenizer(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "tokenizer.json")) or os.path.isfile(
        os.path.join(path, "tokenizer_config.json")
    )


def resolve_checkpoint(
    model_root: str,
    seed: int,
    epoch: int,
    phase_subdir: str = "phase_2_from_ep5",
) -> str:
    """
    Map logical epoch/seed -> checkpoint directory.

    Preferred layout::

        {model_root}/{phase_subdir}/ep{epoch}/seed_{seed}/

    Seedless fallback (e.g. a single ep0 baseline)::

        {model_root}/{phase_subdir}/ep{epoch}/

    when that directory itself contains model weights.
    """
    if epoch < 0:
        raise ValueError(f"epoch must be >= 0, got {epoch}")
    seeded = os.path.join(
        model_root, phase_subdir, f"ep{epoch}", f"seed_{seed}"
    )
    if _is_checkpoint_dir(seeded):
        return seeded
    epoch_dir = os.path.join(model_root, phase_subdir, f"ep{epoch}")
    if _is_checkpoint_dir(epoch_dir):
        return epoch_dir
    return seeded


def _testset_name(json_path: str) -> str:
    return os.path.splitext(os.path.basename(json_path))[0]


def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    logger.info(f"Wrote {len(rows)} rows -> {path}")


def aggregate_rows(
    rows: List[Dict[str, Any]],
    group_keys: Sequence[str],
    metric_keys: Sequence[str],
) -> List[Dict[str, Any]]:
    """Mean / std of metrics across seeds for each group key combination."""
    groups: DefaultDict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        groups[key].append(row)

    out: List[Dict[str, Any]] = []
    for key, items in sorted(groups.items(), key=lambda kv: kv[0]):
        agg: Dict[str, Any] = {k: v for k, v in zip(group_keys, key)}
        agg["n_seeds"] = len(items)
        seeds = sorted({int(r["seed"]) for r in items})
        agg["seeds"] = ",".join(str(s) for s in seeds)
        for mk in metric_keys:
            vals = np.asarray([float(r[mk]) for r in items], dtype=np.float64)
            agg[f"{mk}_mean"] = float(np.mean(vals))
            agg[f"{mk}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        # n_examples should be constant across seeds; keep first
        if "n_examples" in items[0]:
            agg["n_examples"] = items[0]["n_examples"]
        out.append(agg)
    return out


OVERALL_METRIC_KEYS = (
    "accuracy",
    "mean_ppl_gold",
    "mean_ppl_corrupt",
    "mean_surprisal_gold",
    "mean_surprisal_corrupt",
    "mean_surprisal_corrupt_goldnorm",
    "mean_margin_goldnorm",
    "accuracy_goldnorm",
    "margin_nll_mean",
    "f1_prefer_gold",
)

CATEGORY_METRIC_KEYS = (
    "accuracy",
    "mean_ppl_gold",
    "mean_ppl_corrupt",
    "mean_surprisal_gold",
    "mean_surprisal_corrupt",
    "mean_surprisal_corrupt_goldnorm",
    "mean_margin_goldnorm",
    "accuracy_goldnorm",
    "mean_margin_nll",
)


def eval_checkpoint(
    model: Any,
    tokenizer: AutoTokenizer,
    datasets: Dict[str, List[Dict[str, Any]]],
    args: argparse.Namespace,
    device: torch.device,
    run_name: str,
    model_root: str,
    seed: int,
    epoch: int,
    ckpt: str,
    fp: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Score one loaded checkpoint on every dataset; return overall / category / item rows."""
    overall_rows: List[Dict[str, Any]] = []
    category_rows: List[Dict[str, Any]] = []
    item_rows: List[Dict[str, Any]] = []

    for jp, examples in datasets.items():
        testset = _testset_name(jp)
        items: List[Dict[str, Any]] = []
        (
            y_pred, ppl_g, ppl_c, nll_g, nll_c, cats, skipped, nt_g, nt_c,
        ) = score_examples(
            model,
            tokenizer,
            examples,
            args.gold_key,
            args.corrupt_key,
            device,
            args.max_length,
            category_key=args.category_key,
            return_token_counts=True,
            items_out=None if args.no_per_item_csv else items,
        )
        for it in items:
            item_rows.append(
                {
                    "run_name": run_name,
                    "seed": seed,
                    "epoch": epoch,
                    "testset": testset,
                    **it,
                }
            )
        results = summarize_predictions(
            y_pred,
            ppl_g,
            ppl_c,
            nll_g,
            nll_c,
            cats,
            skipped,
            fp,
            bootstrap_n=args.bootstrap_n,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_alpha=args.bootstrap_alpha,
            ntok_gold=nt_g,
            ntok_corrupt=nt_c,
        )

        logger.info(
            f"[{testset}] seed={seed} epoch={epoch} "
            f"acc={100 * results['accuracy']:.2f}% "
            f"n={results['n_examples']} "
            f"mean_ppl_gold={results['mean_ppl_gold']:.4f} "
            f"surprisal_gold/tok={results['mean_surprisal_gold']:.4f}"
        )
        if results["by_category"]:
            logger.info(
                f"[{testset}] by category:\n"
                f"{format_category_report(results['by_category'])}"
            )

        overall_rows.append(
            {
                "run_name": run_name,
                "model_root": model_root,
                "seed": seed,
                "epoch": epoch,
                "checkpoint": ckpt,
                "testset": testset,
                "json_path": jp,
                "n_examples": results["n_examples"],
                "n_skipped": results["n_skipped"],
                "accuracy": results["accuracy"],
                "precision_prefer_gold": results["precision_prefer_gold"],
                "recall_prefer_gold": results["recall_prefer_gold"],
                "f1_prefer_gold": results["f1_prefer_gold"],
                "mean_ppl_gold": results["mean_ppl_gold"],
                "mean_ppl_corrupt": results["mean_ppl_corrupt"],
                "mean_surprisal_gold": results["mean_surprisal_gold"],
                "mean_surprisal_corrupt": results["mean_surprisal_corrupt"],
                "mean_surprisal_corrupt_goldnorm": results["mean_surprisal_corrupt_goldnorm"],
                "mean_margin_goldnorm": results["mean_margin_goldnorm"],
                "accuracy_goldnorm": results["accuracy_goldnorm"],
                "n_tokens_gold": results["n_tokens_gold"],
                "n_tokens_corrupt": results["n_tokens_corrupt"],
                "margin_nll_mean": results["margin_nll_mean"],
                "margin_nll_median": results["margin_nll_median"],
                "margin_nll_std": results["margin_nll_std"],
                "fingerprint": fp,
            }
        )

        for cat, stats in results["by_category"].items():
            category_rows.append(
                {
                    "run_name": run_name,
                    "model_root": model_root,
                    "seed": seed,
                    "epoch": epoch,
                    "checkpoint": ckpt,
                    "testset": testset,
                    "json_path": jp,
                    "category": cat,
                    "n_examples": int(stats["n_examples"]),
                    "n_correct": int(stats["n_correct"]),
                    "accuracy": stats["accuracy"],
                    "mean_ppl_gold": stats["mean_ppl_gold"],
                    "mean_ppl_corrupt": stats["mean_ppl_corrupt"],
                    "mean_surprisal_gold": stats["mean_surprisal_gold"],
                    "mean_surprisal_corrupt": stats["mean_surprisal_corrupt"],
                    "mean_surprisal_corrupt_goldnorm": stats["mean_surprisal_corrupt_goldnorm"],
                    "mean_margin_goldnorm": stats["mean_margin_goldnorm"],
                    "accuracy_goldnorm": stats["accuracy_goldnorm"],
                    "n_tokens_gold": stats["n_tokens_gold"],
                    "n_tokens_corrupt": stats["n_tokens_corrupt"],
                    "mean_margin_nll": stats["mean_margin_nll"],
                    "fingerprint": fp,
                }
            )

    return overall_rows, category_rows, item_rows


def run_multiseed_sweep(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_root = os.path.expanduser(args.model_root)
    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    seeds = parse_int_list(args.seeds)
    epochs = parse_int_list(args.epochs)
    json_paths = [
        os.path.expanduser(p.strip())
        for p in args.json_path.split(",")
        if p.strip()
    ]
    run_name = args.run_name or os.path.basename(os.path.normpath(model_root))
    phase_subdir = args.phase_subdir

    # Preload datasets once
    datasets: Dict[str, List[Dict[str, Any]]] = {}
    for jp in json_paths:
        examples = load_json_dataset(jp)
        if args.split:
            examples = [e for e in examples if e.get("split") == args.split]
        datasets[jp] = examples
        cats = sorted({normalize_category(e.get(args.category_key)) for e in examples})
        logger.info(
            f"Loaded {jp}: {len(examples)} examples; categories={cats}"
        )

    # Resolve checkpoint list once, then load a single shared tokenizer.
    checkpoints: List[Tuple[int, int, str]] = []
    seen_ckpts: set[str] = set()
    for epoch in epochs:
        for seed in seeds:
            ckpt = resolve_checkpoint(
                model_root,
                seed,
                epoch,
                phase_subdir=phase_subdir,
            )
            if not _is_checkpoint_dir(ckpt):
                logger.warning(f"Missing checkpoint, skipping: {ckpt}")
                continue
            if ckpt in seen_ckpts:
                logger.info(
                    f"Seedless checkpoint already queued, skipping "
                    f"seed={seed}: {ckpt}"
                )
                continue
            seen_ckpts.add(ckpt)
            checkpoints.append((epoch, seed, ckpt))

    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints found under {model_root}/{phase_subdir}"
        )

    if args.tokenizer_path is not None:
        tokenizer_path = args.tokenizer_path
    elif _has_tokenizer(model_root):
        tokenizer_path = model_root
    else:
        tokenizer_path = next(
            (ckpt for _, _, ckpt in checkpoints if _has_tokenizer(ckpt)),
            None,
        )
        if tokenizer_path is None:
            raise FileNotFoundError(
                f"No tokenizer found at model_root or any checkpoint under "
                f"{model_root}/{phase_subdir}"
            )
    logger.info(f"Loading shared tokenizer once from {tokenizer_path}")
    tokenizer = load_tokenizer(tokenizer_path)

    overall_rows: List[Dict[str, Any]] = []
    category_rows: List[Dict[str, Any]] = []
    item_rows: List[Dict[str, Any]] = []

    for epoch, seed, ckpt in checkpoints:
        logger.info(
            f"=== {run_name} | {phase_subdir} | seed={seed} | "
            f"epoch={epoch} | {ckpt} ==="
        )
        model = load_model(ckpt, device)
        fp = checkpoint_fingerprint(ckpt)

        try:
            o_rows, c_rows, i_rows = eval_checkpoint(
                model, tokenizer, datasets, args, device,
                run_name, model_root, seed, epoch, ckpt, fp,
            )
            overall_rows.extend(o_rows)
            category_rows.extend(c_rows)
            item_rows.extend(i_rows)
        finally:
            unload_model(model)

    # Per-seed CSVs
    overall_fields = [
        "run_name",
        "model_root",
        "seed",
        "epoch",
        "checkpoint",
        "testset",
        "json_path",
        "n_examples",
        "n_skipped",
        "accuracy",
        "precision_prefer_gold",
        "recall_prefer_gold",
        "f1_prefer_gold",
        "mean_ppl_gold",
        "mean_ppl_corrupt",
        "mean_surprisal_gold",
        "mean_surprisal_corrupt",
        "mean_surprisal_corrupt_goldnorm",
        "mean_margin_goldnorm",
        "accuracy_goldnorm",
        "n_tokens_gold",
        "n_tokens_corrupt",
        "margin_nll_mean",
        "margin_nll_median",
        "margin_nll_std",
        "fingerprint",
    ]
    category_fields = [
        "run_name",
        "model_root",
        "seed",
        "epoch",
        "checkpoint",
        "testset",
        "json_path",
        "category",
        "n_examples",
        "n_correct",
        "accuracy",
        "mean_ppl_gold",
        "mean_ppl_corrupt",
        "mean_surprisal_gold",
        "mean_surprisal_corrupt",
        "mean_surprisal_corrupt_goldnorm",
        "mean_margin_goldnorm",
        "accuracy_goldnorm",
        "n_tokens_gold",
        "n_tokens_corrupt",
        "mean_margin_nll",
        "fingerprint",
    ]

    write_csv(
        os.path.join(out_dir, f"{run_name}_per_seed_overall.csv"),
        overall_rows,
        overall_fields,
    )
    write_csv(
        os.path.join(out_dir, f"{run_name}_per_seed_by_category.csv"),
        category_rows,
        category_fields,
    )

    if not args.no_per_item_csv:
        item_fields = ["run_name", "seed", "epoch", "testset"] + list(item_rows[0])[4:] if item_rows else []
        write_csv(
            os.path.join(out_dir, f"{run_name}_per_item.csv"),
            item_rows,
            item_fields,
        )

    # Aggregates across seeds
    agg_overall = aggregate_rows(
        overall_rows,
        group_keys=("run_name", "epoch", "testset"),
        metric_keys=OVERALL_METRIC_KEYS,
    )
    agg_category = aggregate_rows(
        category_rows,
        group_keys=("run_name", "epoch", "testset", "category"),
        metric_keys=CATEGORY_METRIC_KEYS,
    )

    agg_overall_fields = [
        "run_name",
        "epoch",
        "testset",
        "n_seeds",
        "seeds",
        "n_examples",
        "accuracy_mean",
        "accuracy_std",
        "mean_ppl_gold_mean",
        "mean_ppl_gold_std",
        "mean_ppl_corrupt_mean",
        "mean_ppl_corrupt_std",
        "mean_surprisal_gold_mean",
        "mean_surprisal_gold_std",
        "mean_surprisal_corrupt_mean",
        "mean_surprisal_corrupt_std",
        "mean_surprisal_corrupt_goldnorm_mean",
        "mean_surprisal_corrupt_goldnorm_std",
        "mean_margin_goldnorm_mean",
        "mean_margin_goldnorm_std",
        "accuracy_goldnorm_mean",
        "accuracy_goldnorm_std",
        "margin_nll_mean_mean",
        "margin_nll_mean_std",
        "f1_prefer_gold_mean",
        "f1_prefer_gold_std",
    ]
    agg_category_fields = [
        "run_name",
        "epoch",
        "testset",
        "category",
        "n_seeds",
        "seeds",
        "n_examples",
        "accuracy_mean",
        "accuracy_std",
        "mean_ppl_gold_mean",
        "mean_ppl_gold_std",
        "mean_ppl_corrupt_mean",
        "mean_ppl_corrupt_std",
        "mean_surprisal_gold_mean",
        "mean_surprisal_gold_std",
        "mean_surprisal_corrupt_mean",
        "mean_surprisal_corrupt_std",
        "mean_surprisal_corrupt_goldnorm_mean",
        "mean_surprisal_corrupt_goldnorm_std",
        "mean_margin_goldnorm_mean",
        "mean_margin_goldnorm_std",
        "accuracy_goldnorm_mean",
        "accuracy_goldnorm_std",
        "mean_margin_nll_mean",
        "mean_margin_nll_std",
    ]

    write_csv(
        os.path.join(out_dir, f"{run_name}_aggregate_overall.csv"),
        agg_overall,
        agg_overall_fields,
    )
    write_csv(
        os.path.join(out_dir, f"{run_name}_aggregate_by_category.csv"),
        agg_category,
        agg_category_fields,
    )

    logger.info(f"Done. CSV outputs in {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def discover_checkpoints(scan_root: str) -> List[Tuple[str, str, int, int, str]]:
    """Find every checkpoint dir under ``scan_root``.

    Returns sorted ``(ckpt_path, run_name, epoch, seed, rel_path)`` tuples.
    ``run_name`` is the relative path without the ``ep{k}`` / ``seed_{s}``
    components (so seeds/epochs of one configuration share a run_name);
    missing epoch / seed default to 0.
    """
    found = []
    for dirpath, dirnames, _ in os.walk(scan_root):
        if _is_checkpoint_dir(dirpath):
            dirnames[:] = []
            rel = os.path.relpath(dirpath, scan_root)
            parts = [] if rel == "." else rel.split(os.sep)
            epoch = seed = 0
            config = []
            for part in parts:
                m_ep = re.fullmatch(r"ep(\d+)", part)
                m_sd = re.fullmatch(r"seed_(\d+)", part)
                if m_ep:
                    epoch = int(m_ep.group(1))
                elif m_sd:
                    seed = int(m_sd.group(1))
                else:
                    config.append(part)
            run_name = "_".join(config) or os.path.basename(os.path.normpath(scan_root))
            found.append((dirpath, run_name, epoch, seed, rel))
    return sorted(found, key=lambda t: t[4])


def run_scan(args: argparse.Namespace) -> None:
    """Evaluate every checkpoint found recursively under ``--scan_root``.

    Results go to ``{out_dir}/{relative checkpoint path}/`` (overall.csv,
    by_category.csv, per_item.csv). Checkpoints whose overall.csv already exists
    are skipped, so a preempted job can simply be resubmitted. Use
    ``--shard_index/--num_shards`` to split the list across jobs, then
    ``--merge_only`` to build the combined CSVs.
    """
    scan_root = os.path.abspath(os.path.expanduser(args.scan_root))
    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    if args.merge_only:
        merge_scan_results(out_dir)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    json_paths = [
        os.path.expanduser(p.strip()) for p in args.json_path.split(",") if p.strip()
    ]
    datasets: Dict[str, List[Dict[str, Any]]] = {}
    for jp in json_paths:
        examples = load_json_dataset(jp)
        if args.split:
            examples = [e for e in examples if e.get("split") == args.split]
        datasets[jp] = examples
        logger.info(f"Loaded {jp}: {len(examples)} examples")

    checkpoints = discover_checkpoints(scan_root)
    logger.info(f"Found {len(checkpoints)} checkpoints under {scan_root}")
    checkpoints = checkpoints[args.shard_index :: args.num_shards]
    logger.info(
        f"Shard {args.shard_index}/{args.num_shards}: {len(checkpoints)} checkpoints"
    )
    if not checkpoints:
        return

    tokenizer_path = args.tokenizer_path or next(
        (c for c, *_ in checkpoints if _has_tokenizer(c)), None
    )
    if tokenizer_path is None:
        raise FileNotFoundError("Pass --tokenizer_path (none found in checkpoints)")
    logger.info(f"Loading shared tokenizer once from {tokenizer_path}")
    tokenizer = load_tokenizer(tokenizer_path)

    for n, (ckpt, run_name, epoch, seed, rel) in enumerate(checkpoints, 1):
        ckpt_out = os.path.join(out_dir, rel)
        if os.path.isfile(os.path.join(ckpt_out, "overall.csv")):
            logger.info(f"[{n}/{len(checkpoints)}] already done, skipping: {rel}")
            continue
        logger.info(f"=== [{n}/{len(checkpoints)}] {rel} ({run_name} ep={epoch} seed={seed}) ===")
        model = load_model(ckpt, device)
        try:
            o_rows, c_rows, i_rows = eval_checkpoint(
                model, tokenizer, datasets, args, device,
                run_name, scan_root, seed, epoch, ckpt, checkpoint_fingerprint(ckpt),
            )
        finally:
            unload_model(model)
        for row in o_rows + c_rows:
            row["rel_path"] = rel
        if i_rows:
            write_csv(os.path.join(ckpt_out, "per_item.csv"), i_rows, list(i_rows[0]))
        write_csv(os.path.join(ckpt_out, "by_category.csv"), c_rows, list(c_rows[0]))
        # Written last: its existence marks the checkpoint as finished.
        write_csv(os.path.join(ckpt_out, "overall.csv"), o_rows, list(o_rows[0]))

    logger.info("Scan finished. Run with --merge_only to build combined CSVs.")


def merge_scan_results(out_dir: str) -> None:
    """Concatenate per-checkpoint CSVs and average across seeds."""

    def _collect(name: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for dirpath, _, files in sorted(os.walk(out_dir)):
            if name in files:
                with open(os.path.join(dirpath, name), encoding="utf-8", newline="") as f:
                    rows.extend(csv.DictReader(f))
        for r in rows:
            r["epoch"] = int(r["epoch"])
        return rows

    for name, keys, metrics in (
        ("overall.csv", ("run_name", "epoch", "testset"), OVERALL_METRIC_KEYS),
        ("by_category.csv", ("run_name", "epoch", "testset", "category"), CATEGORY_METRIC_KEYS),
    ):
        rows = _collect(name)
        if not rows:
            logger.warning(f"No {name} found under {out_dir}")
            continue
        stem = name[:-4]
        write_csv(os.path.join(out_dir, f"all_{stem}.csv"), rows, list(rows[0]))
        agg = aggregate_rows(rows, group_keys=keys, metric_keys=metrics)
        write_csv(os.path.join(out_dir, f"all_{stem}_seed_mean.csv"), agg, list(agg[0]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare PPL of gold vs corrupted sentences. "
            "Use --model_path for one checkpoint, or --model_root for a "
            "multi-seed / multi-epoch sweep that writes CSV summaries."
        )
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Single HuggingFace checkpoint directory.",
    )
    mode.add_argument(
        "--model_root",
        type=str,
        default=None,
        help="Root like models/v3/char-en-de (holds tokenizer + phase_* dirs).",
    )

    mode.add_argument(
        "--scan_root",
        type=str,
        default=None,
        help="Evaluate EVERY checkpoint found recursively under this dir; results "
        "go to --out_dir/<relative checkpoint path>/ (resumable).",
    )
    p.add_argument("--shard_index", type=int, default=0, help="--scan_root: this job's shard.")
    p.add_argument("--num_shards", type=int, default=1, help="--scan_root: total shards.")
    p.add_argument(
        "--merge_only",
        action="store_true",
        help="--scan_root: only merge existing per-checkpoint CSVs in --out_dir.",
    )

    p.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Tokenizer dir. Sweep default: --model_root. "
        "Single-checkpoint default: same as --model_path.",
    )
    p.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="Comma-separated JSON dataset paths.",
    )
    p.add_argument("--gold_key", type=str, default="gold_text")
    p.add_argument("--corrupt_key", type=str, default="corrupted")
    p.add_argument(
        "--category_key",
        type=str,
        default="category",
        help="JSON field used for per-category breakdown (missing -> NaN).",
    )
    p.add_argument(
        "--split",
        type=str,
        default=None,
        help="If set, only rows with this `split` value.",
    )
    p.add_argument("--max_length", type=int, default=1024)

    # Sweep options
    p.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Name prefix for CSV files (default: basename of --model_root).",
    )
    p.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated seeds for --model_root mode.",
    )
    p.add_argument(
        "--epochs",
        type=str,
        default="1,2,3,4,5",
        help="Comma-separated epoch indices matching ep{k} directories (e.g. 1,2,3,4,5).",
    )
    p.add_argument(
        "--phase_subdir",
        type=str,
        default="",
        help="Phase directory under --model_root "
        "(e.g. phase_2_from_ep5, phase_2_from_ep10, phase_1).",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="~/Morphological/results",
        help="Directory for CSV outputs in --model_root mode.",
    )

    p.add_argument(
        "--bootstrap_n",
        type=int,
        default=0,
        help="Bootstrap resamples for CI (0 disables; recommended 0 for sweeps).",
    )
    p.add_argument(
        "--no_per_item_csv",
        action="store_true",
        help="Skip the per-test-item CSV (log-probs, lengths, margins) in sweep mode.",
    )
    p.add_argument("--bootstrap_seed", type=int, default=42)
    p.add_argument("--bootstrap_alpha", type=float, default=0.05)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.tokenizer_path:
        args.tokenizer_path = os.path.expanduser(args.tokenizer_path)

    if args.scan_root:
        run_scan(args)
        return

    if args.model_root:
        run_multiseed_sweep(args)
        return

    # Single-checkpoint mode
    args.model_path = os.path.expanduser(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    json_paths = [
        os.path.expanduser(path.strip())
        for path in args.json_path.split(",")
        if path.strip()
    ]

    for j_path in json_paths:
        results = evaluate(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            json_path=j_path,
            gold_key=args.gold_key,
            corrupt_key=args.corrupt_key,
            split_filter=args.split,
            max_length=args.max_length,
            device=device,
            bootstrap_n=args.bootstrap_n,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_alpha=args.bootstrap_alpha,
            category_key=args.category_key,
        )
        logger.info(
            f"\n--- Results for model in {args.model_path}\nTest set in {j_path} ---"
        )
        logger.info(f"Examples evaluated: {results['n_examples']}")
        logger.info(
            f"Accuracy (PPL_gold <= PPL_corrupt): {results['accuracy'] * 100:.2f}%\n"
        )
        if results["by_category"]:
            logger.info(
                f"Accuracy by category "
                f"({results['n_with_category']} labeled, "
                f"{results['n_examples'] - results['n_with_category']} NaN):\n"
                f"{format_category_report(results['by_category'])}\n"
            )


if __name__ == "__main__":
    main()
