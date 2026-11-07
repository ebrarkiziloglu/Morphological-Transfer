#!/usr/bin/env python3
"""Extract a 100M-English-word trilingual EN-TR-DE subset from NLLB.

NLLB only ships pairwise corpora, so triples are built by joining EN-TR and
DE-TR on the (exact) Turkish sentence. Memory-light two-pass design:

  1. Scan DE-TR, filter, and map hash(Turkish) -> DE-TR line index.
  2. Scan EN-TR, filter, and join on Turkish -> (EN-TR line, DE-TR line, EN words).
  3. Shuffle the triples and keep a prefix reaching ~100M English words.
  4. Re-read the files and collect only the selected lines.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
from datasets import Dataset
from dataset_analysis_helper import analyze_dataset, word_count


REPO_ROOT = Path(__file__).resolve().parents[3]
NLLB_DIR = REPO_ROOT / "data" / "NLLB"

EN_TR_EN = NLLB_DIR / "NLLB.en-tr.en"
EN_TR_TR = NLLB_DIR / "NLLB.en-tr.tr"
DE_TR_DE = NLLB_DIR / "NLLB.de-tr.de"
DE_TR_TR = NLLB_DIR / "NLLB.de-tr.tr"

TXT_SAVE_DIR = REPO_ROOT / "data" / "NLLB_100M_en_tr_de"
ARROW_SAVE_PATH = REPO_ROOT / "data" / "arrow" / "NLLB_100M_en_tr_de"
DATASET_NAME = "NLLB_100M_en_tr_de"

# The EN-TR x DE-TR Turkish overlap is limited; if the target is not reached,
# lower MIN_WORDS (a warning is printed and all triples are kept).
MIN_WORDS = 5
MAX_CHARS = 1500
TARGET_EN_WORDS = 100_000_000
SEED = 42
PROGRESS_EVERY = 5_000_000


def ok_pair(a: str, b: str, min_words: int = MIN_WORDS) -> bool:
    """Both sides non-empty, >= min_words words, <= MAX_CHARS chars."""
    return bool(
        a
        and b
        and len(a) <= MAX_CHARS
        and len(b) <= MAX_CHARS
        and word_count(a) >= min_words
        and word_count(b) >= min_words
    )


def check_same_length(f1, f2, name: str) -> None:
    if f1.readline() or f2.readline():
        raise ValueError(f"{name} files have mismatched lengths")


def index_de_tr() -> dict[int, int]:
    """Map hash(Turkish) -> DE-TR line index for filtered lines (first wins)."""
    tr_to_idx: dict[int, int] = {}
    n_seen = n_filtered = n_dup = 0
    t0 = time.time()
    print(f"Indexing DE-TR:\n  {DE_TR_DE}\n  {DE_TR_TR}")
    with (
        DE_TR_DE.open("r", encoding="utf-8") as de_f,
        DE_TR_TR.open("r", encoding="utf-8") as tr_f,
    ):
        for i, (de_line, tr_line) in enumerate(zip(de_f, tr_f)):
            n_seen += 1
            de, tr = de_line.strip(), tr_line.strip()
            if not ok_pair(de, tr):
                n_filtered += 1
            else:
                h = hash(tr)
                if h in tr_to_idx:
                    n_dup += 1
                else:
                    tr_to_idx[h] = i
            if n_seen % PROGRESS_EVERY == 0:
                print(
                    f"  DE-TR scanned {n_seen:,} | indexed {len(tr_to_idx):,} | "
                    f"filtered {n_filtered:,} | dup TR {n_dup:,} | "
                    f"{n_seen / (time.time() - t0):,.0f} lines/s",
                    flush=True,
                )
        check_same_length(de_f, tr_f, "DE-TR")
    print(f"Indexed DE-TR in {(time.time() - t0) / 60:.1f} min: {len(tr_to_idx):,} usable.")
    return tr_to_idx


def join_en_tr(
    tr_to_de: dict[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scan EN-TR; return (EN-TR line idx, DE-TR line idx, EN word counts)."""
    en_idx: list[int] = []
    de_idx: list[int] = []
    counts: list[int] = []
    n_seen = n_filtered = 0
    t0 = time.time()
    print(f"Scanning EN-TR and joining on Turkish:\n  {EN_TR_EN}\n  {EN_TR_TR}")
    with (
        EN_TR_EN.open("r", encoding="utf-8") as en_f,
        EN_TR_TR.open("r", encoding="utf-8") as tr_f,
    ):
        for i, (en_line, tr_line) in enumerate(zip(en_f, tr_f)):
            n_seen += 1
            tr = tr_line.strip()
            # pop => each Turkish sentence yields at most one triple
            j = tr_to_de.pop(hash(tr), None) if tr else None
            if j is not None:
                en = en_line.strip()
                if ok_pair(en, tr):
                    en_idx.append(i)
                    de_idx.append(j)
                    counts.append(word_count(en))
                else:
                    n_filtered += 1
            if n_seen % PROGRESS_EVERY == 0:
                print(
                    f"  EN-TR scanned {n_seen:,} | triples {len(en_idx):,} | "
                    f"filtered {n_filtered:,} | "
                    f"{n_seen / (time.time() - t0):,.0f} lines/s",
                    flush=True,
                )
        check_same_length(en_f, tr_f, "EN-TR")
    print(f"Join done in {(time.time() - t0) / 60:.1f} min: {len(en_idx):,} triples.")
    return (
        np.array(en_idx, dtype=np.int64),
        np.array(de_idx, dtype=np.int64),
        np.array(counts, dtype=np.int64),
    )


def collect_lines(path: Path, wanted: set[int]) -> dict[int, str]:
    out: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in wanted:
                out[i] = line.strip().replace("\n", " ")
    return out


def extract_en_tr_de_subset() -> None:
    for path in (EN_TR_EN, EN_TR_TR, DE_TR_DE, DE_TR_TR):
        if not path.exists():
            raise FileNotFoundError(f"Missing required data file: {path}")

    tr_to_de = index_de_tr()
    en_idx, de_idx, counts = join_en_tr(tr_to_de)
    del tr_to_de
    if len(en_idx) == 0:
        raise RuntimeError("No EN-TR-DE triples remained after join/filter.")

    # Shuffle triples, keep prefix reaching the English-word target.
    order = np.random.default_rng(SEED).permutation(len(en_idx))
    cum = np.cumsum(counts[order])
    n_keep = min(int(np.searchsorted(cum, TARGET_EN_WORDS)) + 1, len(order))
    sel = order[:n_keep]
    total_en_words = int(cum[n_keep - 1])
    sel_en, sel_de = en_idx[sel], de_idx[sel]
    print(f"Selected {n_keep:,} triples, {total_en_words:,} English words.")
    if total_en_words < TARGET_EN_WORDS:
        print(f"WARNING: only reached {total_en_words:,} (target {TARGET_EN_WORDS:,}).")

    # Pass 2: collect the selected lines.
    en_lines = collect_lines(EN_TR_EN, set(sel_en.tolist()))
    tr_lines = collect_lines(EN_TR_TR, set(sel_en.tolist()))
    de_lines = collect_lines(DE_TR_DE, set(sel_de.tolist()))
    triples = [
        {"en": en_lines[a], "tr": tr_lines[a], "de": de_lines[b]}
        for a, b in zip(sel_en.tolist(), sel_de.tolist())
    ]
    del en_lines, tr_lines, de_lines
    random.Random(SEED).shuffle(triples)

    subset = Dataset.from_list(triples)
    print(f"Saving arrow dataset to {ARROW_SAVE_PATH}...")
    ARROW_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    subset.save_to_disk(str(ARROW_SAVE_PATH))

    TXT_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving EN / TR / DE text files to {TXT_SAVE_DIR}...")
    with (
        (TXT_SAVE_DIR / "NLLB_100M.en").open("w", encoding="utf-8") as f_en,
        (TXT_SAVE_DIR / "NLLB_100M.tr").open("w", encoding="utf-8") as f_tr,
        (TXT_SAVE_DIR / "NLLB_100M.de").open("w", encoding="utf-8") as f_de,
    ):
        for item in triples:
            f_en.write(item["en"] + "\n")
            f_tr.write(item["tr"] + "\n")
            f_de.write(item["de"] + "\n")

    print("Analyzing the sample dataset (EN-TR-DE stats)...")
    analyze_dataset(subset, DATASET_NAME)
    print("Done.")


if __name__ == "__main__":
    extract_en_tr_de_subset()
