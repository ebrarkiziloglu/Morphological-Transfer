#!/usr/bin/env python3
"""Extract a 100M-English-word EN-TR parallel subset from NLLB.

Pass 1 scans the corpus, applies length filters and exact-pair dedup, and keeps
only per-line English word counts. Eligible lines are shuffled and taken until
~100M English words. Pass 2 re-reads the files and collects the selected lines.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
from datasets import Dataset
from dataset_analysis_helper import analyze_dataset, word_count


REPO_ROOT = Path(__file__).resolve().parents[3]

EN_PATH = REPO_ROOT / "data" / "NLLB" / "en-tr.txt" / "NLLB.en-tr.en"
TR_PATH = REPO_ROOT / "data" / "NLLB" / "en-tr.txt" / "NLLB.en-tr.tr"

TXT_SAVE_DIR = REPO_ROOT / "data" / "NLLB_100M"
ARROW_SAVE_PATH = REPO_ROOT / "data" / "arrow" / "NLLB_100M"

MIN_WORDS = 10
MAX_CHARS = 1500
TARGET_EN_WORDS = 100_000_000
SEED = 42
PROGRESS_EVERY = 5_000_000


def scan_eligible() -> tuple[np.ndarray, np.ndarray]:
    """Return (line indices, English word counts) of lines passing the filters."""
    idx: list[int] = []
    counts: list[int] = []
    seen: set[int] = set()
    n_seen = n_filtered = n_dup = 0
    t0 = time.time()

    print(f"Scanning:\n  {EN_PATH}\n  {TR_PATH}")
    with (
        EN_PATH.open("r", encoding="utf-8") as en_f,
        TR_PATH.open("r", encoding="utf-8") as tr_f,
    ):
        for i, (en_line, tr_line) in enumerate(zip(en_f, tr_f)):
            n_seen += 1
            en, tr = en_line.strip(), tr_line.strip()
            n_en = word_count(en)
            if (
                not en
                or not tr
                or n_en < MIN_WORDS
                or word_count(tr) < MIN_WORDS
                or len(en) > MAX_CHARS
                or len(tr) > MAX_CHARS
            ):
                n_filtered += 1
            else:
                h = hash((en, tr))
                if h in seen:
                    n_dup += 1
                else:
                    seen.add(h)
                    idx.append(i)
                    counts.append(n_en)

            if n_seen % PROGRESS_EVERY == 0:
                rate = n_seen / (time.time() - t0)
                print(
                    f"  scanned {n_seen:,} | eligible {len(idx):,} | "
                    f"filtered {n_filtered:,} | dup {n_dup:,} | {rate:,.0f} lines/s",
                    flush=True,
                )

        if en_f.readline() or tr_f.readline():
            raise ValueError(f"EN-TR files have mismatched lengths near line {n_seen}")

    print(
        f"Scan done in {(time.time() - t0) / 60:.1f} min: {len(idx):,} eligible of "
        f"{n_seen:,} (filtered {n_filtered:,}, duplicates {n_dup:,})."
    )
    return np.array(idx, dtype=np.int64), np.array(counts, dtype=np.int64)


def select_lines(idx: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, int]:
    """Shuffle eligible lines and keep the prefix reaching the word target."""
    order = np.random.default_rng(SEED).permutation(len(idx))
    cum = np.cumsum(counts[order])
    n_keep = min(int(np.searchsorted(cum, TARGET_EN_WORDS)) + 1, len(order))
    return idx[order[:n_keep]], int(cum[n_keep - 1])


def extract_en_tr_subset() -> None:
    for path in (EN_PATH, TR_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing required data file: {path}")

    idx, counts = scan_eligible()
    if len(idx) == 0:
        raise RuntimeError("No lines remained after filtering.")

    selected, total_en_words = select_lines(idx, counts)
    print(f"Selected {len(selected):,} pairs, {total_en_words:,} English words.")
    if total_en_words < TARGET_EN_WORDS:
        print(f"WARNING: only reached {total_en_words:,} (target {TARGET_EN_WORDS:,}).")

    # Pass 2: collect the selected lines, then shuffle them.
    wanted = set(selected.tolist())
    pairs: list[dict[str, str]] = []
    with (
        EN_PATH.open("r", encoding="utf-8") as en_f,
        TR_PATH.open("r", encoding="utf-8") as tr_f,
    ):
        for i, (en_line, tr_line) in enumerate(zip(en_f, tr_f)):
            if i in wanted:
                pairs.append({"en": en_line.strip(), "tr": tr_line.strip()})
    random.Random(SEED).shuffle(pairs)

    subset = Dataset.from_list(pairs)
    print(f"Saving arrow dataset to {ARROW_SAVE_PATH}...")
    ARROW_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    subset.save_to_disk(str(ARROW_SAVE_PATH))

    TXT_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving EN / TR text files to {TXT_SAVE_DIR}...")
    with (
        (TXT_SAVE_DIR / "NLLB_100M.en").open("w", encoding="utf-8") as f_en,
        (TXT_SAVE_DIR / "NLLB_100M.tr").open("w", encoding="utf-8") as f_tr,
    ):
        for item in pairs:
            f_en.write(item["en"].replace("\n", " ") + "\n")
            f_tr.write(item["tr"].replace("\n", " ") + "\n")

    print("Analyzing the sample dataset (EN-TR stats)...")
    analyze_dataset(subset, "NLLB_100M")
    print("Done.")


if __name__ == "__main__":
    extract_en_tr_subset()
