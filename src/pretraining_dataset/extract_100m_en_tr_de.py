#!/usr/bin/env python3
"""Build a 100M-English-word trilingual EN-TR-DE corpus from NLLB, OpenSubtitles
and CCAligned (in that order).

Only pairwise corpora exist (DE-TR and DE-EN), so triples are built by joining
the two pairs on the (exact) German sentence. For each source, in order:

  1. Scan DE-TR, filter, and map hash(German) -> Turkish sentence.
  2. Stream DE-EN, and for every German sentence found in the map emit a
     (en, tr, de) triple, counting English words.
  3. Stop as soon as TARGET_EN_WORDS is reached; otherwise move on to the next
     source. German sentences already used are skipped in later sources.

Layout expected under data/trilingual_parallel_data/:
  NLLB/{tr_de,en_de}/NLLB.de-tr.{de,tr}, NLLB.de-en.{de,en}
  OpenSubtitles/{tr_de,en_de}/OpenSubtitles.de-tr.{de,tr}, OpenSubtitles.de-en.{de,en}
  CCAligned/tr_de/MultiCCAligned.de-tr.{de,tr}, CCAligned/en_de/CCAligned.de-en.{de,en}
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset
from dataset_analysis_helper import analyze_dataset, word_count


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "trilingual_parallel_data"

TXT_SAVE_DIR = REPO_ROOT / "data" / "trilingual_100M_en_tr_de"
DATASET_NAME = "trilingual_100M_en_tr_de"

TARGET_EN_WORDS = 100_000_000
MAX_CHARS = 1500
SEED = 42
PROGRESS_EVERY = 5_000_000


@dataclass(frozen=True)
class Source:
    name: str
    de_tr_de: Path
    de_tr_tr: Path
    de_en_de: Path
    de_en_en: Path
    min_words: int  # minimum words on EACH side of a pair


SOURCES = [
    Source(
        "NLLB",
        DATA_DIR / "NLLB" / "tr_de" / "NLLB.de-tr.de",
        DATA_DIR / "NLLB" / "tr_de" / "NLLB.de-tr.tr",
        DATA_DIR / "NLLB" / "en_de" / "NLLB.de-en.de",
        DATA_DIR / "NLLB" / "en_de" / "NLLB.de-en.en",
        min_words=6,
    ),
    Source(
        "OpenSubtitles",
        DATA_DIR / "OpenSubtitles" / "tr_de" / "OpenSubtitles.de-tr.de",
        DATA_DIR / "OpenSubtitles" / "tr_de" / "OpenSubtitles.de-tr.tr",
        DATA_DIR / "OpenSubtitles" / "en_de" / "OpenSubtitles.de-en.de",
        DATA_DIR / "OpenSubtitles" / "en_de" / "OpenSubtitles.de-en.en",
        min_words=6,  # subtitles are full of very short utterances
    ),
    Source(
        "CCAligned",
        DATA_DIR / "CCAligned" / "tr_de" / "MultiCCAligned.de-tr.de",
        DATA_DIR / "CCAligned" / "tr_de" / "MultiCCAligned.de-tr.tr",
        DATA_DIR / "CCAligned" / "en_de" / "CCAligned.de-en.de",
        DATA_DIR / "CCAligned" / "en_de" / "CCAligned.de-en.en",
        min_words=6,
    ),
]


def ok_pair(a: str, b: str, min_words: int) -> bool:
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


def index_de_tr(src: Source, used_de: set[int]) -> dict[int, str]:
    """Map hash(German) -> Turkish for filtered DE-TR pairs (first wins)."""
    de_to_tr: dict[int, str] = {}
    n_seen = n_filtered = n_dup = n_used = 0
    t0 = time.time()
    print(f"[{src.name}] Indexing DE-TR:\n  {src.de_tr_de}\n  {src.de_tr_tr}")
    with (
        src.de_tr_de.open("r", encoding="utf-8") as de_f,
        src.de_tr_tr.open("r", encoding="utf-8") as tr_f,
    ):
        for de_line, tr_line in zip(de_f, tr_f):
            n_seen += 1
            de, tr = de_line.strip(), tr_line.strip()
            if not ok_pair(de, tr, src.min_words):
                n_filtered += 1
            else:
                h = hash(de)
                if h in used_de:
                    n_used += 1
                elif h in de_to_tr:
                    n_dup += 1
                else:
                    de_to_tr[h] = tr
            if n_seen % PROGRESS_EVERY == 0:
                print(
                    f"  DE-TR scanned {n_seen:,} | indexed {len(de_to_tr):,} | "
                    f"filtered {n_filtered:,} | dup DE {n_dup:,} | "
                    f"used earlier {n_used:,} | "
                    f"{n_seen / (time.time() - t0):,.0f} lines/s",
                    flush=True,
                )
        check_same_length(de_f, tr_f, f"{src.name} DE-TR")
    print(
        f"[{src.name}] Indexed DE-TR in {(time.time() - t0) / 60:.1f} min: "
        f"{len(de_to_tr):,} usable."
    )
    return de_to_tr


def join_de_en(
    src: Source,
    de_to_tr: dict[int, str],
    used_de: set[int],
    triples: list[dict[str, str]],
    words_so_far: int,
) -> int:
    """Stream DE-EN, append joined triples, stop at the target. Returns new word total."""
    n_seen = n_filtered = 0
    n_start = len(triples)
    t0 = time.time()
    print(f"[{src.name}] Scanning DE-EN and joining on German:\n  {src.de_en_de}\n  {src.de_en_en}")
    with (
        src.de_en_de.open("r", encoding="utf-8") as de_f,
        src.de_en_en.open("r", encoding="utf-8") as en_f,
    ):
        for de_line, en_line in zip(de_f, en_f):
            n_seen += 1
            de = de_line.strip()
            h = hash(de) if de else None
            # pop => each German sentence yields at most one triple
            tr = de_to_tr.pop(h, None) if h is not None else None
            if tr is not None:
                en = en_line.strip()
                if ok_pair(en, de, src.min_words):
                    triples.append({"en": en, "tr": tr, "de": de})
                    used_de.add(h)
                    words_so_far += word_count(en)
                    if words_so_far >= TARGET_EN_WORDS:
                        print(f"[{src.name}] Target reached after {n_seen:,} DE-EN lines.")
                        break
                else:
                    n_filtered += 1
            if n_seen % PROGRESS_EVERY == 0:
                print(
                    f"  DE-EN scanned {n_seen:,} | triples {len(triples) - n_start:,} | "
                    f"filtered {n_filtered:,} | EN words {words_so_far:,} | "
                    f"{n_seen / (time.time() - t0):,.0f} lines/s",
                    flush=True,
                )
        else:
            check_same_length(de_f, en_f, f"{src.name} DE-EN")
    print(
        f"[{src.name}] Join done in {(time.time() - t0) / 60:.1f} min: "
        f"{len(triples) - n_start:,} triples; total EN words {words_so_far:,}."
    )
    return words_so_far


def extract_en_tr_de_subset() -> None:
    for src in SOURCES:
        for path in (src.de_tr_de, src.de_tr_tr, src.de_en_de, src.de_en_en):
            if not path.exists():
                raise FileNotFoundError(f"Missing required data file: {path}")

    triples: list[dict[str, str]] = []
    used_de: set[int] = set()
    total_words = 0
    for src in SOURCES:
        de_to_tr = index_de_tr(src, used_de)
        total_words = join_de_en(src, de_to_tr, used_de, triples, total_words)
        del de_to_tr
        if total_words >= TARGET_EN_WORDS:
            break

    if not triples:
        raise RuntimeError("No EN-TR-DE triples remained after join/filter.")
    print(f"Collected {len(triples):,} triples, {total_words:,} English words.")
    if total_words < TARGET_EN_WORDS:
        print(f"WARNING: only reached {total_words:,} (target {TARGET_EN_WORDS:,}).")

    # Mix sources so the output is not ordered NLLB -> OpenSubtitles -> CCAligned.
    random.Random(SEED).shuffle(triples)

    TXT_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving EN / TR / DE text files to {TXT_SAVE_DIR}...")
    with (
        (TXT_SAVE_DIR / "trilingual_100M.en").open("w", encoding="utf-8") as f_en,
        (TXT_SAVE_DIR / "trilingual_100M.tr").open("w", encoding="utf-8") as f_tr,
        (TXT_SAVE_DIR / "trilingual_100M.de").open("w", encoding="utf-8") as f_de,
    ):
        for item in triples:
            f_en.write(item["en"].replace("\n", " ") + "\n")
            f_tr.write(item["tr"].replace("\n", " ") + "\n")
            f_de.write(item["de"].replace("\n", " ") + "\n")

    print("Analyzing the sample dataset (EN-TR-DE stats)...")
    analyze_dataset(Dataset.from_list(triples), DATASET_NAME)
    print("Done.")


if __name__ == "__main__":
    extract_en_tr_de_subset()
