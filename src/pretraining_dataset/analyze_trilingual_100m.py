#!/usr/bin/env python3
"""Analyze the already-extracted 100M EN-TR-DE corpus (no re-sampling).

Reads data/Trilingual_100M_en_tr_de/Trilingual_100M.{en,tr,de} (line-aligned)
and runs analyze_dataset on it.
"""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset
from dataset_analysis_helper import analyze_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
TXT_DIR = REPO_ROOT / "data" / "Trilingual_100M_en_tr_de"
DATASET_NAME = "Trilingual_100M_en_tr_de"
LANGS = ("en", "tr", "de")


def read_lines(path: Path) -> list[str]:
    # newline="\n": split only on \n so line alignment matches how the files were written
    with path.open("r", encoding="utf-8", newline="\n") as f:
        return [line.rstrip("\n") for line in f]


def main() -> None:
    columns = {}
    for lang in LANGS:
        path = TXT_DIR / f"Trilingual_100M.{lang}"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        print(f"Reading {path}...", flush=True)
        columns[lang] = read_lines(path)

    lengths = {l: len(v) for l, v in columns.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Line counts differ across languages: {lengths}")
    print(f"Loaded {lengths['en']:,} aligned triples.")

    analyze_dataset(Dataset.from_dict(columns), DATASET_NAME)
    print("Done.")


if __name__ == "__main__":
    main()
