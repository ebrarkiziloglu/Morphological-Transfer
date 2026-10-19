import random

DATA_DIR = "./data/NLLB_10M_subset"
LANGS = ["en", "tr", "de", "zh"]
LINES_PER_LANG = 200_000
OUT_PATH = "./data/multilingual_corpus.txt"
SEED = 42

random.seed(SEED)
corpus = []
for lang in LANGS:
    with open(f"{DATA_DIR}/NLLB_10M.{lang}", "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    sample = random.sample(lines, min(LINES_PER_LANG, len(lines)))
    print(f"{lang}: {len(lines)} lines -> sampled {len(sample)}")
    corpus.extend(sample)

random.shuffle(corpus)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(corpus) + "\n")
print(f"Wrote {len(corpus)} lines to {OUT_PATH}")
