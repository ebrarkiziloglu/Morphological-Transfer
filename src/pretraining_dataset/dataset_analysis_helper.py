from datasets import load_dataset, load_from_disk, Dataset, concatenate_datasets
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import os
import re
from pathlib import Path


# Repo root: .../Morphological (this file lives in src/pretraining_dataset/)
REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = REPO_ROOT / "data"


def _analysis_plot_path(filename: str) -> Path:
    """Return an absolute plot path under repo data/, creating the dir if needed."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    return ANALYSIS_DIR / filename


# Helper functions
def word_count(text):
    return len(text.split())

def char_count(text):
    return len(text)

def tokenize_text(text, lowercase=True):
    """
    Tokenize text by removing punctuation and optionally lowercasing.
    Filters out pure digit tokens.
    """
    # Remove punctuation (keep only word characters and whitespace)
    cleaned = re.sub(r'[^\w\s]', ' ', text)
    words = cleaned.split()
    tokens = []
    for word in words:
        word = word.strip()
        if word and not word.isdigit():
            if lowercase:
                tokens.append(word.lower())
            else:
                tokens.append(word)
    return tokens


_LANG_STYLE = {
    'en': ('English', 'blue', 'lightblue'),
    'tr': ('Turkish', 'orange', 'wheat'),
    'de': ('German', 'green', 'lightgreen'),
}
_LANG_ORDER = ['en', 'tr', 'de']


def analyze_dataset(dataset, dataset_name):
    """Analyze a parallel corpus with any number of language columns.

    Every column of `dataset` is treated as a language (en, tr, de first, then
    any others). Per-language statistics are computed for all of them, and
    length ratio / correlation / misalignment statistics for every language pair.
    """
    from itertools import combinations

    cols = dataset.column_names
    langs = [l for l in _LANG_ORDER if l in cols] + [c for c in cols if c not in _LANG_ORDER]
    if len(langs) < 2:
        raise ValueError(f"Need at least 2 language columns, found: {cols}")
    pairs = list(combinations(langs, 2))

    def name(l):
        return _LANG_STYLE.get(l, (l.upper(), 'gray', 'lightgray'))[0]

    def color(l):
        return _LANG_STYLE.get(l, (l.upper(), 'gray', 'lightgray'))[1]

    def light(l):
        return _LANG_STYLE.get(l, (l.upper(), 'gray', 'lightgray'))[2]

    print(f"Extracting data (languages: {', '.join(langs)}):")
    texts = {l: dataset[l] for l in langs}

    print("Compute statistics:")
    wc = {l: [word_count(t) for t in texts[l]] for l in langs}
    cc = {l: [char_count(t) for t in texts[l]] for l in langs}

    print("Vocabulary analysis:")
    # Stream tokens (don't keep every token in memory: matters for 100M-word corpora)
    vocab, n_tokens = {}, {}
    for l in langs:
        v, n = set(), 0
        for t in texts[l]:
            toks = tokenize_text(t)
            n += len(toks)
            v.update(toks)
        vocab[l], n_tokens[l] = v, n
    ttr = {l: len(vocab[l]) / n_tokens[l] if n_tokens[l] else 0 for l in langs}

    print("Length ratios and correlation:")
    ratios, correlation, extreme = {}, {}, {}
    for a, b in pairs:
        r = [x / y for x, y in zip(cc[a], cc[b]) if y > 0 and x > 0]
        ratios[(a, b)] = r
        extreme[(a, b)] = sum(1 for v in r if v < 0.3 or v > 3.0)
        correlation[(a, b)] = np.corrcoef(cc[a], cc[b])[0, 1] if len(cc[a]) > 1 else float('nan')

    print("Quality indicators:")
    empty = {l: sum(1 for t in texts[l] if len(t.strip()) == 0) for l in langs}
    very_short = {l: sum(1 for c in cc[l] if c < 3) for l in langs}
    very_long = {l: sum(1 for c in cc[l] if c > 500) for l in langs}

    # ============================================================
    # VISUALIZATIONS (save only, no display)
    # ============================================================
    print("VISUALIZATIONS")
    ncols = max(len(langs), len(pairs), 2)
    fig, axes = plt.subplots(4, ncols, figsize=(5 * ncols, 18), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')

    def on(row, col):
        ax = axes[row, col]
        ax.axis('on')
        return ax

    # Row 0: character length distribution per language
    for i, l in enumerate(langs):
        ax = on(0, i)
        ax.hist(cc[l], bins=50, alpha=0.7, color=color(l), edgecolor='black')
        ax.axvline(np.mean(cc[l]), color='red', linestyle='--', label=f'Mean: {np.mean(cc[l]):.1f}')
        ax.axvline(np.median(cc[l]), color='green', linestyle='--', label=f'Median: {np.median(cc[l]):.1f}')
        ax.set_xlabel('Character Length')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{name(l)} - Character Length Distribution')
        ax.legend()

    # Row 1: pairwise char-length scatter
    for i, (a, b) in enumerate(pairs):
        ax = on(1, i)
        ax.scatter(cc[a], cc[b], alpha=0.1, s=1)
        m = max(max(cc[a]), max(cc[b]))
        ax.plot([0, m], [0, m], 'r--', label='y=x')
        ax.set_xlabel(f'{name(a)} Character Length')
        ax.set_ylabel(f'{name(b)} Character Length')
        ax.set_title(f'{a.upper()} vs {b.upper()} Length (corr={correlation[(a, b)]:.3f})')
        ax.legend()

    # Row 2: pairwise length ratio distribution
    for i, (a, b) in enumerate(pairs):
        ax = on(2, i)
        r = ratios[(a, b)]
        if r:
            ax.hist(r, bins=50, alpha=0.7, color='purple', edgecolor='black')
            ax.axvline(1.0, color='red', linestyle='--', label='Ratio = 1.0')
            ax.axvline(np.mean(r), color='green', linestyle='--', label=f'Mean: {np.mean(r):.2f}')
            ax.legend()
        ax.set_xlabel(f'Length Ratio ({a.upper()}/{b.upper()})')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{a.upper()}/{b.upper()} Length Ratio Distribution')
        ax.set_xlim(0, 5)

    # Row 3: word count comparison + box plot (all languages)
    ax = on(3, 0)
    for l in langs:
        ax.hist(wc[l], bins=30, alpha=0.5, color=color(l), label=name(l), edgecolor='black')
    ax.set_xlabel('Words per Sentence')
    ax.set_ylabel('Frequency')
    ax.set_title(f"Word Count Distribution ({' vs '.join(l.upper() for l in langs)})")
    ax.legend()

    ax = on(3, 1)
    bp = ax.boxplot([cc[l] for l in langs], labels=[name(l) for l in langs], patch_artist=True)
    for box, l in zip(bp['boxes'], langs):
        box.set_facecolor(light(l))
    ax.set_ylabel('Character Length')
    ax.set_title('Character Length Box Plot')

    plt.tight_layout()
    plot_path = _analysis_plot_path(f"dataset_analysis_{dataset_name}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ============================================================
    # SUMMARY TABLE
    # ============================================================
    print("SUMMARY TABLE")
    metrics = [
        'Total Samples', 'Total Words', 'Total Characters', 'Vocabulary Size',
        'Type-Token Ratio', 'Avg Words/Sentence', 'Avg Chars/Sentence',
        'Median Chars/Sentence', 'Std Dev (chars)', 'Min Chars', 'Max Chars',
        'Empty Sentences', 'Very Short (<3 chars)', 'Very Long (>500 chars)',
    ]
    summary_data = {'Metric': metrics}
    for l in langs:
        summary_data[name(l)] = [
            f"{len(dataset):,}",
            f"{sum(wc[l]):,}",
            f"{sum(cc[l]):,}",
            f"{len(vocab[l]):,}",
            f"{ttr[l]:.4f}",
            f"{np.mean(wc[l]):.2f}",
            f"{np.mean(cc[l]):.2f}",
            f"{np.median(cc[l]):.2f}",
            f"{np.std(cc[l]):.2f}",
            f"{np.min(cc[l])}",
            f"{np.max(cc[l])}",
            f"{empty[l]:,}",
            f"{very_short[l]:,}",
            f"{very_long[l]:,}",
        ]

    summary_df = pd.DataFrame(summary_data)
    print(f"\n{'=' * 60}")
    print(f"DATASET SUMMARY: {dataset_name}")
    print(f"{'=' * 60}")
    print(summary_df.to_string(index=False))

    print(f"\n{'=' * 60}")
    print("PAIRWISE ALIGNMENT STATS")
    print(f"{'=' * 60}")
    for a, b in pairs:
        n = len(ratios[(a, b)])
        pct = 100 * extreme[(a, b)] / n if n else 0.0
        mean_r = np.mean(ratios[(a, b)]) if n else float('nan')
        print(
            f"{a.upper()}-{b.upper()}: char-length corr={correlation[(a, b)]:.4f} | "
            f"mean ratio={mean_r:.2f} | potential misalignments "
            f"(ratio <0.3 or >3.0)={extreme[(a, b)]:,} ({pct:.2f}%)"
        )
    print(f"\nVisualization saved to: {plot_path}")

    # ============================================================
    # SAMPLE DATA INSPECTION
    # ============================================================
    def show(idx, limit=None):
        row = dataset[int(idx)]
        for l in langs:
            t = row[l] if limit is None else row[l][:limit] + '...'
            print(f"  {l.upper()}: {t!r}" if limit is not None else f"{l.upper()}: {t}")

    print("\n" + "=" * 60)
    print("SAMPLE SENTENCE TUPLES")
    print("=" * 60)
    np.random.seed(42)
    sample_indices = np.random.choice(len(dataset), min(5, len(dataset)), replace=False)
    for i, idx in enumerate(sample_indices):
        print(f"\n--- Sample {i+1} (idx={idx}) ---")
        row = dataset[int(idx)]
        for l in langs:
            print(f"{l.upper()}: {row[l]}")

    print("\n" + "=" * 60)
    print("EDGE CASES")
    print("=" * 60)
    for l in langs:
        s_idx, l_idx = int(np.argmin(cc[l])), int(np.argmax(cc[l]))
        print(f"\nShortest {name(l)} sentence (idx={s_idx}):")
        show(s_idx, limit=1000)
        print(f"\nLongest {name(l)} sentence (idx={l_idx}):")
        show(l_idx, limit=1000)


def analyze_single_language_dataset(dataset, dataset_name, language='en'):
    """
    Analyze a single-language dataset and save visualizations.
    
    Args:
        dataset: HuggingFace Dataset with 'text' column (or 'en'/'tr' column)
        dataset_name: Name for output files
        language: Language code ('en' for English, 'tr' for Turkish)
    """
    # Determine the text column name
    if 'text' in dataset.column_names:
        text_column = 'text'
    elif language in dataset.column_names:
        text_column = language
    else:
        raise ValueError(f"Dataset must have 'text' or '{language}' column. Found: {dataset.column_names}")
    
    # Language display names and colors
    lang_info = {
        'en': {'name': 'English', 'color': 'blue', 'light_color': 'lightblue'},
        'tr': {'name': 'Turkish', 'color': 'orange', 'light_color': 'wheat'},
        'de': {'name': 'German', 'color': 'green', 'light_color': 'lightgreen'},
        'deu': {'name': 'German', 'color': 'green', 'light_color': 'lightgreen'},
        'eng': {'name': 'English', 'color': 'blue', 'light_color': 'lightblue'},
        'tur': {'name': 'Turkish', 'color': 'orange', 'light_color': 'wheat'},
    }
    
    lang_name = lang_info.get(language, {'name': language.upper(), 'color': 'gray', 'light_color': 'lightgray'})['name']
    lang_color = lang_info.get(language, {'name': language.upper(), 'color': 'gray', 'light_color': 'lightgray'})['color']
    lang_light_color = lang_info.get(language, {'name': language.upper(), 'color': 'gray', 'light_color': 'lightgray'})['light_color']
    
    print(f"Analyzing {lang_name} dataset: {dataset_name}")
    # Extract data
    print("Extracting data...")
    texts = dataset[text_column]
    
    # Compute statistics
    word_counts = [word_count(t) for t in texts]
    char_counts = [char_count(t) for t in texts]
    
    total_words = sum(word_counts)
    total_chars = sum(char_counts)
    
    # Vocabulary analysis (with proper tokenization - removes punctuation, filters digits)
    print("Vocabulary analysis...")
    all_words = []
    for text in texts:
        all_words.extend(tokenize_text(text))
    vocab = set(all_words)
    ttr = len(vocab) / len(all_words) if len(all_words) > 0 else 0
    
    # Quality indicators
    empty_count = sum(1 for t in texts if len(t.strip()) == 0)
    very_short = sum(1 for c in char_counts if c < 3)
    very_long = sum(1 for c in char_counts if c > 500)
    
    # Word frequency analysis
    print("Word frequency analysis...")
    word_freq = Counter(all_words)
    top_words = word_freq.most_common(20)
    
    # ============================================================
    # VISUALIZATIONS (save only, no display)
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Character length distribution
    axes[0, 0].hist(char_counts, bins=50, alpha=0.7, color=lang_color, edgecolor='black')
    axes[0, 0].axvline(np.mean(char_counts), color='red', linestyle='--', label=f'Mean: {np.mean(char_counts):.1f}')
    axes[0, 0].axvline(np.median(char_counts), color='green', linestyle='--', label=f'Median: {np.median(char_counts):.1f}')
    axes[0, 0].set_xlabel('Character Length')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'{lang_name} - Character Length Distribution')
    axes[0, 0].legend()
    
    # 2. Word count distribution
    axes[0, 1].hist(word_counts, bins=50, alpha=0.7, color=lang_color, edgecolor='black')
    axes[0, 1].axvline(np.mean(word_counts), color='red', linestyle='--', label=f'Mean: {np.mean(word_counts):.1f}')
    axes[0, 1].axvline(np.median(word_counts), color='green', linestyle='--', label=f'Median: {np.median(word_counts):.1f}')
    axes[0, 1].set_xlabel('Words per Sentence')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'{lang_name} - Word Count Distribution')
    axes[0, 1].legend()
    
    # 3. Top 20 words bar chart
    if top_words:
        words, counts = zip(*top_words)
        axes[0, 2].barh(range(len(words)), counts, color=lang_color, edgecolor='black')
        axes[0, 2].set_yticks(range(len(words)))
        axes[0, 2].set_yticklabels(words)
        axes[0, 2].invert_yaxis()
        axes[0, 2].set_xlabel('Frequency')
        axes[0, 2].set_title(f'{lang_name} - Top 20 Words')
    
    # 4. Log-scale character length distribution
    axes[1, 0].hist(char_counts, bins=50, alpha=0.7, color=lang_color, edgecolor='black', log=True)
    axes[1, 0].set_xlabel('Character Length')
    axes[1, 0].set_ylabel('Frequency (log scale)')
    axes[1, 0].set_title(f'{lang_name} - Character Length (Log Scale)')
    
    # 5. Cumulative distribution
    sorted_chars = np.sort(char_counts)
    cumulative = np.arange(1, len(sorted_chars) + 1) / len(sorted_chars)
    axes[1, 1].plot(sorted_chars, cumulative, color=lang_color, linewidth=2)
    axes[1, 1].axhline(0.5, color='red', linestyle='--', alpha=0.5, label='50th percentile')
    axes[1, 1].axhline(0.9, color='orange', linestyle='--', alpha=0.5, label='90th percentile')
    axes[1, 1].set_xlabel('Character Length')
    axes[1, 1].set_ylabel('Cumulative Proportion')
    axes[1, 1].set_title(f'{lang_name} - Cumulative Distribution')
    axes[1, 1].legend()
    
    # 6. Box plot
    bp = axes[1, 2].boxplot([char_counts], labels=[lang_name], patch_artist=True)
    bp['boxes'][0].set_facecolor(lang_light_color)
    axes[1, 2].set_ylabel('Character Length')
    axes[1, 2].set_title(f'{lang_name} - Character Length Box Plot')
    
    plt.tight_layout()
    output_path = _analysis_plot_path(f"dataset_analysis_{dataset_name}_{language}.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # ============================================================
    # SUMMARY TABLE
    # ============================================================
    print("Generating summary...")
    summary_data = {
        'Metric': [
            'Total Samples',
            'Total Words',
            'Total Characters',
            'Vocabulary Size',
            'Type-Token Ratio',
            'Avg Words/Sentence',
            'Avg Chars/Sentence',
            'Median Chars/Sentence',
            'Std Dev (chars)',
            'Min Chars',
            'Max Chars',
            '25th Percentile (chars)',
            '75th Percentile (chars)',
            '90th Percentile (chars)',
            'Empty Sentences',
            'Very Short (<3 chars)',
            'Very Long (>500 chars)',
        ],
        lang_name: [
            f"{len(dataset):,}",
            f"{total_words:,}",
            f"{total_chars:,}",
            f"{len(vocab):,}",
            f"{ttr:.4f}",
            f"{np.mean(word_counts):.2f}",
            f"{np.mean(char_counts):.2f}",
            f"{np.median(char_counts):.2f}",
            f"{np.std(char_counts):.2f}",
            f"{np.min(char_counts)}",
            f"{np.max(char_counts)}",
            f"{np.percentile(char_counts, 25):.2f}",
            f"{np.percentile(char_counts, 75):.2f}",
            f"{np.percentile(char_counts, 90):.2f}",
            f"{empty_count:,}",
            f"{very_short:,}",
            f"{very_long:,}",
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    print(f"\n{'=' * 60}")
    print(f"DATASET SUMMARY: {dataset_name} ({lang_name})")
    print(f"{'=' * 60}")
    print(summary_df.to_string(index=False))
    print(f"\nVisualization saved to: {output_path}")
    
    # ============================================================
    # TOP WORDS
    # ============================================================
    print("\n" + "=" * 60)
    print(f"TOP 20 WORDS ({lang_name})")
    print("=" * 60)
    for i, (word, count) in enumerate(top_words, 1):
        print(f"  {i:2d}. {word}: {count:,}")
    
    # ============================================================
    # SAMPLE DATA INSPECTION
    # ============================================================
    print("\n" + "=" * 60)
    print("SAMPLE SENTENCES")
    print("=" * 60)
    
    np.random.seed(42)
    sample_indices = np.random.choice(len(dataset), min(5, len(dataset)), replace=False)
    
    for i, idx in enumerate(sample_indices):
        print(f"\n--- Sample {i+1} (idx={idx}) ---")
        text = dataset[int(idx)][text_column]
        print(f"  {text[:500]}{'...' if len(text) > 500 else ''}")
    
    # Edge cases
    print("\n" + "=" * 60)
    print("EDGE CASES")
    print("=" * 60)
    
    shortest_idx = np.argmin(char_counts)
    longest_idx = np.argmax(char_counts)
    
    print(f"\nShortest sentence (idx={shortest_idx}, {char_counts[shortest_idx]} chars):")
    print(f"  '{dataset[int(shortest_idx)][text_column]}'")
    
    print(f"\nLongest sentence (idx={longest_idx}, {char_counts[longest_idx]} chars):")
    print(f"  '{dataset[int(longest_idx)][text_column][:1000]}...'")
    
    return {
        'total_samples': len(dataset),
        'total_words': total_words,
        'total_chars': total_chars,
        'vocab_size': len(vocab),
        'ttr': ttr,
        'avg_words': np.mean(word_counts),
        'avg_chars': np.mean(char_counts),
        'median_chars': np.median(char_counts),
    }