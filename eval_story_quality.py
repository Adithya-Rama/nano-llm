"""
eval_story_quality.py — Compute story quality metrics
======================================================
Analyses generated stories (from sample_batch.py JSONL output or plain text)
and computes quantitative metrics useful for Task 2 conclusions.

Metrics computed:
  - Average story length (words)
  - Average sentence count
  - Repetition ratio (fraction of repeated n-grams)
  - Unique token percentage
  - Vocabulary diversity (type-token ratio)
  - Average sentence length (words)
  - Stories with proper endings (ends with . ! ?)

Usage:
    # From JSONL output of sample_batch.py:
    python eval_story_quality.py --input out-rocstories/generated_stories.jsonl

    # From plain text file (stories separated by --- or blank lines):
    python eval_story_quality.py --input generated.txt --format text

    # Compare multiple model outputs:
    python eval_story_quality.py \
        --input out-rocstories-baseline/generated_stories.jsonl \
        --input out-rocstories/generated_stories.jsonl \
        --labels "Baseline,Full Modern"
"""

import argparse
import json
import os
import re
import sys
from collections import Counter


def load_stories_jsonl(path):
    """Load stories from a JSONL file (output of sample_batch.py)."""
    stories = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = record.get('generated_text', '')
            if text.strip():
                stories.append(text.strip())
    return stories


def load_stories_text(path):
    """Load stories from a plain text file (separated by --- or blank lines)."""
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split on --- or double newlines
    if '---' in content:
        parts = content.split('---')
    else:
        parts = re.split(r'\n\s*\n', content)

    stories = [p.strip() for p in parts if p.strip()]
    return stories


def compute_metrics(stories):
    """Compute quality metrics for a list of story strings."""
    if not stories:
        return {}

    all_words = []
    all_lengths = []
    all_sent_counts = []
    all_sent_lengths = []
    proper_endings = 0
    total_bigrams = 0
    repeated_bigrams = 0
    total_trigrams = 0
    repeated_trigrams = 0

    for story in stories:
        words = story.split()
        all_words.extend(words)
        all_lengths.append(len(words))

        # Sentence count (split on . ! ?)
        sentences = re.split(r'(?<=[.!?])\s+', story)
        sentences = [s.strip() for s in sentences if s.strip()]
        all_sent_counts.append(len(sentences))

        for sent in sentences:
            all_sent_lengths.append(len(sent.split()))

        # Proper ending check
        stripped = story.rstrip()
        if stripped and stripped[-1] in '.!?':
            proper_endings += 1

        # Bigram repetition
        if len(words) >= 2:
            bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
            bigram_counts = Counter(bigrams)
            total_bigrams += len(bigrams)
            repeated_bigrams += sum(c - 1 for c in bigram_counts.values() if c > 1)

        # Trigram repetition
        if len(words) >= 3:
            trigrams = [(words[i], words[i+1], words[i+2]) for i in range(len(words)-2)]
            trigram_counts = Counter(trigrams)
            total_trigrams += len(trigrams)
            repeated_trigrams += sum(c - 1 for c in trigram_counts.values() if c > 1)

    n = len(stories)
    total_words = len(all_words)
    unique_words = len(set(w.lower() for w in all_words))

    metrics = {
        'num_stories': n,
        'avg_length_words': sum(all_lengths) / n,
        'avg_sentences': sum(all_sent_counts) / n,
        'avg_sentence_length': sum(all_sent_lengths) / max(len(all_sent_lengths), 1),
        'unique_token_pct': 100.0 * unique_words / max(total_words, 1),
        'type_token_ratio': unique_words / max(total_words, 1),
        'proper_ending_pct': 100.0 * proper_endings / n,
        'bigram_repetition_pct': 100.0 * repeated_bigrams / max(total_bigrams, 1),
        'trigram_repetition_pct': 100.0 * repeated_trigrams / max(total_trigrams, 1),
    }
    return metrics


def print_metrics(metrics, label=None):
    """Pretty-print a metrics dictionary."""
    header = f"Story Quality Metrics: {label}" if label else "Story Quality Metrics"
    print(f"\n{'='*60}")
    print(f"  {header}")
    print(f"{'='*60}")
    print(f"  Stories analysed       : {metrics['num_stories']}")
    print(f"  Avg length (words)     : {metrics['avg_length_words']:.1f}")
    print(f"  Avg sentences          : {metrics['avg_sentences']:.1f}")
    print(f"  Avg sentence length    : {metrics['avg_sentence_length']:.1f} words")
    print(f"  Unique token %         : {metrics['unique_token_pct']:.1f}%")
    print(f"  Type-token ratio       : {metrics['type_token_ratio']:.3f}")
    print(f"  Proper endings %       : {metrics['proper_ending_pct']:.1f}%")
    print(f"  Bigram repetition %    : {metrics['bigram_repetition_pct']:.1f}%")
    print(f"  Trigram repetition %   : {metrics['trigram_repetition_pct']:.1f}%")
    print(f"{'='*60}")


def print_comparison_table(all_metrics, all_labels):
    """Print a side-by-side comparison table for the report."""
    print(f"\n{'='*70}")
    print(f"  COMPARISON TABLE (copy into report)")
    print(f"{'='*70}")

    # Header
    header = f"{'Metric':<25s}"
    for label in all_labels:
        header += f" | {label:>12s}"
    print(header)
    print("-" * len(header))

    # Rows
    metric_keys = [
        ('avg_length_words', 'Avg Length (words)'),
        ('avg_sentences', 'Avg Sentences'),
        ('unique_token_pct', 'Unique Token %'),
        ('proper_ending_pct', 'Proper Ending %'),
        ('bigram_repetition_pct', 'Bigram Repeat %'),
        ('trigram_repetition_pct', 'Trigram Repeat %'),
    ]

    for key, display_name in metric_keys:
        row = f"{display_name:<25s}"
        for m in all_metrics:
            val = m.get(key, 0)
            if 'pct' in key or 'ratio' in key:
                row += f" | {val:>11.1f}%"
            else:
                row += f" | {val:>12.1f}"
        print(row)

    print(f"{'='*70}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate story generation quality')
    parser.add_argument('--input', type=str, action='append', required=True,
                        help='Path to generated stories (JSONL or text). Can specify multiple.')
    parser.add_argument('--format', type=str, default='jsonl', choices=['jsonl', 'text'],
                        help='Input format: jsonl (from sample_batch.py) or text')
    parser.add_argument('--labels', type=str, default=None,
                        help='Comma-separated labels for comparison')
    args = parser.parse_args()

    loader = load_stories_jsonl if args.format == 'jsonl' else load_stories_text

    if args.labels:
        labels = [l.strip() for l in args.labels.split(',')]
    else:
        labels = [os.path.basename(os.path.dirname(p)) or os.path.basename(p) for p in args.input]

    all_metrics = []
    for path, label in zip(args.input, labels):
        if not os.path.exists(path):
            print(f"WARNING: File not found: {path}")
            continue
        stories = loader(path)
        metrics = compute_metrics(stories)
        all_metrics.append(metrics)
        print_metrics(metrics, label=label)

    if len(all_metrics) > 1:
        print_comparison_table(all_metrics, labels)