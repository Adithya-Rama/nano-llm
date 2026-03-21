"""
ROCStories Data Preparation for nanoGPT
========================================
Downloads ROCStories from HuggingFace, formats each story as a single text block,
tokenises with tiktoken GPT-2 BPE, and writes train.bin / val.bin.

Dataset:  https://huggingface.co/datasets/mintujupally/ROCStories
Tokeniser: tiktoken gpt2 (vocab = 50257; stored as uint16 in .bin files)

Supports TWO formats:

  Plain format (default -- used for Task 1 + Task 3):
      [Title]\n Sentence1 Sentence2 ... Sentence5

  Structured format (--structured flag -- used for Task 2 experiment):
      <story>
      <title>Title</title>
      <s1>Sentence1</s1>
      ...
      <s5>Sentence5</s5>
      </story>

Usage:
    # Plain format (default):
    python data/rocstories/prepare.py

    # Structured format (for Task 2 ablation):
    python data/rocstories/prepare.py --structured

    # Custom output directory:
    python data/rocstories/prepare.py --structured --out_dir data/rocstories_structured

Outputs:
    data/rocstories/train.bin   (plain, default — all downloaded stories)
    data/rocstories/val.bin     (tokenised eval_stories.txt only — no overlap with train)
"""

import os
import sys
import re
import argparse
import numpy as np
import tiktoken

# -- Hyper-parameters ---------------------------------------------------------
SEED          = 42
EVAL_STORIES_FILE = 'eval_stories.txt'  # held-out stories for val.bin (must not overlap train)
MIN_WORDS     = 10      # filter stories that are suspiciously short

# Primary: plain-text dataset (each row is a full story in one text field)
HF_DATASET    = "mintujupally/ROCStories"

# Fallback: tabular dataset with individual sentence columns
HF_FALLBACK   = "Sharathhebbar24/ROCStories"
# -----------------------------------------------------------------------------


def _split_into_sentences(text):
    """Heuristically split a paragraph into sentences on . ! ?"""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def _row_to_plain_text_format(row):
    """
    Convert one dataset row to plain text story.
    Handles two dataset schemas:
      1. Plain-text schema: single 'text' column with full story
      2. Tabular schema:    sentence1, sentence2, ..., sentence5 columns
    """
    # Schema 1: plain text (mintujupally/ROCStories)
    raw = (row.get('text', '') or '').strip()
    if raw:
        return raw  # already a full story string

    # Schema 2: tabular sentences (Sharathhebbar24/ROCStories)
    sentences = []
    for key in ['sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5',
                'Sentence1', 'Sentence2', 'Sentence3', 'Sentence4', 'Sentence5']:
        val = row.get(key, '') or ''
        val = val.strip()
        if val:
            sentences.append(val)

    if sentences:
        title = (row.get('storytitle', '') or row.get('title', '') or
                 row.get('StoryTitle', '') or '').strip()
        story_body = ' '.join(sentences)
        if title:
            return f"{title}\n{story_body}"
        return story_body

    # Schema 3: single 'story' column
    story_col = (row.get('story', '') or '').strip()
    if story_col:
        return story_col

    return ''


def _row_to_structured(row):
    """
    Convert one dataset row to structured format with explicit sentence markers.
    <story><title>...</title><s1>...</s1>...<s5>...</s5></story>
    """
    raw = (row.get('text', '') or '').strip()
    if raw:
        # Plain text: split into sentences heuristically
        sentences = _split_into_sentences(raw)
        title = ''
    else:
        # Tabular: extract individually
        title = (row.get('storytitle', '') or row.get('title', '') or '').strip()
        sentences = []
        for key in ['sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5']:
            val = row.get(key, '') or ''
            val = val.strip()
            if val:
                sentences.append(val)

    if not sentences:
        return ''

    parts = ["<story>"]
    if title:
        parts.append(f"<title>{title}</title>")
    for i, sent in enumerate(sentences[:5], 1):
        parts.append(f"<s{i}>{sent}</s{i}>")
    parts.append("</story>")
    return "\n".join(parts)


def load_rocstories(structured=False):
    """Load ROCStories from HuggingFace. Returns a list of story strings."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not found.")
        print("  Install it with:  pip install datasets")
        sys.exit(1)

    formatter = _row_to_structured if structured else _row_to_plain_text_format
    format_name = "structured" if structured else "plain"

    stories = []
    for slug in [HF_DATASET, HF_FALLBACK]:
        try:
            print(f"[prepare] Trying to load: {slug} ...")
            # NOTE: trust_remote_code removed — deprecated in datasets >= 2.20
            ds = load_dataset(slug, split="train")
            print(f"[prepare] Columns: {ds.column_names}")
            for row in ds:
                text = formatter(row)
                if text and len(text.split()) >= MIN_WORDS:
                    stories.append(text)
            if stories:
                print(f"[prepare] Loaded {len(stories):,} stories from {slug} ({format_name} format)")
                return stories
            else:
                print(f"  [warn] {slug} loaded but extracted 0 stories — unexpected schema, trying fallback")
        except Exception as e:
            print(f"  [warn] Could not load {slug}: {e}")

    # Last resort: try to read a local CSV (ROCStories is also distributed as CSV)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for fname in ['roc_train.csv', 'ROCStories_winter2017.csv',
                  'ROCStories_spring2016.csv', 'rocstories_train.csv']:
        csv_path = os.path.join(script_dir, fname)
        if os.path.exists(csv_path):
            print(f"[prepare] Reading local CSV: {csv_path}")
            import csv as csv_mod
            with open(csv_path, newline='', encoding='utf-8') as f:
                reader = csv_mod.DictReader(f)
                for row in reader:
                    text = formatter(row)
                    if text and len(text.split()) >= MIN_WORDS:
                        stories.append(text)
            if stories:
                print(f"[prepare] Loaded {len(stories):,} stories from local CSV ({format_name} format)")
                return stories

    raise RuntimeError(
        "Could not load ROCStories from HuggingFace or local CSV.\n"
        "Options:\n"
        "  1. Run:  pip install datasets  then retry\n"
        "  2. Download the CSV from https://cs.rochester.edu/nlp/rocstories/ "
        "and place it in data/rocstories/\n"
    )


def _load_eval_stories_holdout(out_dir):
    """Load professor / assignment eval stories — never part of HF train download."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for path in (
        os.path.join(out_dir, EVAL_STORIES_FILE),
        os.path.join(script_dir, EVAL_STORIES_FILE),
    ):
        if os.path.isfile(path):
            with open(path, encoding='utf-8') as f:
                raw = f.read()
            parts = [p.strip() for p in raw.split('\n\n') if p.strip()]
            return parts, path
    return None, None


def tokenise_and_save(stories, out_dir):
    """Tokenise all stories and write train.bin / val.bin."""
    enc = tiktoken.get_encoding("gpt2")
    eot = 50256  # endoftext token id for GPT-2

    os.makedirs(out_dir, exist_ok=True)

    # Shuffle — train on ALL HF stories (no holdout from this pool for val)
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(stories))
    train_stories = [stories[i] for i in indices]

    # Val = separate file only (eval_stories.txt). Never overlaps train — avoids contaminated PPL.
    val_stories, eval_path = _load_eval_stories_holdout(out_dir)
    if val_stories is None:
        raise RuntimeError(
            f"Missing {EVAL_STORIES_FILE!r} (looked next to output dir and in data/rocstories/). "
            "Place the professor/assignment eval stories file there so val.bin is a clean holdout."
        )

    print(f"[prepare] Split: {len(train_stories):,} train (full corpus) | "
          f"{len(val_stories):,} val (from {os.path.basename(eval_path)} — no train overlap)")

    # Print a sample story for verification
    print(f"\n[prepare] Sample story:\n{stories[0][:300]}\n")

    def encode_split(split_stories, name):
        all_tokens = []
        for story in split_stories:
            tokens = enc.encode_ordinary(story)   # fast, no special-token handling
            all_tokens.extend(tokens)
            all_tokens.append(eot)                # story separator
        arr = np.array(all_tokens, dtype=np.uint16)
        path = os.path.join(out_dir, f"{name}.bin")
        arr.tofile(path)
        print(f"[prepare] {name}.bin: {len(arr):,} tokens  "
              f"(avg {len(arr)/len(split_stories):.1f} tok/story)  -> {path}")
        return arr

    train_arr = encode_split(train_stories, 'train')
    val_arr   = encode_split(val_stories,   'val')

    # Save a few example stories for inspection
    examples_path = os.path.join(out_dir, 'examples.txt')
    with open(examples_path, 'w', encoding='utf-8') as f:
        for story in stories[:5]:
            f.write(story + "\n\n---\n\n")
    print(f"[prepare] Saved 5 example stories to {examples_path}")

    print(f"\n[prepare] Done!")
    print(f"  vocab_size : 50257 (GPT-2 tiktoken BPE)")
    print(f"  train size : {len(train_arr)/1e6:.2f}M tokens")
    print(f"  val size   : {len(val_arr)/1e3:.1f}K tokens")
    print(f"  val source: {EVAL_STORIES_FILE} (held out — not in train.bin)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare ROCStories for nanoGPT')
    parser.add_argument('--structured', action='store_true',
                        help='Use structured format with <story><s1>...</s1>...</story> tags')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output directory (default: script directory, or data/rocstories_structured if --structured)')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.out_dir:
        out_dir = args.out_dir
    elif args.structured:
        out_dir = os.path.join(os.path.dirname(script_dir), 'rocstories_structured')
    else:
        out_dir = script_dir

    stories = load_rocstories(structured=args.structured)
    tokenise_and_save(stories, out_dir)
