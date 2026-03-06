"""
TinyStories Data Preparation for nanoGPT
==========================================
Downloads TinyStories from HuggingFace, tokenises with tiktoken GPT-2 BPE,
and writes train.bin / val.bin.

TinyStories contains ~2.1M simple short stories written in basic English,
designed by Eldan and Li (2023) to train small language models.

When used together with ROCStories, TinyStories provides additional narrative
data that helps the model learn story structure and basic English before
specializing on the 5-sentence ROCStories format.

Dataset:  https://huggingface.co/datasets/roneneldan/TinyStories
Tokeniser: tiktoken gpt2 (vocab = 50257; stored as uint16 in .bin files)

Usage:
    python data/tinystories/prepare.py

Outputs:
    data/tinystories/train.bin   (uint16 numpy array)
    data/tinystories/val.bin     (uint16 numpy array)
"""

import os
import sys
import numpy as np
import tiktoken

# ── Hyper-parameters ──────────────────────────────────────────────────────────
VAL_FRACTION  = 0.05     # 5% validation (TinyStories is large)
SEED          = 42
MAX_STORIES   = 500000   # Cap at 500K stories to keep data manageable
HF_DATASET    = "roneneldan/TinyStories"
# ─────────────────────────────────────────────────────────────────────────────


def load_tinystories():
    """Load TinyStories from HuggingFace. Returns a list of story strings."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not found.")
        print("  Install it with:  pip install datasets")
        sys.exit(1)

    print(f"[prepare] Loading TinyStories from {HF_DATASET}...")
    print(f"[prepare] (This may take a few minutes on first download)")

    stories = []
    try:
        ds = load_dataset(HF_DATASET, split="train", trust_remote_code=True)
        for row in ds:
            # TinyStories has a 'text' column
            text = (row.get('text', '') or '').strip()
            if text and len(text.split()) >= 10:
                stories.append(text)
                if len(stories) >= MAX_STORIES:
                    break
        print(f"[prepare] Loaded {len(stories):,} stories from TinyStories")
    except Exception as e:
        print(f"  [error] Could not load TinyStories: {e}")
        sys.exit(1)

    return stories


def tokenise_and_save(stories, out_dir):
    """Tokenise all stories and write train.bin / val.bin."""
    enc = tiktoken.get_encoding("gpt2")
    eot = 50256  # endoftext token id for GPT-2

    # Shuffle and split
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(stories))
    n_val   = max(1, int(len(stories) * VAL_FRACTION))
    val_idx   = indices[:n_val]
    train_idx = indices[n_val:]

    train_stories = [stories[i] for i in train_idx]
    val_stories   = [stories[i] for i in val_idx]

    print(f"[prepare] Split: {len(train_stories):,} train | {len(val_stories):,} val")

    def encode_split(split_stories, name):
        all_tokens = []
        for story in split_stories:
            tokens = enc.encode_ordinary(story)
            all_tokens.extend(tokens)
            all_tokens.append(eot)
        arr = np.array(all_tokens, dtype=np.uint16)
        path = os.path.join(out_dir, f"{name}.bin")
        arr.tofile(path)
        print(f"[prepare] {name}.bin: {len(arr):,} tokens  "
              f"(avg {len(arr)/len(split_stories):.1f} tok/story)  -> {path}")
        return arr

    train_arr = encode_split(train_stories, 'train')
    val_arr   = encode_split(val_stories,   'val')

    print(f"\n[prepare] Done!")
    print(f"  vocab_size : 50257 (GPT-2 tiktoken BPE)")
    print(f"  train size : {len(train_arr)/1e6:.2f}M tokens")
    print(f"  val size   : {len(val_arr)/1e3:.1f}K tokens")
    print(f"  train/val split: {100*(1-VAL_FRACTION):.0f}% / {100*VAL_FRACTION:.0f}%")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    stories = load_tinystories()
    tokenise_and_save(stories, script_dir)
