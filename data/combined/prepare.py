"""
Combined Data Preparation — Task 4 Arena Corpus
================================================
Builds a large multi-source narrative corpus for training the 152M arena model.

Default sources (always included):
  • ROCStories (~7.8M tokens)   — core task domain
  • TinyStories (~200M tokens)  — simple English narrative (streaming, OOM-safe)

Optional sources (add via flags):
  • WritingPrompts (~180M tokens)  — longer Reddit stories, exercises 512-ctx window
  • Children's stories (~25M tokens) — strong entity tracking, cause-effect chains

Usage:
  # Minimal corpus (~207M tokens):
  python data/combined/prepare.py

  # Dry run (check token counts without writing):
  python data/combined/prepare.py --dry_run

  # Full corpus (~412M tokens, recommended for arena):
  python data/combined/prepare.py --with_writing_prompts --with_childrens

Prerequisites:
  python data/rocstories/prepare.py
  python data/tinystories/prepare.py

Outputs:
  data/combined/train.bin   (uint16, concatenated and shuffled at story level)
  data/combined/val.bin     (uint16)

Token length filter:
  Stories with < 20 tokens or > 2048 tokens are excluded to remove corrupt
  entries and stories too long for the 512-token context window.
"""

import os
import sys
import argparse
import numpy as np

# Token length filter thresholds
MIN_TOKENS = 20
MAX_TOKENS = 2048

# EOT separator (GPT-2 <|endoftext|>)
EOT = 50256

SEED = 42


def _count_tokens(path):
    """Return token count of a .bin file without loading all of it."""
    return os.path.getsize(path) // 2  # uint16 = 2 bytes per token


def _load_bin(path):
    return np.fromfile(path, dtype=np.uint16)


def _split_stories(tokens: np.ndarray):
    """Split a token array at EOT boundaries into a list of story token arrays.

    Each returned array includes the trailing EOT token.
    """
    eot_positions = np.where(tokens == EOT)[0]
    stories = []
    start = 0
    for end in eot_positions:
        story = tokens[start : end + 1]  # include EOT
        n = len(story)
        if MIN_TOKENS <= n <= MAX_TOKENS:
            stories.append(story)
        start = end + 1
    return stories


def _stream_tinystories(path, max_tokens=None):
    """Load TinyStories in chunks to avoid OOM (file can be > 400MB).

    Returns a list of story token arrays, capped at max_tokens if specified.
    """
    print(f"[combine] Streaming TinyStories from {path} …")
    arr = _load_bin(path)
    if max_tokens is not None and len(arr) > max_tokens:
        arr = arr[:max_tokens]
        print(f"[combine]   Capped at {len(arr)/1e6:.1f}M tokens")
    stories = _split_stories(arr)
    print(f"[combine]   TinyStories: {len(stories):,} stories after length filter")
    return stories


def _load_stories_from_bin(path, name):
    """Load a .bin file and split into filtered stories."""
    arr = _load_bin(path)
    stories = _split_stories(arr)
    print(f"[combine]   {name}: {len(stories):,} stories ({len(arr)/1e6:.1f}M raw tokens)")
    return stories


def _load_writing_prompts():
    """Load WritingPrompts from HuggingFace datasets (Eureka-Moment/writing-prompts)."""
    try:
        from datasets import load_dataset
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        print("[combine] Downloading WritingPrompts …")
        ds = load_dataset("Eureka-Moment/writing-prompts", split="train", trust_remote_code=True)
        stories = []
        for row in ds:
            text = row.get("story", row.get("text", "")).strip()
            if not text:
                continue
            toks = enc.encode_ordinary(text) + [EOT]
            if MIN_TOKENS <= len(toks) <= MAX_TOKENS:
                stories.append(np.array(toks, dtype=np.uint16))
        print(f"[combine]   WritingPrompts: {len(stories):,} stories after length filter")
        return stories
    except Exception as e:
        print(f"[combine] ⚠  Could not load WritingPrompts: {e}")
        return []


def _load_childrens_stories():
    """Load children's stories from HuggingFace (ajibawa-2023/childrens-stories-collection)."""
    try:
        from datasets import load_dataset
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        print("[combine] Downloading Children's Stories …")
        ds = load_dataset("ajibawa-2023/childrens-stories-collection",
                          split="train", trust_remote_code=True)
        stories = []
        for row in ds:
            text = row.get("story", row.get("text", "")).strip()
            if not text:
                continue
            toks = enc.encode_ordinary(text) + [EOT]
            if MIN_TOKENS <= len(toks) <= MAX_TOKENS:
                stories.append(np.array(toks, dtype=np.uint16))
        print(f"[combine]   Children's: {len(stories):,} stories after length filter")
        return stories
    except Exception as e:
        print(f"[combine] ⚠  Could not load Children's stories: {e}")
        return []


def build_combined(with_writing_prompts=False, with_childrens=False, dry_run=False):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir   = os.path.dirname(script_dir)
    rng = np.random.default_rng(SEED)

    print("[combine] === Task 4 Arena Corpus Builder ===")

    all_train_stories = []
    all_val_stories   = []

    # ── ROCStories ────────────────────────────────────────────────────────────
    roc_train = os.path.join(data_dir, 'rocstories', 'train.bin')
    roc_val   = os.path.join(data_dir, 'rocstories', 'val.bin')
    if os.path.exists(roc_train):
        _roc = _load_stories_from_bin(roc_train, "ROCStories train")
        all_train_stories.extend(_roc * 5)   # 5x upsample: ROCStories → 9% of corpus (20.5M / 220M)
        print(f"[combine]   ROCStories 5x upsampled: {len(_roc)*5:,} story copies")
    else:
        print("[combine] ⚠  ROCStories train.bin not found — run data/rocstories/prepare.py")
    if os.path.exists(roc_val):
        all_val_stories.extend(_load_stories_from_bin(roc_val, "ROCStories val"))

    # ── TinyStories (streaming) ───────────────────────────────────────────────
    tiny_train = os.path.join(data_dir, 'tinystories', 'train.bin')
    tiny_val   = os.path.join(data_dir, 'tinystories', 'val.bin')
    if os.path.exists(tiny_train):
        all_train_stories.extend(_stream_tinystories(tiny_train, max_tokens=200_000_000))  # full TinyStories — 200M tokens
    else:
        print("[combine] ⚠  TinyStories train.bin not found — run data/tinystories/prepare.py")
    if os.path.exists(tiny_val):
        all_val_stories.extend(_load_stories_from_bin(tiny_val, "TinyStories val"))

    # ── Optional: WritingPrompts ──────────────────────────────────────────────
    if with_writing_prompts:
        wp_stories = _load_writing_prompts()
        # 90/10 split for WritingPrompts
        n_wp = len(wp_stories)
        n_wp_val = max(1, int(n_wp * 0.1))
        wp_idx = rng.permutation(n_wp)
        all_val_stories.extend([wp_stories[i] for i in wp_idx[:n_wp_val]])
        all_train_stories.extend([wp_stories[i] for i in wp_idx[n_wp_val:]])

    # ── Optional: Children's stories ─────────────────────────────────────────
    if with_childrens:
        ch_stories = _load_childrens_stories()
        n_ch = len(ch_stories)
        n_ch_val = max(1, int(n_ch * 0.1))
        ch_idx = rng.permutation(n_ch)
        all_val_stories.extend([ch_stories[i] for i in ch_idx[:n_ch_val]])
        all_train_stories.extend([ch_stories[i] for i in ch_idx[n_ch_val:]])

    # ── Summary ───────────────────────────────────────────────────────────────
    total_train_tok = sum(len(s) for s in all_train_stories)
    total_val_tok   = sum(len(s) for s in all_val_stories)

    print(f"\n[combine] Corpus summary:")
    print(f"  Train: {len(all_train_stories):,} stories  |  {total_train_tok/1e6:.1f}M tokens")
    print(f"  Val  : {len(all_val_stories):,} stories  |  {total_val_tok/1e6:.1f}M tokens")

    if dry_run:
        print("\n[combine] --dry_run: no files written.")
        return

    # ── Shuffle at story level ────────────────────────────────────────────────
    print("[combine] Shuffling train stories …")
    train_idx = rng.permutation(len(all_train_stories))
    all_train_stories = [all_train_stories[i] for i in train_idx]

    # ── Write ─────────────────────────────────────────────────────────────────
    os.makedirs(script_dir, exist_ok=True)

    for split, stories in [('train', all_train_stories), ('val', all_val_stories)]:
        out_path = os.path.join(script_dir, f'{split}.bin')
        combined = np.concatenate(stories)
        combined.tofile(out_path)
        print(f"[combine] Wrote {split}.bin: {len(combined)/1e6:.1f}M tokens → {out_path}")

    print("\n[combine] Done! Combined dataset ready.")
    print("  Next: python train.py config/train_t4_arena.py")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build Task 4 arena training corpus')
    parser.add_argument('--with_writing_prompts', action='store_true',
                        help='Include WritingPrompts (~180M tokens)')
    parser.add_argument('--with_childrens', action='store_true',
                        help="Include Children's stories (~25M tokens)")
    parser.add_argument('--dry_run', action='store_true',
                        help='Count tokens and print summary without writing files')
    args = parser.parse_args()
    build_combined(
        with_writing_prompts=args.with_writing_prompts,
        with_childrens=args.with_childrens,
        dry_run=args.dry_run,
    )
