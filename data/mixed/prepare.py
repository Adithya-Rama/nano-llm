"""
Mixed Dataset Preparation for nanoGPT — Task 2 Instruction Experiment
======================================================================
Creates a training mixture of three text formats from ROCStories:

  Format A — Plain continuation (55% of tokens)
  Format B — Instruction-prefixed stories (30% of tokens)
  Format C — Structured narrative (15% of tokens)

Validation set is ALWAYS plain-text ROCStories only, so PPL
comparisons across experiments remain apples-to-apples.

FIX (v2): val stories are now reserved BEFORE the train/format split
so they never appear in any training format. The original code used
the same SEED for both the train shuffle and the val selection, making
the first 10% of stories appear in both val and plain-train (100%
overlap). This fix reserves val indices first, then assigns the
remaining 90% across formats A/B/C.
"""

import os
import re
import sys
import argparse
import numpy as np

RATIO_PLAIN       = 0.55
RATIO_INSTRUCTION = 0.30
RATIO_STRUCTURED  = 0.15
SEED = 42

INSTRUCTION_TEMPLATES = [
    "Write a short story about: {topic}.\n",
    "Tell me a 5-sentence story involving {topic}.\n",
    "Generate a brief narrative about {topic}.\n",
    "Create a story: {topic}.\n",
    "Complete this story prompt — {first_sent}\n",
]


def _extract_topic(story_text: str) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', story_text.strip())
    first = sentences[0].rstrip('.!?').strip() if sentences else story_text[:60]
    words = first.split()[:6]
    return ' '.join(words).lower()


def _extract_first_sentence(story_text: str) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', story_text.strip())
    return sentences[0].strip() if sentences else story_text[:80]


def _to_structured(story_text: str) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', story_text.strip())
    parts = ["<story>"]
    for i, s in enumerate(sentences[:5], 1):
        parts.append(f"<s{i}>{s.strip()}</s{i}>")
    parts.append("</story>")
    return "\n".join(parts)


def _format_instruction(story_text: str, rng: np.random.Generator) -> str:
    topic = _extract_topic(story_text)
    first_sent = _extract_first_sentence(story_text)
    tmpl = INSTRUCTION_TEMPLATES[rng.integers(len(INSTRUCTION_TEMPLATES))]
    prompt = tmpl.format(topic=topic, first_sent=first_sent)
    return prompt + story_text


def load_raw_stories(structured: bool = False):
    roc_dir = os.path.join(os.path.dirname(__file__), '..', 'rocstories')
    sys.path.insert(0, os.path.abspath(roc_dir))
    from prepare import load_rocstories
    return load_rocstories(structured=structured)


def build_mixed_dataset(with_tinystories: bool = False):
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    eot = 50256
    rng = np.random.default_rng(SEED)

    print("[mixed] Loading ROCStories (plain format)…")
    all_stories = load_raw_stories(structured=False)
    n = len(all_stories)
    print(f"[mixed] {n:,} stories loaded")

    # ── STEP 1: Reserve val stories BEFORE any format assignment ──────────
    # This is the critical fix. We carve out val_idx first so these stories
    # never appear in any training format (plain, instruction, or structured).
    idx = rng.permutation(n)
    n_val = max(1, int(n * 0.10))

    val_idx_arr   = idx[:n_val]      # first 10% → held-out validation
    train_idx_arr = idx[n_val:]      # remaining 90% → training only

    val_stories   = [all_stories[i] for i in val_idx_arr]
    train_stories = [all_stories[i] for i in train_idx_arr]
    n_train = len(train_stories)

    print(f"[mixed] Train/val split: {n_train:,} train | {n_val:,} val (0% overlap)")

    # ── STEP 2: Assign TRAIN stories to formats ───────────────────────────
    # Ratios applied to the 90% training pool only.
    n_plain       = int(n_train * RATIO_PLAIN)
    n_instruction = int(n_train * RATIO_INSTRUCTION)
    n_structured  = n_train - n_plain - n_instruction

    plain_stories       = train_stories[:n_plain]
    instruction_stories = train_stories[n_plain : n_plain + n_instruction]
    structured_stories  = train_stories[n_plain + n_instruction:]

    print(f"[mixed] Format split (train only): {n_plain} plain | "
          f"{n_instruction} instruction | {n_structured} structured")

    instruction_texts = [_format_instruction(s, rng) for s in instruction_stories]
    structured_texts  = [_to_structured(s) for s in structured_stories]

    # Mix and shuffle all three formats
    train_texts = plain_stories + instruction_texts + structured_texts
    train_shuffle = rng.permutation(len(train_texts))
    train_texts = [train_texts[i] for i in train_shuffle]

    # ── STEP 3: Optionally prepend TinyStories ────────────────────────────
    tiny_train_tokens = []
    if with_tinystories:
        tiny_path = os.path.join(
            os.path.dirname(__file__), '..', 'tinystories', 'train.bin')
        if os.path.exists(tiny_path):
            print("[mixed] Loading TinyStories tokens…")
            tiny_arr = np.fromfile(tiny_path, dtype=np.uint16)
            cap = min(len(tiny_arr), 100_000_000)  # use up to ~100M TinyStories tokens (~half full corpus)
            tiny_train_tokens = tiny_arr[:cap].tolist()
            print(f"[mixed] TinyStories: {len(tiny_train_tokens)/1e6:.1f}M tokens capped")
        else:
            print("[mixed] ⚠ TinyStories not found — skipping")

    # ── STEP 4: Tokenise training texts ───────────────────────────────────
    print("[mixed] Tokenising training texts…")
    train_tokens = list(tiny_train_tokens)
    for text in train_texts:
        toks = enc.encode_ordinary(text)
        train_tokens.extend(toks)
        train_tokens.append(eot)

    # ── STEP 5: Tokenise val (plain only, zero train overlap) ─────────────
    print("[mixed] Tokenising val set (plain ROCStories, no train overlap)…")
    val_tokens = []
    for text in val_stories:
        toks = enc.encode_ordinary(text)
        val_tokens.extend(toks)
        val_tokens.append(eot)

    # ── STEP 6: Save ──────────────────────────────────────────────────────
    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    train_arr = np.array(train_tokens, dtype=np.uint16)
    val_arr   = np.array(val_tokens,   dtype=np.uint16)

    train_arr.tofile(os.path.join(out_dir, 'train.bin'))
    val_arr.tofile(os.path.join(out_dir, 'val.bin'))

    print(f"\n[mixed] Done!")
    print(f"  train.bin : {len(train_arr)/1e6:.2f}M tokens")
    print(f"  val.bin   : {len(val_arr)/1e3:.1f}K tokens")
    print(f"  Val/train overlap: 0 stories (fixed)")
    print(f"\n  Format breakdown (train only):")
    print(f"    Plain continuation : {n_plain:,} stories")
    print(f"    Instruction-prefix : {n_instruction:,} stories")
    print(f"    Structured (XML)   : {n_structured:,} stories")
    if tiny_train_tokens:
        print(f"    TinyStories prefix : {len(tiny_train_tokens)/1e6:.1f}M tokens")
    print(f"\n  Val set : {len(val_stories):,} plain stories "
          f"(reserved before format split — clean PPL comparison)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--with_tinystories', action='store_true')
    args = parser.parse_args()
    build_mixed_dataset(with_tinystories=args.with_tinystories)
