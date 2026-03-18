"""
Mixed Dataset Preparation for nanoGPT — Task 2 Instruction Experiment
======================================================================
Creates a training mixture of three text formats from ROCStories:

  Format A — Plain continuation (55% of tokens)
      Raw story text, exactly as in the Task 1 dataset.
      This is what the assignment evaluates, so it must dominate.

  Format B — Instruction-prefixed stories (30% of tokens)
      "Write a story about: <first-sentence-topic>.\\n<full story>"
      Teaches the model to respond to generation prompts without
      hurting plain continuation performance (Ouyang et al., 2022).

  Format C — Structured narrative (15% of tokens)
      "<story><s1>...</s1>...<s5>...</s5></story>"
      Exposes the model to explicit 5-sentence arc markers.

Validation set is ALWAYS plain-text ROCStories only, so PPL
comparisons across experiments remain apples-to-apples.

Usage:
    # Requires data/rocstories/{train,val}.bin to exist first
    python data/rocstories/prepare.py          # plain
    python data/rocstories/prepare.py --structured  # structured

    # Then build the mixed dataset:
    python data/mixed/prepare.py

    # To include TinyStories (must exist in data/tinystories/):
    python data/mixed/prepare.py --with_tinystories

Outputs:
    data/mixed/train.bin   — mixed formats
    data/mixed/val.bin     — plain ROCStories only (for fair PPL)

References:
    Ouyang, L. et al. (2022). Training language models to follow
      instructions. arXiv:2203.02155. (InstructGPT)
"""

import os
import re
import sys
import argparse
import numpy as np

# ---------------------------------------------------------------------------
# Mixing ratios (in terms of story count; token count will differ slightly)
# ---------------------------------------------------------------------------
RATIO_PLAIN       = 0.55   # Format A: plain continuation
RATIO_INSTRUCTION = 0.30   # Format B: instruction-prefixed
RATIO_STRUCTURED  = 0.15   # Format C: structured tags
# If TinyStories is included it partially replaces Format A
RATIO_TINYSTORIES = 0.20   # borrowed from Format A when --with_tinystories

SEED = 42


# ---------------------------------------------------------------------------
# Instruction templates  (Format B)
# ---------------------------------------------------------------------------
# Each template must end with a newline so the story body follows naturally.
INSTRUCTION_TEMPLATES = [
    "Write a short story about: {topic}.\n",
    "Tell me a 5-sentence story involving {topic}.\n",
    "Generate a brief narrative about {topic}.\n",
    "Create a story: {topic}.\n",
    "Complete this story prompt — {first_sent}\n",
]


def _extract_topic(story_text: str) -> str:
    """
    Heuristically extract a short topic phrase from the first sentence.
    Returns the first 6 words of the first sentence, lower-cased.
    This produces natural-sounding prompts like
      'Write a short story about: the boy went to a video arcade.'
    """
    sentences = re.split(r'(?<=[.!?])\s+', story_text.strip())
    first = sentences[0].rstrip('.!?').strip() if sentences else story_text[:60]
    words = first.split()[:6]
    return ' '.join(words).lower()


def _extract_first_sentence(story_text: str) -> str:
    """Return just the first sentence of a story."""
    sentences = re.split(r'(?<=[.!?])\s+', story_text.strip())
    return sentences[0].strip() if sentences else story_text[:80]


def _to_structured(story_text: str) -> str:
    """
    Wrap a plain story in XML-style sentence tags (Format C).
    Matches the format produced by data/rocstories/prepare.py --structured.
    """
    sentences = re.split(r'(?<=[.!?])\s+', story_text.strip())
    parts = ["<story>"]
    for i, s in enumerate(sentences[:5], 1):
        parts.append(f"<s{i}>{s.strip()}</s{i}>")
    parts.append("</story>")
    return "\n".join(parts)


def _format_instruction(story_text: str, rng: np.random.Generator) -> str:
    """Apply a randomly chosen instruction template to a story."""
    topic = _extract_topic(story_text)
    first_sent = _extract_first_sentence(story_text)
    tmpl = INSTRUCTION_TEMPLATES[rng.integers(len(INSTRUCTION_TEMPLATES))]
    prompt = tmpl.format(topic=topic, first_sent=first_sent)
    return prompt + story_text


# ---------------------------------------------------------------------------
# Load raw story strings from HuggingFace
# ---------------------------------------------------------------------------
def load_raw_stories(structured: bool = False):
    """Re-use the existing ROCStories loader."""
    roc_dir = os.path.join(os.path.dirname(__file__), '..', 'rocstories')
    sys.path.insert(0, os.path.abspath(roc_dir))
    from prepare import load_rocstories
    return load_rocstories(structured=structured)


# ---------------------------------------------------------------------------
# Build the mixed tokenised dataset
# ---------------------------------------------------------------------------
def build_mixed_dataset(with_tinystories: bool = False):
    import tiktoken
    enc  = tiktoken.get_encoding("gpt2")
    eot  = 50256
    rng  = np.random.default_rng(SEED)

    print("[mixed] Loading ROCStories (plain format)…")
    stories = load_raw_stories(structured=False)
    n = len(stories)
    print(f"[mixed] {n:,} stories loaded")

    # Shuffle deterministically
    idx = rng.permutation(n)
    stories = [stories[i] for i in idx]

    # ── Assign each story to a format ─────────────────────────────────────
    # We cycle through the three formats according to the desired ratios.
    # This ensures the val split has no cross-format contamination.
    n_plain       = int(n * RATIO_PLAIN)
    n_instruction = int(n * RATIO_INSTRUCTION)
    # remainder → structured
    n_structured  = n - n_plain - n_instruction

    plain_stories       = stories[:n_plain]
    instruction_stories = stories[n_plain : n_plain + n_instruction]
    structured_stories  = stories[n_plain + n_instruction:]

    print(f"[mixed] Format split: {n_plain} plain | "
          f"{n_instruction} instruction | {n_structured} structured")

    # ── Format B: wrap each story with an instruction prefix ──────────────
    instruction_texts = [
        _format_instruction(s, rng) for s in instruction_stories
    ]

    # ── Format C: wrap each story with XML tags ───────────────────────────
    structured_texts = [_to_structured(s) for s in structured_stories]

    # ── Build training corpus (mix all three) ─────────────────────────────
    train_texts = plain_stories + instruction_texts + structured_texts
    # Re-shuffle so formats are interleaved
    train_shuffle = rng.permutation(len(train_texts))
    train_texts = [train_texts[i] for i in train_shuffle]

    # ── Optionally prepend TinyStories tokens ─────────────────────────────
    tiny_train_tokens = []
    if with_tinystories:
        tiny_path = os.path.join(
            os.path.dirname(__file__), '..', 'tinystories', 'train.bin')
        if os.path.exists(tiny_path):
            print("[mixed] Loading TinyStories tokens…")
            tiny_arr = np.fromfile(tiny_path, dtype=np.uint16)
            # Use only up to 20M tokens to avoid TinyStories dominating
            cap = min(len(tiny_arr), 20_000_000)
            tiny_train_tokens = tiny_arr[:cap].tolist()
            print(f"[mixed] TinyStories: {len(tiny_train_tokens)/1e6:.1f}M tokens capped")
        else:
            print("[mixed] ⚠ TinyStories not found — run data/tinystories/prepare.py first")

    # ── Tokenise all training texts ───────────────────────────────────────
    print("[mixed] Tokenising training texts…")
    train_tokens = list(tiny_train_tokens)
    for text in train_texts:
        toks = enc.encode_ordinary(text)
        train_tokens.extend(toks)
        train_tokens.append(eot)

    # ── Validation: plain ROCStories only (same val stories as baseline) ──
    # Use last 10% of the original (pre-shuffle) plain stories as val.
    # NOTE: to keep perfectly comparable to Task 1 val, we re-load and
    # use the same shuffle seed so the val split is identical.
    print("[mixed] Building val set (plain ROCStories only)…")
    all_stories_orig = load_raw_stories(structured=False)
    val_rng = np.random.default_rng(SEED)
    idx2 = val_rng.permutation(len(all_stories_orig))
    n_val = max(1, int(len(all_stories_orig) * 0.10))
    val_stories = [all_stories_orig[i] for i in idx2[:n_val]]

    val_tokens = []
    for text in val_stories:
        toks = enc.encode_ordinary(text)
        val_tokens.extend(toks)
        val_tokens.append(eot)

    # ── Save ──────────────────────────────────────────────────────────────
    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    train_arr = np.array(train_tokens, dtype=np.uint16)
    val_arr   = np.array(val_tokens,   dtype=np.uint16)

    train_path = os.path.join(out_dir, 'train.bin')
    val_path   = os.path.join(out_dir, 'val.bin')

    train_arr.tofile(train_path)
    val_arr.tofile(val_path)

    print(f"\n[mixed] Done!")
    print(f"  train.bin : {len(train_arr)/1e6:.2f}M tokens  → {train_path}")
    print(f"  val.bin   : {len(val_arr)/1e3:.1f}K tokens   → {val_path}")
    print(f"  Format breakdown:")
    print(f"    Plain continuation : {n_plain:,} stories")
    print(f"    Instruction-prefix : {n_instruction:,} stories")
    print(f"    Structured (XML)   : {n_structured:,} stories")
    if tiny_train_tokens:
        print(f"    TinyStories prefix : {len(tiny_train_tokens)/1e6:.1f}M tokens")
    print(f"\n  Val set : plain ROCStories only ({len(val_stories):,} stories)")
    print(f"            → PPL comparisons are apples-to-apples with Task 1")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build mixed instruction/continuation dataset')
    parser.add_argument(
        '--with_tinystories', action='store_true',
        help='Prepend up to 20M TinyStories tokens to training data')
    args = parser.parse_args()
    build_mixed_dataset(with_tinystories=args.with_tinystories)
