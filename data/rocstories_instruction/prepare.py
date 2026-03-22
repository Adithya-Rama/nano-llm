"""
ROCStories Instruction Format for T4 Arena Fine-tuning.

Dataset has only a 'text' column (confirmed: mintujupally/ROCStories).
Format: "Continue this story: {first_sentence}\n{full_story}"

This matches arena judging exactly — prof gives opening sentence,
model continues. Stage 2 fine-tune teaches this exact pattern.

Prereq: data/rocstories/train.bin must exist.

Usage:
    python data/rocstories_instruction/prepare.py
"""
import os
import sys
import numpy as np
import tiktoken

SEED       = 42
HF_DATASET = "mintujupally/ROCStories"
VAL_FRAC   = 0.05   # 5% of stories held out for val monitoring


def load_and_format():
    from datasets import load_dataset
    enc = tiktoken.get_encoding("gpt2")
    eot = 50256

    try:
        ds = load_dataset(HF_DATASET, split="train")
    except Exception as e:
        print(f"[rocstories_instruction] Error loading dataset: {e}")
        sys.exit(1)

    print(f"[rocstories_instruction] Dataset columns: {ds.column_names}")

    stories = []
    skipped = 0
    for row in ds:
        text = (row.get("text", "") or "").strip()
        if not text or len(text.split()) < 10:
            skipped += 1
            continue

        # Extract first sentence as the continuation prompt
        # Split on '. ' to get sentences, take first one
        dot_idx = text.find(". ")
        if dot_idx > 0 and dot_idx < len(text) - 2:
            first_sentence = text[: dot_idx + 1].strip()
        else:
            # fallback: use first 10 words
            words = text.split()
            first_sentence = " ".join(words[:10]) + "."

        # Instruction format: matches arena prompt style exactly
        formatted = f"Continue this story: {first_sentence}\n{text}"
        stories.append(formatted)

    print(f"[rocstories_instruction] Loaded {len(stories):,} stories "
          f"(skipped {skipped} short/empty)")

    # Shuffle deterministically
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(stories))
    stories = [stories[i] for i in idx]

    # Split train / val
    n_val   = max(100, int(len(stories) * VAL_FRAC))
    n_train = len(stories) - n_val
    train_stories = stories[:n_train]
    val_stories   = stories[n_train:]

    print(f"[rocstories_instruction] Split: {n_train:,} train | {n_val:,} val")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    for split, split_stories in [("train", train_stories), ("val", val_stories)]:
        all_tokens = []
        for story in split_stories:
            all_tokens.extend(enc.encode_ordinary(story))
            all_tokens.append(eot)
        arr = np.array(all_tokens, dtype=np.uint16)
        out_path = os.path.join(out_dir, f"{split}.bin")
        arr.tofile(out_path)
        print(f"[rocstories_instruction] {split}.bin: {len(arr)/1e6:.2f}M tokens → {out_path}")

    print(f"[rocstories_instruction] Done.")


if __name__ == "__main__":
    load_and_format()
