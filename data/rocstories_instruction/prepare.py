"""
ROCStories Instruction Format for T4 Arena Fine-tuning.
Formats each story as: "Write a story about: {title}\n{story_body}"
Used for Stage 2 instruction fine-tuning of the T4 arena model.

Prereq: data/rocstories/val.bin must exist (run data/rocstories/prepare.py first).

Usage:
    python data/rocstories_instruction/prepare.py
"""
import os
import sys
import numpy as np
import tiktoken

SEED = 42
HF_DATASET = "mintujupally/ROCStories"


def load_and_format():
    from datasets import load_dataset
    enc = tiktoken.get_encoding("gpt2")
    eot = 50256

    try:
        ds = load_dataset(HF_DATASET, split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    stories = []
    for row in ds:
        raw   = (row.get('text', '') or '').strip()
        title = (row.get('storytitle', '') or row.get('title', '') or '').strip()

        if raw and title:
            text = f"Write a story about: {title}\n{raw}"
        elif title:
            # Tabular format (sentence1..sentence5 columns)
            sents = [row.get(f'sentence{i}', '').strip() for i in range(1, 6)]
            sents = [s for s in sents if s]
            if not sents:
                continue
            text = f"Write a story about: {title}\n{' '.join(sents)}"
        else:
            # No title → skip (instruction format needs a title prompt)
            continue

        if len(text.split()) >= 10:
            stories.append(text)

    print(f"[rocstories_instruction] Loaded {len(stories):,} instruction-format stories")

    # Shuffle deterministically
    rng = np.random.default_rng(SEED)
    stories = [stories[i] for i in rng.permutation(len(stories))]

    # Encode all stories
    all_tokens = []
    for story in stories:
        all_tokens.extend(enc.encode_ordinary(story))
        all_tokens.append(eot)

    arr = np.array(all_tokens, dtype=np.uint16)
    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    arr.tofile(os.path.join(out_dir, 'train.bin'))
    print(f"[rocstories_instruction] train.bin: {len(arr)/1e6:.2f}M tokens")

    # Val = reuse rocstories/val.bin (eval_stories.txt — plain format for PPL monitoring)
    import shutil
    roc_val = os.path.join(os.path.dirname(__file__), '..', 'rocstories', 'val.bin')
    roc_val = os.path.normpath(roc_val)
    if os.path.exists(roc_val):
        shutil.copy(roc_val, os.path.join(out_dir, 'val.bin'))
        print(f"[rocstories_instruction] val.bin: copied from rocstories/val.bin (eval_stories.txt)")
    else:
        print(f"[rocstories_instruction] WARNING: rocstories/val.bin not found — run data/rocstories/prepare.py first")

    print(f"[rocstories_instruction] Done → {out_dir}/")


if __name__ == '__main__':
    load_and_format()
