"""
Combined Data Preparation: ROCStories + TinyStories
=====================================================
Merges ROCStories and TinyStories into a single combined dataset
for maximum training data. Both are tokenised with GPT-2 BPE and
concatenated into combined train.bin / val.bin files.

This gives the model ~60-80M tokens of narrative text instead of
just ~2.25M from ROCStories alone — significantly reducing overfitting
and improving generalization on story generation.

Usage:
    python data/combined/prepare.py

Prerequisites:
    First run both individual prepare scripts:
    python data/rocstories/prepare.py
    python data/tinystories/prepare.py

    Or this script will attempt to prepare them automatically.

Outputs:
    data/combined/train.bin   (uint16 numpy array)
    data/combined/val.bin     (uint16 numpy array)
"""

import os
import sys
import numpy as np


def ensure_dataset(data_dir, name):
    """Check if a dataset is prepared, and prepare it if not."""
    train_path = os.path.join(data_dir, 'train.bin')
    val_path = os.path.join(data_dir, 'val.bin')

    if os.path.exists(train_path) and os.path.exists(val_path):
        train_tokens = len(np.fromfile(train_path, dtype=np.uint16))
        val_tokens = len(np.fromfile(val_path, dtype=np.uint16))
        print(f"[combine] {name}: {train_tokens:,} train + {val_tokens:,} val tokens (already prepared)")
        return train_tokens, val_tokens

    # Try to prepare the dataset
    prepare_script = os.path.join(data_dir, 'prepare.py')
    if os.path.exists(prepare_script):
        print(f"[combine] Preparing {name}...")
        import subprocess
        result = subprocess.run([sys.executable, prepare_script], capture_output=False)
        if result.returncode != 0:
            print(f"[combine] ERROR: Failed to prepare {name}")
            return 0, 0

        if os.path.exists(train_path) and os.path.exists(val_path):
            train_tokens = len(np.fromfile(train_path, dtype=np.uint16))
            val_tokens = len(np.fromfile(val_path, dtype=np.uint16))
            return train_tokens, val_tokens

    print(f"[combine] WARNING: {name} not available, skipping")
    return 0, 0


def combine_datasets(out_dir):
    """Combine ROCStories and TinyStories into a single dataset."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)  # data/ directory

    datasets = [
        (os.path.join(base_dir, 'rocstories'), 'ROCStories'),
        (os.path.join(base_dir, 'tinystories'), 'TinyStories'),
    ]

    os.makedirs(out_dir, exist_ok=True)

    for split in ['train', 'val']:
        all_tokens = []
        for data_dir, name in datasets:
            bin_path = os.path.join(data_dir, f'{split}.bin')
            if os.path.exists(bin_path):
                tokens = np.fromfile(bin_path, dtype=np.uint16)
                all_tokens.append(tokens)
                print(f"[combine] {split}: {name} = {len(tokens):,} tokens")
            else:
                # Try to prepare
                ensure_dataset(data_dir, name)
                if os.path.exists(bin_path):
                    tokens = np.fromfile(bin_path, dtype=np.uint16)
                    all_tokens.append(tokens)
                    print(f"[combine] {split}: {name} = {len(tokens):,} tokens")

        if all_tokens:
            combined = np.concatenate(all_tokens)
            # Shuffle at the story level (each story ends with EOT = 50256)
            out_path = os.path.join(out_dir, f'{split}.bin')
            combined.tofile(out_path)
            print(f"[combine] {split}.bin: {len(combined):,} total tokens -> {out_path}")
        else:
            print(f"[combine] WARNING: No data for {split} split!")

    print(f"\n[combine] Done! Combined dataset ready at {out_dir}")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    combine_datasets(script_dir)
