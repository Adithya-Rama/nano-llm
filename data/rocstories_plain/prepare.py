"""
ROCStories Plain Format for T4 Arena Stage 2 Fine-tuning.

Splits rocstories train corpus 90/10 into train/val.
Val is ~410 stories (~33K tokens) — large enough for block_size=512.

Does NOT modify data/rocstories/ — that dataset is used for T1/T2/T3.

Prereq: data/rocstories/train.bin must exist.
  Run: python data/rocstories/prepare.py

Usage:
    python data/rocstories_plain/prepare.py
"""
import os
import numpy as np

SEED     = 42
VAL_FRAC = 0.10   # 10% val → ~410 stories, ~33K tokens > block_size=512

EOT = 50256


def _split_at_eot(arr: np.ndarray):
    """Split token array into individual stories at EOT boundaries."""
    eot_positions = np.where(arr == EOT)[0]
    stories = []
    start = 0
    for end in eot_positions:
        stories.append(arr[start: end + 1])   # include EOT
        start = end + 1
    return stories


def build():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_train  = os.path.normpath(
        os.path.join(script_dir, '..', 'rocstories', 'train.bin'))

    if not os.path.exists(src_train):
        print(f"[rocstories_plain] ERROR: {src_train} not found.")
        print("  Run: python data/rocstories/prepare.py  first.")
        raise SystemExit(1)

    print(f"[rocstories_plain] Loading {src_train} …")
    arr = np.fromfile(src_train, dtype=np.uint16)
    print(f"[rocstories_plain]   {len(arr)/1e6:.2f}M tokens loaded")

    stories = _split_at_eot(arr)
    print(f"[rocstories_plain]   {len(stories):,} stories found")

    # Deterministic shuffle then 90/10 split
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(stories))
    stories = [stories[i] for i in idx]

    n_val   = max(200, int(len(stories) * VAL_FRAC))
    n_train = len(stories) - n_val

    train_stories = stories[:n_train]
    val_stories   = stories[n_train:]

    train_tokens = sum(len(s) for s in train_stories)
    val_tokens   = sum(len(s) for s in val_stories)

    print(f"[rocstories_plain] Split: {n_train:,} train ({train_tokens/1e6:.2f}M tok) "
          f"| {n_val:,} val ({val_tokens/1e3:.0f}K tok)")

    # Sanity check: val must be larger than block_size=512
    assert val_tokens > 1024, \
        f"val.bin too small ({val_tokens} tokens) for block_size=512!"

    os.makedirs(script_dir, exist_ok=True)

    for split, split_stories in [('train', train_stories), ('val', val_stories)]:
        out_path = os.path.join(script_dir, f'{split}.bin')
        combined = np.concatenate(split_stories)
        combined.tofile(out_path)
        print(f"[rocstories_plain] Wrote {split}.bin: "
              f"{len(combined)/1e6:.2f}M tokens → {out_path}")

    print("[rocstories_plain] Done — ready for train_t4_finetune.py")


if __name__ == '__main__':
    build()
