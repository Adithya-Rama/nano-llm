"""
colab_setup.py — Google Colab End-to-End Training Guide
=========================================================
Copy each CELL section into a separate Colab cell and run sequentially.

Model: ~152M parameter LLaMA-style transformer with QK-Norm
       (12 layers, 12 heads, 768-dim)

Workflow:
  CELL 1  → Install dependencies
  CELL 2  → Mount Drive + clone repo
  CELL 3  → Verify GPU + model fit
  CELL 4  → Prepare datasets (ROCStories + TinyStories)
  CELL 5  → Train model (Task 1 — safe to re-run, auto-resumes)
  CELL 6  → Evaluate (perplexity + qualitative samples)
  CELL 7  → Run ablation experiments (Task 2 — 5 configs)
  CELL 8  → Package & upload to HuggingFace (Task 3 submission)
  CELL 9  → Keep-alive (run in parallel to prevent idle disconnect)
"""

# ══════════════════════════════════════════════════════════════════════════════
# CELL 1 — Install dependencies
# ══════════════════════════════════════════════════════════════════════════════
CELL_1 = """
!pip install -q tiktoken datasets huggingface_hub

import torch
print(f"PyTorch  : {torch.__version__}")
print(f"CUDA     : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU      : {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM     : {vram:.1f} GB")
    # Enable TF32 for faster matmul on Ampere GPUs (no quality loss)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
print("✓ Dependencies ready")
"""

# ══════════════════════════════════════════════════════════════════════════════
# CELL 2 — Mount Drive and clone repo
# ══════════════════════════════════════════════════════════════════════════════
CELL_2 = """
from google.colab import drive
import os

drive.mount('/content/drive', force_remount=True)

# ── Clone your repo from GitHub ──────────────────────────────────────────────
# Replace <YOUR_GITHUB_USERNAME> and <YOUR_REPO_NAME> with your actual values.
REPO_URL = "https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>.git"
LOCAL_DIR = "/content/nanogpt_project"

if not os.path.exists(LOCAL_DIR):
    !git clone {REPO_URL} {LOCAL_DIR}
else:
    !cd {LOCAL_DIR} && git pull   # keep up to date

os.chdir(f"{LOCAL_DIR}/nanoGPT_code/nanoGPT")
print(f"✓ Working directory: {os.getcwd()}")

# ── Create checkpoint directory on Drive (persists across sessions) ──────────
DRIVE_CKPT = "/content/drive/MyDrive/nanogpt_rocstories_checkpoints"
os.makedirs(DRIVE_CKPT, exist_ok=True)
print(f"✓ Checkpoint dir (Drive): {DRIVE_CKPT}")
"""

# ══════════════════════════════════════════════════════════════════════════════
# CELL 3 — GPU memory sanity check (152M model)
# ══════════════════════════════════════════════════════════════════════════════
CELL_3 = """
import torch, sys
sys.path.insert(0, "/content/nanogpt_project/nanoGPT_code/nanoGPT")

from model import GPT, GPTConfig

# Build the ~152M parameter model and estimate VRAM
cfg = GPTConfig(
    n_layer=12, n_head=12, n_embd=768, block_size=256, vocab_size=50304,
    bias=False, dropout=0.1,
    use_rmsnorm=True, use_rope=True, use_swiglu=True, use_qk_norm=True
)
model = GPT(cfg)
n_params = sum(p.numel() for p in model.parameters()) / 1e6

# Rough VRAM estimate (bfloat16 weights + fp32 Adam states + activations)
batch_size, seq_len = 16, 256
bytes_model   = n_params * 1e6 * 2          # bfloat16
bytes_adam    = n_params * 1e6 * 8          # 2× fp32 Adam states
bytes_act     = batch_size * seq_len * cfg.n_embd * cfg.n_layer * 2  # rough
total_gb = (bytes_model + bytes_adam + bytes_act) / 1e9

print(f"Model parameters : {n_params:.1f}M")
print(f"Estimated VRAM   : {total_gb:.1f} GB  (rough upper bound)")

if torch.cuda.is_available():
    free_gb = torch.cuda.mem_get_info()[0] / 1e9
    avail_gb = torch.cuda.mem_get_info()[1] / 1e9
    print(f"Available VRAM   : {free_gb:.1f} GB / {avail_gb:.1f} GB total")
    if total_gb < free_gb * 0.85:
        print("✓ Model fits with headroom")
    else:
        print("⚠  Tight fit! Consider reducing batch_size to 8.")

del model
torch.cuda.empty_cache()
"""

# ══════════════════════════════════════════════════════════════════════════════
# CELL 4 — Download and prepare datasets
#
# ROCStories (plain):      Required for Task 1 + Task 3
# ROCStories (structured): Task 2 experiment (sentence markers)
# TinyStories:             Task 2 experiment (more data)
# Combined:                Task 2 experiment (ROC + Tiny merged)
# ══════════════════════════════════════════════════════════════════════════════
CELL_4 = """
import os, sys
os.chdir("/content/nanogpt_project/nanoGPT_code/nanoGPT")

import numpy as np

# 1. Prepare ROCStories — plain text (required for Task 1 + Task 3)
print("=" * 60)
print("Preparing ROCStories (plain format)...")
print("=" * 60)
!python data/rocstories/prepare.py

# 2. Prepare ROCStories — structured format (Task 2 experiment)
print()
print("=" * 60)
print("Preparing ROCStories (structured format with sentence tags)...")
print("=" * 60)
!python data/rocstories/prepare.py --structured

# 3. Prepare TinyStories (Task 2 experiment — extra data)
print()
print("=" * 60)
print("Preparing TinyStories (this may take a few minutes)...")
print("=" * 60)
!python data/tinystories/prepare.py

# 4. Create combined dataset (Task 2 experiment)
print()
print("=" * 60)
print("Creating combined dataset...")
print("=" * 60)
!python data/combined/prepare.py

# Verify all outputs
print()
print("=" * 60)
print("Dataset Summary")
print("=" * 60)
for ds_name in ['rocstories', 'rocstories_structured', 'tinystories', 'combined']:
    for split in ['train', 'val']:
        fpath = f'data/{ds_name}/{split}.bin'
        if os.path.exists(fpath):
            arr = np.fromfile(fpath, dtype=np.uint16)
            print(f"  {ds_name}/{split}.bin: {len(arr):,} tokens")

print("\\n✓ All datasets ready")
"""

# ══════════════════════════════════════════════════════════════════════════════
# CELL 5 — Train (Task 1 + Task 3)
#
# SAFE TO RE-RUN after any disconnect — init_from=resume restores
# everything including optimizer state, scaler, and training position.
#
# First run: uses init_from='scratch' (from config file)
# After crash: change to --init_from=resume below
#
# Choose one of these configs:
#   config/train_rocstories.py          — ROCStories only (Task 1)
#   config/train_rocstories_combined.py — Combined data (best Task 3)
# ══════════════════════════════════════════════════════════════════════════════
CELL_5 = """
import os
os.chdir("/content/nanogpt_project/nanoGPT_code/nanoGPT")

# ── OPTION A: Train on ROCStories only (Task 1) ─────────────────────────────
# !python train.py config/train_rocstories.py

# ── OPTION B: Train on combined ROCStories+TinyStories (best for Task 3) ────
!python train.py config/train_rocstories_combined.py

# ── To resume after crash: ───────────────────────────────────────────────────
# !python train.py config/train_rocstories_combined.py --init_from=resume

print("✓ Training complete (or interrupted — re-run to resume)")
"""

# ══════════════════════════════════════════════════════════════════════════════
# CELL 6 — Evaluate and generate samples
# ══════════════════════════════════════════════════════════════════════════════
CELL_6 = """
import os
os.chdir("/content/nanogpt_project/nanoGPT_code/nanoGPT")

# Change out_dir to match whichever config you trained with
OUT_DIR = "out-rocstories-combined"  # or "out-rocstories"

print("=" * 60)
print("PERPLEXITY EVALUATION on eval_stories.txt")
print("=" * 60)
!python eval.py --init_from=resume --out_dir={OUT_DIR} \\
    --input_file=data/rocstories/eval_stories.txt

print()
print("=" * 60)
print("STORY GENERATION from eval_prompts.txt")
print("=" * 60)
!python sample_batch.py \\
    --init_from=resume \\
    --out_dir={OUT_DIR} \\
    --start="FILE:data/rocstories/eval_prompts.txt" \\
    --batch_prompts=True \\
    --max_new_tokens=200 \\
    --output_file={OUT_DIR}/generated_stories.jsonl
"""

# ══════════════════════════════════════════════════════════════════════════════
# CELL 7 — Task 2: 6-way Ablation experiments
#
# Run each config separately and compare val loss / generated quality.
# All results are saved in separate out-rocstories-* directories.
#
# Experiments:
#   A. Baseline (vanilla GPT)
#   B. +RoPE only
#   C. +RMSNorm + SwiGLU
#   D. +QK-Norm only (novel)
#   E. All modern (full model)
#   F. All modern + Structured format (data experiment)
#
# ⚠ This takes ~4-5 hours total on A100 (6 × ~40 min each)
# ══════════════════════════════════════════════════════════════════════════════
CELL_7 = """
import os, subprocess, json
os.chdir("/content/nanogpt_project/nanoGPT_code/nanoGPT")

experiments = [
    # (config_file, out_dir, description)
    ("config/train_rocstories_baseline.py",       "out-rocstories-baseline",    "A. Vanilla GPT"),
    ("config/train_rocstories_rope_only.py",      "out-rocstories-rope",        "B. +RoPE only"),
    ("config/train_rocstories_rmsnorm_swiglu.py",  "out-rocstories-ffn",        "C. +RMSNorm+SwiGLU"),
    ("config/train_rocstories_qknorm.py",          "out-rocstories-qknorm",     "D. +QK-Norm only"),
    ("config/train_rocstories.py",                 "out-rocstories",            "E. All modern (full)"),
    ("config/train_rocstories_structured.py",      "out-rocstories-structured", "F. Structured format"),
]

results = {}
for config, out_dir, desc in experiments:
    print(f"\\n{'='*60}")
    print(f"Running: {desc}")
    print('='*60)
    # Train
    subprocess.run(["python", "train.py", config], check=True)
    # Evaluate PPL
    result = subprocess.run(
        ["python", "eval.py",
         f"--init_from=resume", f"--out_dir={out_dir}",
         "--input_file=data/rocstories/eval_stories.txt"],
        capture_output=True, text=True
    )
    print(result.stdout[-500:])
    for line in result.stdout.split('\\n'):
        if 'ppl' in line.lower():
            results[desc] = line.strip()
    # Generate stories for quality evaluation
    subprocess.run([
        "python", "sample_batch.py",
        f"--init_from=resume", f"--out_dir={out_dir}",
        "--start=FILE:data/rocstories/eval_prompts.txt",
        "--batch_prompts=True", "--max_new_tokens=200",
        f"--output_file={out_dir}/generated_stories.jsonl"
    ])

print("\\n" + "="*60)
print("ABLATION SUMMARY")
print("="*60)
for desc, val in results.items():
    print(f"  {desc:40s}: {val}")
"""

# ══════════════════════════════════════════════════════════════════════════════
# CELL 7b — Plot learning curves + Story quality metrics
#
# Run AFTER CELL 7 to generate report-ready charts and tables.
# ══════════════════════════════════════════════════════════════════════════════
CELL_7b = """
import os
os.chdir("/content/nanogpt_project/nanoGPT_code/nanoGPT")

!pip install -q matplotlib

# ── 1. Plot ablation comparison (val loss) ───────────────────────────────────
print("=" * 60)
print("Generating ablation learning curve plot...")
print("=" * 60)
!python plot_training.py \\
    --log out-rocstories-baseline/train_log.jsonl \\
    --log out-rocstories-rope/train_log.jsonl \\
    --log out-rocstories-ffn/train_log.jsonl \\
    --log out-rocstories-qknorm/train_log.jsonl \\
    --log out-rocstories/train_log.jsonl \\
    --labels "Baseline,+RoPE,+RMSNorm+SwiGLU,+QK-Norm,All Modern" \\
    --output ablation_curves.png \\
    --title "Task 2: Ablation — Validation Loss"

# ── 2. Plot single best model training curve ─────────────────────────────────
!python plot_training.py \\
    --log out-rocstories/train_log.jsonl \\
    --output training_curve.png \\
    --title "Task 1: Training Curves (152M All Modern)"

# ── 3. Story quality comparison ──────────────────────────────────────────────
print()
print("=" * 60)
print("Computing story quality metrics...")
print("=" * 60)
!python eval_story_quality.py \\
    --input out-rocstories-baseline/generated_stories.jsonl \\
    --input out-rocstories-rope/generated_stories.jsonl \\
    --input out-rocstories-ffn/generated_stories.jsonl \\
    --input out-rocstories-qknorm/generated_stories.jsonl \\
    --input out-rocstories/generated_stories.jsonl \\
    --labels "Baseline,+RoPE,+RMSNorm+SwiGLU,+QK-Norm,All Modern"

print("\\n✓ Charts saved: ablation_curves.png, training_curve.png")
print("  Download these for your report!")
"""

# ══════════════════════════════════════════════════════════════════════════════
# CELL 8 — Package and upload to HuggingFace (Task 3 submission)
#
# Prerequisites:
#   1. Set HF_USERNAME to your HuggingFace username
#   2. Set HF_TOKEN to your HuggingFace write token
# ══════════════════════════════════════════════════════════════════════════════
CELL_8 = """
import os, shutil, json
os.chdir("/content/nanogpt_project/nanoGPT_code/nanoGPT")

HF_USERNAME = "YOUR_HF_USERNAME"    # ← REPLACE THIS
HF_TOKEN    = "YOUR_HF_TOKEN"       # ← REPLACE THIS

# ── Choose your best model ───────────────────────────────────────────────────
# Change this to whichever out_dir produced the best results
BEST_OUT_DIR = "out-rocstories-combined"  # or "out-rocstories"

# ── Copy ckpt.pt + model.py + sample_params.json into submission folder ──────
SUBMIT_DIR = "submission_hf"
os.makedirs(SUBMIT_DIR, exist_ok=True)

shutil.copy(f"{BEST_OUT_DIR}/ckpt.pt",        f"{SUBMIT_DIR}/ckpt.pt")
shutil.copy("model.py",                       f"{SUBMIT_DIR}/model.py")
shutil.copy("sample_params.json",             f"{SUBMIT_DIR}/sample_params.json")
shutil.copy("eval.py",                        f"{SUBMIT_DIR}/eval.py")
shutil.copy("configurator.py",                f"{SUBMIT_DIR}/configurator.py")

print("Submission folder contents:")
for f in sorted(os.listdir(SUBMIT_DIR)):
    size_mb = os.path.getsize(f"{SUBMIT_DIR}/{f}") / 1e6
    print(f"  {f:<30s} {size_mb:.1f} MB")

# ── Upload to HuggingFace ─────────────────────────────────────────────────────
REPO_ID = f"{HF_USERNAME}/nanoGPT_hw"

!python hf_load.py upload \\
    --local-dir {SUBMIT_DIR} \\
    --repo-id {REPO_ID} \\
    --token {HF_TOKEN} \\
    --commit-message "Final ROCStories nanoGPT submission (152M params)"

print(f"✓ Model uploaded to: https://huggingface.co/{REPO_ID}")
print(f"  Submit to Canvas: repo = {REPO_ID}  token = {HF_TOKEN}")
"""

# ══════════════════════════════════════════════════════════════════════════════
# CELL 9 — Anti-idle keep-alive (run in a SEPARATE cell during training)
# ══════════════════════════════════════════════════════════════════════════════
CELL_9 = """
import time, threading
from IPython.display import display, Javascript

def keep_alive():
    while True:
        time.sleep(55)
        display(Javascript('console.log("keep-alive")'))
        print(f"  [keep-alive] {time.strftime('%H:%M:%S')}", end="\\r")

t = threading.Thread(target=keep_alive, daemon=True)
t.start()
print("✓ Keep-alive started (runs every 55 s)")
"""


# ── Print all cells when executed directly ───────────────────────────────────
if __name__ == "__main__":
    cells = [
        ("CELL 1 — Install dependencies",          CELL_1),
        ("CELL 2 — Mount Drive + clone repo",       CELL_2),
        ("CELL 3 — GPU memory check (152M model)",  CELL_3),
        ("CELL 4 — Prepare datasets",               CELL_4),
        ("CELL 5 — Train (Task 1 + Task 3)",        CELL_5),
        ("CELL 6 — Evaluate",                       CELL_6),
        ("CELL 7 — 5-way ablation (Task 2)",        CELL_7),
        ("CELL 8 — Upload to HuggingFace (Task 3)", CELL_8),
        ("CELL 9 — Keep-alive",                     CELL_9),
    ]
    for title, code in cells:
        print(f"\n{'='*70}")
        print(f"# {title}")
        print('='*70)
        print(code)
