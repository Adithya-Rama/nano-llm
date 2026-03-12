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
  CELL 5a → Check for existing checkpoints (auto-detects resume vs scratch)
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
# CELL 5a — Checkpoint detection (run BEFORE CELL 5)
#
# Checks Google Drive and the local out-dir for an existing ckpt.pt.
# Sets INIT_FROM automatically so CELL 5 never trains from scratch
# by accident after a Colab disconnect.
# ══════════════════════════════════════════════════════════════════════════════
CELL_5a = """
import os

# ── Config: match whichever config you want to train ────────────────────────
CONFIG_FILE  = "config/train_rocstories_combined.py"  # ← change if needed
OUT_DIR      = "out-rocstories-combined"              # ← must match config
DRIVE_CKPT   = "/content/drive/MyDrive/nanogpt_rocstories_checkpoints"

os.chdir("/content/nanogpt_project/nanoGPT_code/nanoGPT")

# ── Locations to check for an existing checkpoint ───────────────────────────
local_ckpt = os.path.join(OUT_DIR, "ckpt.pt")
drive_ckpt = os.path.join(DRIVE_CKPT, "ckpt.pt")

if os.path.exists(local_ckpt):
    INIT_FROM = "resume"
    print(f"✓ Checkpoint found locally:  {local_ckpt}")
elif os.path.exists(drive_ckpt):
    # Restore from Drive into the expected out-dir so train.py can find it
    os.makedirs(OUT_DIR, exist_ok=True)
    import shutil
    shutil.copy(drive_ckpt, local_ckpt)
    INIT_FROM = "resume"
    print(f"✓ Checkpoint restored from Drive → {local_ckpt}")
    # Also copy train_log.jsonl if present (for learning-curve continuity)
    drive_log = os.path.join(DRIVE_CKPT, "train_log.jsonl")
    if os.path.exists(drive_log):
        shutil.copy(drive_log, os.path.join(OUT_DIR, "train_log.jsonl"))
        print("  (train_log.jsonl restored too)")
else:
    INIT_FROM = "scratch"
    print("ℹ  No checkpoint found — will train from scratch.")

print(f"\n→ INIT_FROM = '{INIT_FROM}'")
print(f"→ CONFIG    = '{CONFIG_FILE}'")
print("\nCopy the variables above into CELL 5, or just run CELL 5 next —")
print("it reads INIT_FROM automatically if you keep both cells in the same session.")
"""

# ══════════════════════════════════════════════════════════════════════════════
# CELL 5 — Train (Task 1 + Task 3)
#
# SAFE TO RE-RUN after any disconnect.
# Run CELL 5a first — it sets INIT_FROM to 'resume' or 'scratch' automatically.
# If you skipped CELL 5a, the cell falls back to checking for ckpt.pt itself.
#
# Choose one of these configs by setting CONFIG_FILE / OUT_DIR below:
#   config/train_rocstories.py          → ROCStories only  (Task 1)
#   config/train_rocstories_combined.py → Combined data    (Task 3 best)
# ══════════════════════════════════════════════════════════════════════════════
CELL_5 = """
import os
os.chdir("/content/nanogpt_project/nanoGPT_code/nanoGPT")

# ── Choose config (keep in sync with CELL 5a) ───────────────────────────────
CONFIG_FILE = "config/train_rocstories_combined.py"  # ← change if needed
OUT_DIR     = "out-rocstories-combined"              # ← must match config

# ── Auto-detect whether to resume or start fresh ────────────────────────────
try:
    # INIT_FROM may already be set by CELL 5a in this session
    print(f"Using INIT_FROM='{INIT_FROM}' (set by CELL 5a)")
except NameError:
    # CELL 5a wasn't run (e.g. fresh session) — check ourselves
    ckpt_path = os.path.join(OUT_DIR, "ckpt.pt")
    INIT_FROM = "resume" if os.path.exists(ckpt_path) else "scratch"
    if INIT_FROM == "resume":
        print(f"✓ ckpt.pt found in {OUT_DIR} — resuming from checkpoint")
    else:
        print("ℹ  No checkpoint found — starting from scratch")

print(f"\n→ Running: python train.py {CONFIG_FILE} --init_from={INIT_FROM}\n")

!python train.py {CONFIG_FILE} --init_from={INIT_FROM}

# ── Backup checkpoint to Drive after training finishes ───────────────────────
DRIVE_CKPT = "/content/drive/MyDrive/nanogpt_rocstories_checkpoints"
if os.path.exists(os.path.join(OUT_DIR, "ckpt.pt")):
    import shutil
    os.makedirs(DRIVE_CKPT, exist_ok=True)
    try:
        shutil.copy(os.path.join(OUT_DIR, "ckpt.pt"), os.path.join(DRIVE_CKPT, "ckpt.pt"))
        log_src = os.path.join(OUT_DIR, "train_log.jsonl")
        if os.path.exists(log_src):
            shutil.copy(log_src, os.path.join(DRIVE_CKPT, "train_log.jsonl"))
        print(f"✓ Checkpoint backed up to Drive: {DRIVE_CKPT}")
    except shutil.SameFileError:
        print("✓ Checkpoint already on Drive (repo is on Drive) — no backup needed")

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
# Features:
#   • Detailed progress logging with timestamps and ETA
#   • Auto-resume: skips experiments that already have ckpt.pt
#   • Live output streaming (no silent subprocess)
#   • Saves ablation_results.json for CELL 7b
#
# Experiments:
#   A. Baseline (vanilla GPT)                    B. +RoPE only
#   C. +RMSNorm + SwiGLU                         D. +QK-Norm only (novel)
#   E. All modern (full model)                    F. Structured format
#
# ⚠ This takes ~4-5 hours total on A100 (6 × ~40 min each)
# ══════════════════════════════════════════════════════════════════════════════
CELL_7 = """
import os, subprocess, json, time
os.chdir(LOCAL_DIR)

experiments = [
    # (config_file, out_dir, description)
    ("config/train_rocstories_baseline.py",       "out-rocstories-baseline",    "A. Vanilla GPT"),
    ("config/train_rocstories_rope_only.py",      "out-rocstories-rope",        "B. +RoPE only"),
    ("config/train_rocstories_rmsnorm_swiglu.py",  "out-rocstories-ffn",        "C. +RMSNorm+SwiGLU"),
    ("config/train_rocstories_qknorm.py",          "out-rocstories-qknorm",     "D. +QK-Norm only"),
    ("config/train_rocstories.py",                 "out-rocstories",            "E. All modern (full)"),
    ("config/train_rocstories_structured.py",      "out-rocstories-structured", "F. Structured format"),
]

# ── Load any previously saved results (resume support) ───────────────────────
RESULTS_FILE = "ablation_results.json"
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE) as f:
        results = json.load(f)
    print(f"✓ Loaded {len(results)} previous results from {RESULTS_FILE}")
else:
    results = {}

total = len(experiments)
overall_start = time.time()
completed = 0
skipped = 0

for i, (config, out_dir, desc) in enumerate(experiments, 1):
    exp_start = time.time()
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    stories_path = os.path.join(out_dir, "generated_stories.jsonl")

    print(f"\\n{'━'*70}")
    print(f"  EXPERIMENT {i}/{total}: {desc}")
    print(f"  Config : {config}")
    print(f"  Out dir: {out_dir}")
    print(f"  Time   : {time.strftime('%H:%M:%S')}")
    print(f"{'━'*70}")

    # ── Phase 1: TRAIN ───────────────────────────────────────────────────────
    if os.path.exists(ckpt_path):
        print(f"  ⏩ SKIP training — checkpoint already exists: {ckpt_path}")
        skipped += 1
    else:
        print(f"  🚀 [1/3] Training {desc}...")
        train_result = subprocess.run(
            ["python", "train.py", config],
            check=False
        )
        if train_result.returncode != 0:
            print(f"  ❌ Training FAILED for {desc} (exit code {train_result.returncode})")
            print(f"     Skipping evaluation — continuing to next experiment.")
            continue
        elapsed = time.time() - exp_start
        print(f"  ✓ Training done ({elapsed/60:.1f} min)")

    # ── Phase 2: EVALUATE PERPLEXITY ─────────────────────────────────────────
    print(f"  📊 [2/3] Evaluating perplexity...")
    eval_result = subprocess.run(
        ["python", "eval.py",
         "--init_from=resume", f"--out_dir={out_dir}",
         "--input_file=data/rocstories/eval_stories.txt"],
        capture_output=True, text=True
    )
    # Print eval output
    if eval_result.stdout:
        for line in eval_result.stdout.strip().split('\\n'):
            print(f"     {line}")
    if eval_result.stderr:
        # Only print non-empty stderr lines that aren't just warnings
        for line in eval_result.stderr.strip().split('\\n'):
            if line.strip() and 'warning' not in line.lower():
                print(f"     [stderr] {line}")
    # Extract perplexity
    ppl_found = False
    for line in eval_result.stdout.split('\\n'):
        if 'ppl' in line.lower() or 'perplexity' in line.lower():
            results[desc] = line.strip()
            ppl_found = True
    if not ppl_found:
        results[desc] = f"(eval completed, check {out_dir} manually)"
    print(f"  ✓ Eval done")

    # ── Phase 3: GENERATE STORIES ────────────────────────────────────────────
    print(f"  ✍️  [3/3] Generating stories for quality evaluation...")
    gen_result = subprocess.run([
        "python", "sample_batch.py",
        "--init_from=resume", f"--out_dir={out_dir}",
        "--start=FILE:data/rocstories/eval_prompts.txt",
        "--batch_prompts=True", "--max_new_tokens=200",
        f"--output_file={out_dir}/generated_stories.jsonl"
    ], check=False)
    if gen_result.returncode == 0:
        print(f"  ✓ Stories saved to {out_dir}/generated_stories.jsonl")
    else:
        print(f"  ⚠ Story generation had issues (exit code {gen_result.returncode})")

    # ── Save results after each experiment (crash resilience) ────────────────
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    completed += 1
    elapsed = time.time() - exp_start
    total_elapsed = time.time() - overall_start
    remaining = total - i
    avg_per_exp = total_elapsed / (completed + skipped) if (completed + skipped) > 0 else 0
    eta_min = (remaining * avg_per_exp) / 60

    print(f"\\n  ⏱  This experiment: {elapsed/60:.1f} min")
    print(f"  📈 Progress: {i}/{total} done | {remaining} remaining | ETA: ~{eta_min:.0f} min")

# ── Final summary ────────────────────────────────────────────────────────────
total_time = time.time() - overall_start
print(f"\\n{'━'*70}")
print(f"  ABLATION COMPLETE — {completed} trained, {skipped} skipped (had checkpoint)")
print(f"  Total time: {total_time/60:.1f} min ({total_time/3600:.1f} hrs)")
print(f"{'━'*70}")
print()
print(f"{'='*60}")
print(f"  ABLATION RESULTS SUMMARY")
print(f"{'='*60}")
for desc, val in results.items():
    print(f"  {desc:40s}: {val}")
print(f"{'='*60}")
print(f"\\n✓ Results saved to {RESULTS_FILE}")
print(f"  Run CELL 7b next for graphs and analysis!")
"""

# ══════════════════════════════════════════════════════════════════════════════
# CELL 7b — Plot learning curves + Story quality metrics + Bar charts
#
# Run AFTER CELL 7 to generate report-ready charts and tables.
# Generates:
#   1. Ablation overlay: val loss curves (5 architectures)
#   2. Full 6-way comparison: includes structured format experiment
#   3. Single best model: train + val loss curves
#   4. Bar chart: final val loss per experiment
#   5. Story quality metrics table (for report)
# ══════════════════════════════════════════════════════════════════════════════
CELL_7b = """
import os, json
os.chdir(LOCAL_DIR)

!pip install -q matplotlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Discover which experiments actually completed ────────────────────────────
all_experiments = [
    ("out-rocstories-baseline",    "Baseline",        "out-rocstories-baseline/train_log.jsonl"),
    ("out-rocstories-rope",        "+RoPE",           "out-rocstories-rope/train_log.jsonl"),
    ("out-rocstories-ffn",         "+RMSNorm+SwiGLU", "out-rocstories-ffn/train_log.jsonl"),
    ("out-rocstories-qknorm",      "+QK-Norm",        "out-rocstories-qknorm/train_log.jsonl"),
    ("out-rocstories",             "All Modern",      "out-rocstories/train_log.jsonl"),
    ("out-rocstories-structured",  "Structured Fmt",  "out-rocstories-structured/train_log.jsonl"),
]

available = [(d, l, p) for d, l, p in all_experiments if os.path.exists(p)]
print(f"Found {len(available)}/{len(all_experiments)} completed experiments:\\n")
for out_dir, label, log_path in available:
    size_mb = os.path.getsize(log_path) / 1e6
    print(f"  ✓ {label:20s} — {log_path} ({size_mb:.1f} MB)")
missing = [(d, l, p) for d, l, p in all_experiments if not os.path.exists(p)]
for out_dir, label, log_path in missing:
    print(f"  ✗ {label:20s} — NOT FOUND (skipping)")

if len(available) == 0:
    print("\\n❌ No completed experiments found! Run CELL 7 first.")
else:
    # ── 1. Architecture ablation (exclude structured format) ─────────────
    arch_exps = [(d, l, p) for d, l, p in available if "structured" not in d]
    if len(arch_exps) >= 2:
        print(f"\\n{'='*60}")
        print("1. Architecture ablation learning curves...")
        print(f"{'='*60}")
        log_args = []
        labels = []
        for _, label, log_path in arch_exps:
            log_args.extend(["--log", log_path])
            labels.append(label)
        !python plot_training.py {' '.join(log_args)} \\
            --labels "{','.join(labels)}" \\
            --output ablation_curves.png \\
            --title "Task 2: Architecture Ablation — Validation Loss"
    else:
        print("\\n⚠ Not enough architecture experiments for ablation plot")

    # ── 2. Full 6-way comparison (if structured exists too) ──────────────
    if len(available) > len(arch_exps):
        print(f"\\n{'='*60}")
        print("2. Full 6-way comparison (including structured format)...")
        print(f"{'='*60}")
        log_args = []
        labels = []
        for _, label, log_path in available:
            log_args.extend(["--log", log_path])
            labels.append(label)
        !python plot_training.py {' '.join(log_args)} \\
            --labels "{','.join(labels)}" \\
            --output ablation_curves_full.png \\
            --title "Task 2: All Experiments — Validation Loss"

    # ── 3. Single best model training curve ──────────────────────────────
    best_log = "out-rocstories/train_log.jsonl"
    if not os.path.exists(best_log):
        best_log = "out-rocstories-combined/train_log.jsonl"
    if os.path.exists(best_log):
        print(f"\\n{'='*60}")
        print(f"3. Best model training curve ({best_log})...")
        print(f"{'='*60}")
        !python plot_training.py \\
            --log {best_log} \\
            --output training_curve.png \\
            --title "Task 1: Training Curves (152M All Modern)"

    # ── 4. Bar chart: final val loss per experiment ──────────────────────
    print(f"\\n{'='*60}")
    print("4. Final validation loss bar chart...")
    print(f"{'='*60}")
    bar_labels = []
    bar_vals = []
    for out_dir, label, log_path in available:
        with open(log_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        # Find last line with val_loss
        for line in reversed(lines):
            entry = json.loads(line)
            if 'val_loss' in entry and entry['val_loss'] is not None:
                bar_labels.append(label)
                bar_vals.append(entry['val_loss'])
                break
    if bar_vals:
        fig, ax = plt.subplots(figsize=(9, 5))
        colors = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA', '#00ACC1']
        bars = ax.bar(bar_labels, bar_vals, color=colors[:len(bar_vals)], edgecolor='white', linewidth=1.5)
        ax.set_ylabel('Final Validation Loss', fontsize=12)
        ax.set_title('Task 2: Final Validation Loss by Configuration', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        # Add value labels on bars
        for bar, val in zip(bars, bar_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        plt.xticks(rotation=15, ha='right', fontsize=10)
        plt.tight_layout()
        plt.savefig('ablation_bar_chart.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: ablation_bar_chart.png")

        # Print table too
        print(f"\\n  {'Config':25s} {'Val Loss':>10s}")
        print(f"  {'-'*37}")
        for lbl, val in sorted(zip(bar_labels, bar_vals), key=lambda x: x[1]):
            print(f"  {lbl:25s} {val:10.4f}")

    # ── 5. Story quality comparison ──────────────────────────────────────
    quality_inputs = []
    quality_labels = []
    for out_dir, label, _ in available:
        stories_path = os.path.join(out_dir, "generated_stories.jsonl")
        if os.path.exists(stories_path):
            quality_inputs.extend(["--input", stories_path])
            quality_labels.append(label)
        else:
            print(f"  ⚠ No generated stories for {label} — skipping quality eval")

    if quality_labels:
        print(f"\\n{'='*60}")
        print(f"5. Story quality metrics ({len(quality_labels)} models)...")
        print(f"{'='*60}")
        !python eval_story_quality.py {' '.join(quality_inputs)} \\
            --labels "{','.join(quality_labels)}"

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\\n{'━'*60}")
    print("  OUTPUTS GENERATED:")
    for fname in ['ablation_curves.png', 'ablation_curves_full.png',
                   'training_curve.png', 'ablation_bar_chart.png']:
        if os.path.exists(fname):
            print(f"    ✓ {fname}")
        else:
            print(f"    ✗ {fname} (not generated)")
    print(f"{'━'*60}")
    print("  Download the PNG files for your report!")
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
        ("CELL 1 — Install dependencies",                CELL_1),
        ("CELL 2 — Mount Drive + clone repo",             CELL_2),
        ("CELL 3 — GPU memory check (152M model)",        CELL_3),
        ("CELL 4 — Prepare datasets",                     CELL_4),
        ("CELL 5a — Checkpoint detection (resume check)", CELL_5a),
        ("CELL 5 — Train (Task 1 + Task 3)",              CELL_5),
        ("CELL 6 — Evaluate",                             CELL_6),
        ("CELL 7 — 5-way ablation (Task 2)",              CELL_7),
        ("CELL 8 — Upload to HuggingFace (Task 3)",       CELL_8),
        ("CELL 9 — Keep-alive",                           CELL_9),
    ]
    for title, code in cells:
        print(f"\n{'='*70}")
        print(f"# {title}")
        print('='*70)
        print(code)
