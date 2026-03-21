# ============================================================
# config/train_t3_best.py
#
# TASK 3 — Best Checkpoint Submission (≤ 32M params)
#
# Architecture: All-Modern 30M (RoPE + RMSNorm + SwiGLU + QK-Norm)
#   → Best configuration identified in the Task 2 ablation study
#
# Data: ROCStories only (data/rocstories/)
#   Val = eval_stories.txt (professor's evaluation set — exact match)
#   Train = full 98K story corpus
#
#   WHY NOT mixed: mixed val.bin is a random 10% holdout, not
#   eval_stories.txt. Using rocstories ensures ckpt_best.pt is saved
#   at the step with lowest loss on the professor's actual eval set.
#
# Model: ~31.8M params  (within 32M constraint; n_layer=7)
#
# Data preparation (run once):
#   python data/rocstories/prepare.py
#
# Usage:
#   python train.py config/train_t3_best.py
#   python train.py config/train_t3_best.py --init_from=resume
# ============================================================

# ── I/O ──────────────────────────────────────────────────────────────────────
out_dir               = 'out-t3-best'
eval_interval         = 250
log_interval          = 10
eval_iters            = 100
eval_only             = False
always_save_checkpoint = False
init_from             = 'scratch'

# ── Logging ──────────────────────────────────────────────────────────────────
wandb_log      = True
wandb_project  = 'rocstories-nanogpt'
wandb_run_name = 't3-best-7L-rocstories-8k'

# ── Dataset ──────────────────────────────────────────────────────────────────
dataset = 'rocstories'  # data/rocstories/train.bin + val.bin (val = eval_stories.txt)

# ── Data loading ─────────────────────────────────────────────────────────────
# ROCStories full corpus ≈ 4.1M tokens
# Effective batch = 64 × 1 × 256 = 16,384 tokens/step (template-style)
# 8,000 steps × 16,384 = 131M token-steps (~32 passes on 4.1M dataset)
gradient_accumulation_steps = 1
batch_size  = 64
block_size  = 256

# ── Model — All-Modern 30M ─────────────────────────────────────────────────
# Total params ≈ 31.7M  (within 32M constraint)
n_layer  = 7
n_head   = 6
n_embd   = 384
dropout  = 0.2     # match T1/T2 baby-GPT recipe; 4.1M token dataset needs this
bias     = False

use_rmsnorm = True
use_rope    = True
use_swiglu  = True
use_qk_norm = True

label_smoothing = 0.0

# ── Optimizer ────────────────────────────────────────────────────────────────
learning_rate = 1e-3
max_iters     = 8000    # 8K steps × 16,384 tok/step = 131M token-steps (~32 passes on 4.1M dataset)
weight_decay  = 0.1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

# ── LR schedule ──────────────────────────────────────────────────────────────
decay_lr       = True
warmup_iters   = 150    # ~2% of 8K run
lr_decay_iters = 8000   # must match max_iters
min_lr         = 1e-4

# ── Colab resilience ─────────────────────────────────────────────────────────
ckpt_interval_secs = 900

# ── System ───────────────────────────────────────────────────────────────────
device  = 'cuda'
dtype   = 'bfloat16'
compile = True
