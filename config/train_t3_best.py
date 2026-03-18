# ============================================================
# config/train_t3_best.py
#
# TASK 3 — Best Checkpoint Submission (≤ 32M params)
#
# Architecture: All-Modern 30M (RoPE + RMSNorm + SwiGLU + QK-Norm)
#   → Best configuration identified in the Task 2 ablation study
#
# Data: Mixed instruction dataset (data/mixed/)
#   55% plain continuation (what PPL evaluation uses)
#   30% instruction-prefixed stories
#   15% structured XML stories
#   Optional: +20M TinyStories tokens for richer language
#
# The instruction mixture teaches the model to handle both
# natural prompts and raw continuation, without hurting PPL
# on the plain test set (Ouyang et al., 2022).
#
# Model: ~30.1M params  (within 32M constraint)
#
# Data preparation (run in order):
#   python data/rocstories/prepare.py
#   python data/rocstories/prepare.py --structured
#   python data/tinystories/prepare.py   # optional
#   python data/mixed/prepare.py         # or --with_tinystories
#
# Usage:
#   python train.py config/train_t3_best.py
#   python train.py config/train_t3_best.py --init_from=resume
# ============================================================

# ── I/O ──────────────────────────────────────────────────────────────────────
out_dir               = 'out-t3-best'
eval_interval         = 500
log_interval          = 10
eval_iters            = 200
eval_only             = False
always_save_checkpoint = True
init_from             = 'scratch'

# ── Logging ──────────────────────────────────────────────────────────────────
wandb_log      = False
wandb_project  = 'rocstories-nanogpt'
wandb_run_name = 'task3-best-30M-mixed'

# ── Dataset ──────────────────────────────────────────────────────────────────
dataset = 'mixed'    # data/mixed/train.bin + val.bin

# ── Data loading ─────────────────────────────────────────────────────────────
# Mixed dataset ≈ 3.7M × (1/0.55) ≈ 6.7M tokens
# Effective batch = 32 × 4 × 256 = 32,768 tokens/step
# 15,000 steps × 32,768 = 491M token-steps ≈ 73 passes
gradient_accumulation_steps = 4
batch_size  = 32
block_size  = 256

# ── Model — All-Modern 30M ─────────────────────────────────────────────────
# Total params ≈ 30.1M  (within 32M constraint)
n_layer  = 6
n_head   = 6
n_embd   = 384
dropout  = 0.05    # lower dropout: mixed data provides implicit regularisation
bias     = False

use_rmsnorm = True
use_rope    = True
use_swiglu  = True
use_qk_norm = True

label_smoothing = 0.1

# ── Optimizer ────────────────────────────────────────────────────────────────
learning_rate = 6e-4
max_iters     = 15000
weight_decay  = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# ── LR schedule ──────────────────────────────────────────────────────────────
decay_lr       = True
warmup_iters   = 500
lr_decay_iters = 15000
min_lr         = 6e-5

# ── Colab resilience ─────────────────────────────────────────────────────────
ckpt_interval_secs = 900

# ── System ───────────────────────────────────────────────────────────────────
device  = 'cuda'
dtype   = 'bfloat16'
compile = True
