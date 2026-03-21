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
#   Optional: up to ~100M TinyStories tokens (cap in data/mixed/prepare.py)
#
# The instruction mixture teaches the model to handle both
# natural prompts and raw continuation, without hurting PPL
# on the plain test set (Ouyang et al., 2022).
#
# Model: ~31.7M params  (within 32M constraint; n_layer=7 sprint)
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
eval_interval         = 250
log_interval          = 10
eval_iters            = 100
eval_only             = False
always_save_checkpoint = False
init_from             = 'scratch'

# ── Logging ──────────────────────────────────────────────────────────────────
wandb_log      = False
wandb_project  = 'rocstories-nanogpt'
wandb_run_name = 'task3-best-30M-mixed'

# ── Dataset ──────────────────────────────────────────────────────────────────
dataset = 'mixed'    # data/mixed/train.bin + val.bin

# ── Data loading ─────────────────────────────────────────────────────────────
# Mixed ROCStories formats + optional TinyStories prefix (~100M cap) → ~104M+ train tokens
# Effective batch = 64 × 1 × 256 = 16,384 tokens/step (template-style)
# 10,000 steps × 16,384 = 164M token-steps (submission sprint)
gradient_accumulation_steps = 1
batch_size  = 64
block_size  = 256

# ── Model — All-Modern 30M ─────────────────────────────────────────────────
# Total params ≈ 31.7M  (within 32M constraint)
n_layer  = 7
n_head   = 6
n_embd   = 384
dropout  = 0.2     # baby-GPT / train_shakespeare_char recipe (same as T1/T2)
bias     = False

use_rmsnorm = True
use_rope    = True
use_swiglu  = True
use_qk_norm = True

label_smoothing = 0.0

# ── Optimizer ────────────────────────────────────────────────────────────────
learning_rate = 1e-3
max_iters     = 10000   # submission — longer than T2 5K; PPL < 25 sprint
weight_decay  = 0.1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

# ── LR schedule ──────────────────────────────────────────────────────────────
decay_lr       = True
warmup_iters   = 100    # ~1% of 10K run
lr_decay_iters = 10000  # must match max_iters
min_lr         = 1e-4

# ── Colab resilience ─────────────────────────────────────────────────────────
ckpt_interval_secs = 900

# ── System ───────────────────────────────────────────────────────────────────
device  = 'cuda'
dtype   = 'bfloat16'
compile = True
