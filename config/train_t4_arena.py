# ============================================================
# config/train_t4_arena.py
#
# TASK 4 (Optional) — Arena Competition Model
# NO model size limit for the arena competition.
#
# Architecture: LLaMA-style 152M (same as original code_v1 runs)
#   12 layers, 12 heads, 768-dim
#
# Data: Combined ROCStories + TinyStories (data/combined/)
#   This gives the model rich language + narrative patterns
#   from ~110M tokens, then fine-tunes on ROCStories style.
#
# NOTE: This model is ONLY for the arena competition (Task 4).
#       It must NOT be submitted to HuggingFace for Task 3
#       as it exceeds the 32M parameter constraint.
#
# Data preparation:
#   python data/rocstories/prepare.py
#   python data/tinystories/prepare.py
#   python data/combined/prepare.py
#
# Usage:
#   python train.py config/train_t4_arena.py
#   python train.py config/train_t4_arena.py --init_from=resume
# ============================================================

# ── I/O ──────────────────────────────────────────────────────────────────────
out_dir               = 'out-t4-arena'
eval_interval         = 500
log_interval          = 20   # heavier model — log every 20 steps to keep JSONL manageable
eval_iters            = 100
eval_only             = False
always_save_checkpoint = True
init_from             = 'scratch'

# ── Logging ──────────────────────────────────────────────────────────────────
wandb_log      = False
wandb_project  = 'rocstories-nanogpt'
wandb_run_name = 'task4-arena-152M'

# ── Dataset ──────────────────────────────────────────────────────────────────
dataset = 'combined'   # data/combined/train.bin + val.bin

# ── Data loading ─────────────────────────────────────────────────────────────
# Combined ≈ 110M tokens
# Effective batch = 32 × 8 × 512 = 131,072 tokens/step
# 20,000 steps × 131,072 = 2.6B token-steps ≈ 26 passes (combined corpus)
# block_size=512: WritingPrompts stories avg 600 tokens — must be at least 512
gradient_accumulation_steps = 8
batch_size  = 32
block_size  = 512

# ── Model — 152M (NO size constraint for arena) ──────────────────────────────
n_layer  = 12
n_head   = 12
n_embd   = 768
dropout  = 0.0     # no dropout for large-scale pretraining
bias     = False

use_rmsnorm = True
use_rope    = True
use_swiglu  = True
use_qk_norm = True

label_smoothing = 0.0    # skip label smoothing for pretraining

# ── Optimizer ────────────────────────────────────────────────────────────────
learning_rate = 3e-4    # lower LR for stability with 152M model
max_iters     = 20000
weight_decay  = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# ── LR schedule ──────────────────────────────────────────────────────────────
decay_lr       = True
warmup_iters   = 200     # ~1% of 20K run
lr_decay_iters = 20000
min_lr         = 3e-5

# ── Colab resilience ─────────────────────────────────────────────────────────
ckpt_interval_secs = 900

# ── Memory management ────────────────────────────────────────────────────────
# 152M model in bfloat16 + batch=32 + block=512 ≈ 22–24 GB VRAM.
# Set to True on a 16 GB GPU (e.g., T4) — trades ~20% speed for ~30% less VRAM.
use_gradient_checkpointing = False   # set True if OOM

# ── System ───────────────────────────────────────────────────────────────────
device  = 'cuda'
dtype   = 'bfloat16'
compile = True
