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
# Usage — Stage 1 (this config):
#   python train.py config/train_t4_arena.py
#
# After Stage 1 completes, run Stage 2:
#   mkdir -p out-t4-arena
#   cp out-t4-pretrain/ckpt_best.pt out-t4-arena/ckpt.pt
#   python train.py config/train_t4_finetune.py
# ============================================================

# ── I/O ──────────────────────────────────────────────────────────────────────
out_dir               = 'out-t4-pretrain'
eval_interval         = 500
log_interval          = 20   # heavier model — log every 20 steps to keep JSONL manageable
eval_iters            = 100
eval_only             = False
always_save_checkpoint = True
init_from             = 'scratch'

# ── Logging ──────────────────────────────────────────────────────────────────
wandb_log      = True
wandb_project  = 'rocstories-nanogpt'
wandb_run_name = 't4-pretrain-124M-30k-v2'

# ── Dataset ──────────────────────────────────────────────────────────────────
dataset = 'combined'   # data/combined/train.bin + val.bin

# ── Data loading ─────────────────────────────────────────────────────────────
# Combined ≈ 220M tokens (ROCStories 5x upsampled + full 200M TinyStories)
# Effective batch = 32 × 8 × 512 = 131,072 tokens/step
# 30,000 steps × 131,072 = 3.93B token-steps ≈ 18 passes (healthy for 124M)
# Best checkpoint expected around step 15K-20K (vs step 2K with 54M corpus)
gradient_accumulation_steps = 8
batch_size  = 32
block_size  = 512

# ── Model — 152M (NO size constraint for arena) ──────────────────────────────
n_layer  = 12
n_head   = 12
n_embd   = 768
dropout  = 0.1     # regularise — 124M model on 220M tokens needs this
bias     = False

use_rmsnorm = True
use_rope    = True
use_swiglu  = True
use_qk_norm = True

label_smoothing = 0.0    # skip label smoothing for pretraining

# ── Optimizer ────────────────────────────────────────────────────────────────
# NOTE: Not using train_shakespeare_char recipe (1e-3 / dropout 0.2) — 152M + large corpus = pretraining scale
learning_rate = 3e-4    # lower LR for stability with 152M model
max_iters     = 30000
weight_decay  = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# ── LR schedule ──────────────────────────────────────────────────────────────
decay_lr       = True
warmup_iters   = 300     # ~1% of 30K run
lr_decay_iters = 30000
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
