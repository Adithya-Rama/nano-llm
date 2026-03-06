# ============================================================
# config/train_rocstories_combined.py
#
# Train on combined ROCStories + TinyStories dataset.
# This provides ~60-80M tokens of narrative text instead of
# just ~2.25M from ROCStories alone, significantly reducing
# overfitting and improving generalization.
#
# Use this config for Task 3 (best checkpoint submission).
# The model will learn general story patterns from TinyStories
# and specific 5-sentence narrative arcs from ROCStories.
#
# Data preparation:
#   python data/rocstories/prepare.py
#   python data/tinystories/prepare.py
#   python data/combined/prepare.py
#
# Usage:
#   python train.py config/train_rocstories_combined.py
# ============================================================

# ── I/O ──────────────────────────────────────────────────────────────────────
out_dir    = 'out-rocstories-combined'
eval_interval  = 500
log_interval   = 10
eval_iters     = 100
eval_only      = False
always_save_checkpoint = True
init_from = 'scratch'

# ── Logging ──────────────────────────────────────────────────────────────────
wandb_log     = False
wandb_project = 'rocstories-nanogpt'
wandb_run_name = 'rocstories-combined-152M'

# ── Dataset ──────────────────────────────────────────────────────────────────
dataset = 'combined'        # data/combined/train.bin + val.bin

# ── Data loading ─────────────────────────────────────────────────────────────
# With combined dataset (~60M+ tokens), 20K steps covers fewer epochs
# but gives better generalization due to diverse training data
gradient_accumulation_steps = 8
batch_size  = 16
block_size  = 256

# ── Model ────────────────────────────────────────────────────────────────────
n_layer  = 12
n_head   = 12
n_embd   = 768
dropout  = 0.05          # Lower dropout with more data (less overfit risk)
bias     = False

# All modern improvements ON
use_rmsnorm = True
use_rope    = True
use_swiglu  = True
use_qk_norm = True

# Training enhancements
use_gradient_checkpointing = False
label_smoothing = 0.1

# ── Optimizer ────────────────────────────────────────────────────────────────
learning_rate = 3e-4
max_iters     = 20000
weight_decay  = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# ── LR schedule ──────────────────────────────────────────────────────────────
decay_lr       = True
warmup_iters   = 1000
lr_decay_iters = 20000
min_lr         = 3e-5

# ── Colab resilience ─────────────────────────────────────────────────────────
ckpt_interval_secs = 900

# ── System ───────────────────────────────────────────────────────────────────
device  = 'cuda'
dtype   = 'bfloat16'
compile = True
