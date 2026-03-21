# ============================================================
# config/train_rocstories_baseline.py
#
# Task 2 (Exploration) — Ablation A: Vanilla GPT architecture.
# No modern improvements: uses learned positional embeddings,
# LayerNorm, and standard GELU MLP — identical to original nanoGPT.
#
# Model size matches the modern config (~152M) for fair comparison.
#
# Usage:
#   python train.py config/train_rocstories_baseline.py
# ============================================================

out_dir   = 'out-rocstories-baseline'
eval_interval  = 500
log_interval   = 10
eval_iters     = 100
always_save_checkpoint = True
init_from = 'scratch'

wandb_log      = False
wandb_project  = 'rocstories-nanogpt'
wandb_run_name = 'rocstories-baseline'

dataset = 'rocstories'

gradient_accumulation_steps = 8
batch_size = 16
block_size = 256

# Same size as the modern model for fair comparison
n_layer  = 12
n_head   = 12
n_embd   = 768
dropout  = 0.1
bias     = False

# Vanilla nanoGPT — no modern improvements
use_rmsnorm = False
use_rope    = False
use_swiglu  = False
use_qk_norm = False

# Training enhancements still applied for fair comparison
label_smoothing = 0.0

learning_rate = 3e-4
max_iters     = 20000
weight_decay  = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr       = True
warmup_iters   = 1000
lr_decay_iters = 20000
min_lr         = 3e-5

ckpt_interval_secs = 900

device  = 'cuda'
dtype   = 'bfloat16'
compile = True
