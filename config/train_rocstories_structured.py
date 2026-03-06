# ============================================================
# config/train_rocstories_structured.py
#
# Task 2 (Exploration) — Ablation E: Structured Story Format.
# Compares training with explicit sentence markers
# (<story><s1>...</s1>...<s5>...</s5></story>) against
# plain text format.
#
# Hypothesis: Explicit structure markers help small models
# learn the 5-sentence narrative arc of ROCStories, improving
# coherence and Qwen scores.
#
# Data preparation (run first):
#   python data/rocstories/prepare.py --structured
#
# Usage:
#   python train.py config/train_rocstories_structured.py
# ============================================================

out_dir   = 'out-rocstories-structured'
eval_interval  = 500
log_interval   = 10
eval_iters     = 100
always_save_checkpoint = True
init_from = 'scratch'

wandb_log      = False
wandb_project  = 'rocstories-nanogpt'
wandb_run_name = 'rocstories-structured'

# Points to structured data directory
dataset = 'rocstories_structured'

gradient_accumulation_steps = 8
batch_size = 16
block_size = 256

# Same full modern architecture as the main config
n_layer  = 12
n_head   = 12
n_embd   = 768
dropout  = 0.1
bias     = False

use_rmsnorm = True
use_rope    = True
use_swiglu  = True
use_qk_norm = True

label_smoothing = 0.1

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
