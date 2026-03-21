# ============================================================
# config/train_t2_ablation_a.py
#
# TASK 2 — Ablation A: Vanilla GPT (reference point)
#
# Same architecture as Task 1 baseline but trained for
# 10,000 steps to match the other ablation experiments.
# Serves as the reference for measuring the gain from each
# architectural modification in Ablations B–E.
#
# Model: ~31.8M params  (within 32M constraint; n_layer=7 sprint)
# ============================================================

out_dir               = 'out-t2-vanilla'
eval_interval         = 250
log_interval          = 10
eval_iters            = 100
always_save_checkpoint = False
init_from             = 'scratch'

wandb_log      = False
wandb_project  = 'rocstories-ablations'
wandb_run_name = 't2-ablation-a-vanilla'

dataset = 'rocstories'

gradient_accumulation_steps = 1
batch_size  = 64
block_size  = 256

n_layer  = 7
n_head   = 6
n_embd   = 384
dropout  = 0.2    # match train_shakespeare_char / baby-GPT recipe
bias     = False

use_rmsnorm = False
use_rope    = False
use_swiglu  = False
use_qk_norm = False

label_smoothing = 0.0

learning_rate = 1e-3
max_iters     = 5000    # best val at step ~2250 — stop before overfitting
weight_decay  = 0.1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

decay_lr       = True
warmup_iters   = 100
lr_decay_iters = 5000   # must match max_iters
min_lr         = 1e-4

ckpt_interval_secs = 900

device  = 'cuda'
dtype   = 'bfloat16'
compile = True
