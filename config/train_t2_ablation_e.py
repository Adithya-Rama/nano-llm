# ============================================================
# config/train_t2_ablation_e.py
#
# TASK 2 — Ablation E: All Modern (LLaMA-style at 30M)
#
# Full combination of all architectural improvements:
#   RoPE + RMSNorm + SwiGLU + QK-Norm
#
# This is the "fully modernised" 30M model and serves as
# the upper bound for the architecture ablation study.
# Comparing with A–D isolates each component's contribution.
#
# Model: ~30.1M params  (within 32M constraint)
# ============================================================

out_dir               = 'out-t2-all-modern'
eval_interval         = 250
log_interval          = 10
eval_iters            = 100
always_save_checkpoint = True
init_from             = 'scratch'

wandb_log      = False
wandb_project  = 'rocstories-ablations'
wandb_run_name = 't2-ablation-e-all-modern'

dataset = 'rocstories'

gradient_accumulation_steps = 4
batch_size  = 32
block_size  = 256

n_layer  = 6
n_head   = 6
n_embd   = 384
dropout  = 0.1
bias     = False

use_rmsnorm = True
use_rope    = True
use_swiglu  = True
use_qk_norm = True

label_smoothing = 0.1

learning_rate = 6e-4
max_iters     = 10000
weight_decay  = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr       = True
warmup_iters   = 200
lr_decay_iters = 10000
min_lr         = 6e-5

ckpt_interval_secs = 900

device  = 'cuda'
dtype   = 'bfloat16'
compile = True
