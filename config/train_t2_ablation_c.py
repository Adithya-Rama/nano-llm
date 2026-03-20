# ============================================================
# config/train_t2_ablation_c.py
#
# TASK 2 — Ablation C: +RMSNorm + SwiGLU
#
# RMSNorm replaces LayerNorm: removes the mean-centering
# operation, reducing compute while maintaining stability
# (Zhang & Sennrich, 2019).
#
# SwiGLU replaces GELU MLP: uses a gated architecture that
# has become the default in LLaMA, Mistral, and Gemma
# (Shazeer, 2020). Hidden dimension is scaled by 2/3 to
# keep total parameter count equal to the baseline MLP.
#
# Change from Ablation A:  use_rmsnorm = True, use_swiglu = True
#
# References:
#   Zhang, B. & Sennrich, R. (2019). Root mean square layer
#     normalization. arXiv:1910.07467.
#   Shazeer, N. (2020). GLU variants improve transformer.
#     arXiv:2002.05202.
#
# Model: ~30.2M params  (SwiGLU 8/3 scaling keeps param count equal)
# ============================================================

out_dir               = 'out-t2-ffn'
eval_interval         = 250
log_interval          = 10
eval_iters            = 100
always_save_checkpoint = False
init_from             = 'scratch'

wandb_log      = False
wandb_project  = 'rocstories-ablations'
wandb_run_name = 't2-ablation-c-rmsnorm-swiglu'

dataset = 'rocstories'

gradient_accumulation_steps = 4
batch_size  = 32
block_size  = 256

n_layer  = 6
n_head   = 6
n_embd   = 384
dropout  = 0.2
bias     = False

use_rmsnorm = True     # ← changed
use_rope    = False
use_swiglu  = True     # ← changed
use_qk_norm = False

label_smoothing = 0.1

learning_rate = 1e-3
max_iters     = 5000    # best val at step ~1750 — stop before overfitting
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
