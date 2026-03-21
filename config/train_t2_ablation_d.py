# ============================================================
# config/train_t2_ablation_d.py
#
# TASK 2 — Ablation D: +QK-Norm  [Novel Contribution]
#
# Query-Key Normalisation applies RMSNorm to Q and K vectors
# per attention head *before* the dot-product is computed.
# This prevents attention logit explosion at higher learning
# rates — a stability problem particularly acute in small
# models trained on narrow domains like ROCStories.
#
# Recently adopted by: Gemma 2 (Google, 2024), Cohere
# Command-R, and Meta Chameleon (2024). Its impact on
# sub-32M models trained on short narrative text has NOT
# been previously studied — this is our novel contribution.
#
# Change from Ablation A:  use_qk_norm = True
#
# Reference:
#   Henry, A. et al. (2020). Query-key normalization for
#     transformers. arXiv:2010.04245.
#   Team, G. (2024). Gemma 2: Improving open language
#     models at a practical size. arXiv:2408.00118.
#
# Model: ~31.8M params  (QK-Norm adds only 2×head_dim=128 params per layer)
# ============================================================

out_dir               = 'out-t2-qknorm'
eval_interval         = 250
log_interval          = 10
eval_iters            = 100
always_save_checkpoint = False
init_from             = 'scratch'

wandb_log      = False
wandb_project  = 'rocstories-ablations'
wandb_run_name = 't2-ablation-d-qknorm'

dataset = 'rocstories'

gradient_accumulation_steps = 1
batch_size  = 64
block_size  = 256

n_layer  = 7
n_head   = 6
n_embd   = 384
dropout  = 0.2
bias     = False

use_rmsnorm = False
use_rope    = False
use_swiglu  = False
use_qk_norm = True     # ← only change (novel contribution)

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
