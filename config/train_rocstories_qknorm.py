# ============================================================
# config/train_rocstories_qknorm.py
#
# Task 2 (Exploration) — Ablation D: +QK-Norm only.
# Tests the isolated contribution of Query-Key Normalisation
# (Gemma 2, Cohere Command-R, 2024) over the vanilla baseline.
#
# QK-Norm applies RMSNorm to Q and K per head before attention
# scores, preventing logit explosion and enabling stable training.
#
# Model size matches the modern config for fair comparison.
# ============================================================

out_dir   = 'out-rocstories-qknorm'
eval_interval  = 500
log_interval   = 10
eval_iters     = 100
always_save_checkpoint = True
init_from = 'scratch'

wandb_log      = False
wandb_project  = 'rocstories-nanogpt'
wandb_run_name = 'rocstories-qknorm'

dataset = 'rocstories'

gradient_accumulation_steps = 8
batch_size = 16
block_size = 256

n_layer  = 12
n_head   = 12
n_embd   = 768
dropout  = 0.1
bias     = False

use_rmsnorm = False   # LayerNorm
use_rope    = False   # Learned positional embeddings
use_swiglu  = False   # GELU MLP
use_qk_norm = True    # <<< QK-Norm (novel contribution)

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
