# ============================================================
# config/train_t2_ablation_b.py
#
# TASK 2 — Ablation B: +RoPE (Rotary Positional Embedding)
#
# Hypothesis: RoPE encodes *relative* token distances rather
# than absolute positions, which should generalise better
# across story lengths and help the model learn narrative
# dependencies (Su et al., 2022).
#
# Change from Ablation A:  use_rope = True
# Everything else identical for clean isolated measurement.
#
# Reference: Su, J. et al. (2022). RoFormer: Enhanced transformer
#   with rotary position embedding. arXiv:2104.09864.
#
# Model: ~30.1M params  (RoPE removes learned pos-emb → 0.1M less)
# ============================================================

out_dir               = 'out-t2-rope'
eval_interval         = 250
log_interval          = 10
eval_iters            = 100
always_save_checkpoint = False
init_from             = 'scratch'

wandb_log      = False
wandb_project  = 'rocstories-ablations'
wandb_run_name = 't2-ablation-b-rope'

dataset = 'rocstories'

gradient_accumulation_steps = 4
batch_size  = 32
block_size  = 256

n_layer  = 6
n_head   = 6
n_embd   = 384
dropout  = 0.2
bias     = False

use_rmsnorm = False
use_rope    = True     # ← only change
use_swiglu  = False
use_qk_norm = False

label_smoothing = 0.1

learning_rate = 1e-3
max_iters     = 5000    # best val at step ~1500 — stop before overfitting
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
