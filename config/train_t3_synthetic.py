# ============================================================
# config/train_t3_synthetic.py
#
# TASK 3 — Synthetic Data Retrain (≤ 32M params)
#
# Full retrain from scratch on gpt-oss-120B synthetic stories
# mixed with original ROCStories.
#
# WHY: Current T3 best PPL = 23.54 (step 3000) — hit hard
# capacity ceiling on 4.1M original tokens. With 22M tokens
# of high-quality synthetic data:
#   - Model learns causal narrative reasoning (not memorisation)
#   - Train/val gap stays low for far more steps
#   - Qwen quality scores dramatically improve
#   - PPL expected to drop to ~20-22
#
# Architecture: IDENTICAL to train_t3_best.py — must stay ≤32M.
# Val.bin = original ROCStories only → authentic PPL signal.
#
# PREREQS:
#   1. python data/rocstories/prepare.py
#   2. python data/rocstories_synthetic/prepare.py \
#        --json_path synthetic_stories_gptoss120b.json
#
# Usage:
#   python train.py config/train_t3_synthetic.py
# ============================================================

# ── I/O ──────────────────────────────────────────────────────────────────────
out_dir               = 'out-t3-synthetic'
eval_interval         = 250
log_interval          = 10
eval_iters            = 200         # more eval iters — larger val set now
eval_only             = False
always_save_checkpoint = False
init_from             = 'scratch'

# ── Logging ──────────────────────────────────────────────────────────────────
wandb_log      = True
wandb_project  = 'rocstories-nanogpt'
wandb_run_name = 't3-synthetic-7L-31.8M-gptoss120b'

# ── Dataset ──────────────────────────────────────────────────────────────────
dataset = 'rocstories_synthetic'    # synthetic + original mix

# ── Data loading ─────────────────────────────────────────────────────────────
# 22M token corpus (19.5M synthetic + 2.9M original)
# Effective batch = 64 × 1 × 256 = 16,384 tokens/step
# 12,000 steps × 16,384 = 196M token-steps ≈ 9 passes over 22M corpus
# Compare: original 32 passes over 4.1M → overfit at step 3000
# 9 passes over 22M → much healthier generalisation
gradient_accumulation_steps = 1
batch_size  = 64
block_size  = 256

# ── Model — IDENTICAL to train_t3_best.py (≤32M constraint) ───────────────
n_layer  = 7
n_head   = 6
n_embd   = 384
dropout  = 0.15    # slightly less dropout — synthetic data is cleaner/
                   # more consistent than raw human text; less noise to fight
bias     = False

use_rmsnorm = True
use_rope    = True
use_swiglu  = True
use_qk_norm = True

label_smoothing = 0.0

# ── Optimizer ────────────────────────────────────────────────────────────────
learning_rate = 1e-3
max_iters     = 12000   # 12K steps — 9 passes over 22M token corpus
                        # More steps than original 8K because dataset is 5x larger
weight_decay  = 0.1
beta1         = 0.9
beta2         = 0.99
grad_clip     = 1.0

# ── LR schedule ──────────────────────────────────────────────────────────────
decay_lr       = True
warmup_iters   = 200     # ~1.7% of 12K run
lr_decay_iters = 12000   # must match max_iters
min_lr         = 1e-4

# ── Colab resilience ─────────────────────────────────────────────────────────
ckpt_interval_secs = 900

# ── System ───────────────────────────────────────────────────────────────────
device  = 'cuda'
dtype   = 'bfloat16'
compile = True
