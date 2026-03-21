# ============================================================
# config/train_t1_baseline.py
#
# TASK 1 — Official Baseline nanoGPT on ROCStories
#
# Follows the official nanoGPT "baby GPT" configuration:
#   https://github.com/karpathy/nanoGPT/blob/master/config/train_shakespeare_char.py
#
# Prof constraint: n_layer=7, n_head=6, n_embd=384  ≈ 31.8M params (PPL sprint)
#                 (must not exceed 32M total params)
#
# Architecture: Vanilla nanoGPT (learned PE, LayerNorm, GELU MLP)
# No modern improvements — this is the required Task 1 baseline.
#
# Usage:
#   python train.py config/train_t1_baseline.py
#   python train.py config/train_t1_baseline.py --init_from=resume
# ============================================================

# ── I/O ──────────────────────────────────────────────────────────────────────
out_dir               = 'out-t1-baseline'
eval_interval         = 250
log_interval          = 10
eval_iters            = 100
eval_only             = False
always_save_checkpoint = False  # nanoGPT char: only save when val improves (overfit expected)
init_from             = 'scratch'

# ── Logging ──────────────────────────────────────────────────────────────────
wandb_log      = True
wandb_project  = 'rocstories-nanogpt'
wandb_run_name = 't1-baseline-7L-31.8M'

# ── Dataset ──────────────────────────────────────────────────────────────────
dataset = 'rocstories'   # data/rocstories/train.bin + val.bin

# ── Data loading ─────────────────────────────────────────────────────────────
# Effective batch = 64 × 1 × 256 = 16,384 tokens/step (nanoGPT train_shakespeare_char style)
# ROCStories train ≈ full corpus (~4.1M tok); val = 500 stories monitor only
gradient_accumulation_steps = 1
batch_size  = 64
block_size  = 256

# ── Model — Official baby GPT config ─────────────────────────────────────────
# Total params ≈ 31.8M  (within 32M constraint)
n_layer  = 7
n_head   = 6
n_embd   = 384
dropout  = 0.2    # nanoGPT train_shakespeare_char style — baby net + small data
bias     = False

# Vanilla nanoGPT — no modern improvements (Task 1 requirement)
use_rmsnorm = False
use_rope    = False
use_swiglu  = False
use_qk_norm = False

# No label smoothing — val loss is unsmoothed CE (template has no smoothing)
label_smoothing = 0.0

# ── Optimizer ────────────────────────────────────────────────────────────────
learning_rate = 1e-3    # baby net + small data (nanoGPT train_shakespeare_char), not GPT-2 OWT defaults
max_iters     = 5000    # best val at step 2250 — no need to train past 5K
weight_decay  = 0.1
beta1 = 0.9
beta2 = 0.99   # slightly higher — few tokens per iter vs large-scale pretraining
grad_clip = 1.0

# ── LR schedule — cosine decay ───────────────────────────────────────────────
decay_lr       = True
warmup_iters   = 100    # ~2% of 5K run
lr_decay_iters = 5000   # must match max_iters
min_lr         = 1e-4   # 0.1 × peak_lr for 1e-3 schedule

# ── Colab resilience ─────────────────────────────────────────────────────────
ckpt_interval_secs = 900  # time-based checkpoint every 15 min

# ── System ───────────────────────────────────────────────────────────────────
device  = 'cuda'
dtype   = 'bfloat16'
compile = True           # ~15% speedup via torch.compile (PyTorch ≥ 2.0)
