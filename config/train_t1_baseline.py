# ============================================================
# config/train_t1_baseline.py
#
# TASK 1 — Official Baseline nanoGPT on ROCStories
#
# Follows the official nanoGPT "baby GPT" configuration:
#   https://github.com/karpathy/nanoGPT/blob/master/config/train_shakespeare_char.py
#
# Prof constraint: n_layer=6, n_head=6, n_embd=384  ≈ 30.2M params
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
always_save_checkpoint = True
init_from             = 'scratch'

# ── Logging ──────────────────────────────────────────────────────────────────
wandb_log      = False
wandb_project  = 'rocstories-nanogpt'
wandb_run_name = 'task1-baseline-30M'

# ── Dataset ──────────────────────────────────────────────────────────────────
dataset = 'rocstories'   # data/rocstories/train.bin + val.bin

# ── Data loading ─────────────────────────────────────────────────────────────
# Effective batch = 32 × 4 × 256 = 32,768 tokens/step
# ROCStories train ≈ 3.7M tokens → ~113 steps/epoch
# 10,000 steps ≈ 88 epochs
gradient_accumulation_steps = 4
batch_size  = 32
block_size  = 256

# ── Model — Official baby GPT config ─────────────────────────────────────────
# Total params ≈ 30.2M  (within 32M constraint)
n_layer  = 6
n_head   = 6
n_embd   = 384
dropout  = 0.1
bias     = False

# Vanilla nanoGPT — no modern improvements (Task 1 requirement)
use_rmsnorm = False
use_rope    = False
use_swiglu  = False
use_qk_norm = False

# Label smoothing for regularisation on the small dataset
label_smoothing = 0.1

# ── Optimizer ────────────────────────────────────────────────────────────────
learning_rate = 6e-4    # standard peak LR for ~30M GPT (Kaplan et al., 2020)
max_iters     = 10000
weight_decay  = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# ── LR schedule — cosine decay ───────────────────────────────────────────────
decay_lr       = True
warmup_iters   = 200    # short warmup; dataset is small
lr_decay_iters = 10000  # decay to min_lr by the end of training
min_lr         = 6e-5   # 0.1 × peak_lr (Chinchilla scaling)

# ── Colab resilience ─────────────────────────────────────────────────────────
ckpt_interval_secs = 900  # time-based checkpoint every 15 min

# ── System ───────────────────────────────────────────────────────────────────
device  = 'cuda'
dtype   = 'bfloat16'
compile = True           # ~15% speedup via torch.compile (PyTorch ≥ 2.0)
