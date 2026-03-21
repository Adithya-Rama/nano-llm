# ============================================================
# config/train_rocstories.py
#
# Task 1 (primary) + Task 3 (competition submission) config.
# Modern LLaMA-style architecture with QK-Norm:
#   RoPE + RMSNorm + SwiGLU + QK-Norm (Gemma 2).
#
# Model: ~152M parameters (12L/12H/768D)
# Fits comfortably on A100 40GB with batch_size=16.
#
# Recommended hardware: A100 (Colab Pro).
# Training time estimate:
#   A100 (40GB VRAM): ~40–50 min for 20,000 steps
#
# Usage (from nanoGPT/ directory):
#   python train.py config/train_rocstories.py
#
# Resume after a crash:
#   python train.py config/train_rocstories.py --init_from=resume
# ============================================================

# ── I/O ──────────────────────────────────────────────────────────────────────
out_dir    = 'out-rocstories'
eval_interval  = 500        # evaluate every N steps
log_interval   = 10         # print loss every N steps
eval_iters     = 100        # how many batches to average for val loss
eval_only      = False
always_save_checkpoint = True   # save best val-loss checkpoint always
init_from = 'scratch'

# ── Logging (Weights & Biases — optional) ────────────────────────────────────
wandb_log     = False
wandb_project = 'rocstories-nanogpt'
wandb_run_name = 'rocstories-modern-152M'

# ── Dataset ──────────────────────────────────────────────────────────────────
dataset = 'rocstories'           # data/rocstories/train.bin + val.bin

# ── Data loading ─────────────────────────────────────────────────────────────
# Effective batch size = gradient_accumulation_steps × batch_size × block_size tokens
#   = 8 × 16 × 256 = 32,768 tokens/step
# ROCStories training set ≈ 2.25M tokens → ~69 steps per "epoch"
# 20,000 steps ≈ 290 passes through the dataset
gradient_accumulation_steps = 8
batch_size  = 16          # sequences per micro-batch (fits A100 with ~152M model)
block_size  = 256         # context window: stories rarely exceed 150 tokens

# ── Model ────────────────────────────────────────────────────────────────────
# ~152M parameter LLaMA-style model with QK-Norm.
# 12 layers, 12 heads, 768-dim — 2.5× bigger than the default config.
n_layer  = 12
n_head   = 12
n_embd   = 768
dropout  = 0.1            # regularisation; important for small datasets
bias     = False          # no bias → slightly faster + better generalisation

# Modern architecture improvements
use_rmsnorm = True        # RMSNorm: numerically stable, no mean-centering bias
use_rope    = True        # RoPE: relative positions, better on short stories
use_swiglu  = True        # SwiGLU: gated activations, standard in LLaMA/Mistral
use_qk_norm = True        # QK-Norm: stabilises attention (Gemma 2, 2024)

# Training enhancements
use_gradient_checkpointing = False  # Not needed — 152M fits on A100 without it
label_smoothing = 0.0

# ── Optimizer ────────────────────────────────────────────────────────────────
learning_rate = 3e-4      # peak LR; lower than 6e-4 for stability with larger model
max_iters     = 20000
weight_decay  = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# ── LR schedule ──────────────────────────────────────────────────────────────
decay_lr      = True
warmup_iters  = 1000      # longer warmup for larger model stability
lr_decay_iters = 20000    # cosine decay ends here = max_iters
min_lr        = 3e-5      # floor = 0.1 × max_lr (Chinchilla rule)

# ── Colab resilience ─────────────────────────────────────────────────────────
ckpt_interval_secs = 900  # time-based checkpoint every 15 minutes

# ── System ───────────────────────────────────────────────────────────────────
device  = 'cuda'
dtype   = 'bfloat16'      # A100 supports bfloat16 natively
compile = True            # torch.compile gives ~15% speedup on PyTorch >= 2.0
