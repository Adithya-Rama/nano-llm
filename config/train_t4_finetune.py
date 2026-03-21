# ============================================================
# config/train_t4_finetune.py
#
# TASK 4 — Arena Model (Stage 2: Instruction Fine-tune)
#
# Resumes from Stage 1 pretrain checkpoint (out-t4-arena/ckpt.pt).
# Fine-tunes on instruction-format stories so the model learns
# to respond to "Write a story about: {title}" prompts for arena judging.
#
# PREREQS (run in order):
#   1. python data/rocstories/prepare.py         (creates rocstories/val.bin)
#   2. python data/rocstories_instruction/prepare.py
#   3. python train.py config/train_t4_arena.py  (Stage 1 pretrain)
#   4. mkdir -p out-t4-arena
#      cp out-t4-pretrain/ckpt_best.pt out-t4-arena/ckpt.pt
#   5. python train.py config/train_t4_finetune.py   (this config)
#
# Architecture must exactly match Stage 1 (train_t4_arena.py).
# ============================================================

# ── I/O ──────────────────────────────────────────────────────────────────────
out_dir               = 'out-t4-arena'       # saves fine-tuned checkpoints here
init_from             = 'resume'             # resume from out-t4-arena/ckpt.pt (Stage 1 best)
eval_interval         = 200
log_interval          = 10
eval_iters            = 100
eval_only             = False
always_save_checkpoint = False

# ── Logging ──────────────────────────────────────────────────────────────────
wandb_log      = True
wandb_project  = 'rocstories-nanogpt'
wandb_run_name = 't4-finetune-124M-2k'

# ── Dataset ──────────────────────────────────────────────────────────────────
dataset = 'rocstories_instruction'  # instruction-format stories for fine-tuning

# ── Data loading ─────────────────────────────────────────────────────────────
# Same batch config as Stage 1 — must match for resume to work cleanly
# Effective batch = 16 × 8 × 512 = 65,536 tokens/step
batch_size                  = 16
gradient_accumulation_steps = 8
block_size                  = 512

# ── Model — must exactly match Stage 1 architecture ──────────────────────────
n_layer  = 12
n_head   = 12
n_embd   = 768
bias     = False

use_rope    = True
use_rmsnorm = True
use_swiglu  = True
use_qk_norm = True
use_gradient_checkpointing = True   # needed for 124M on Colab

# ── Regularisation ───────────────────────────────────────────────────────────
dropout         = 0.05    # light dropout for fine-tuning (was 0.1 in pretrain)
label_smoothing = 0.0

# ── Optimizer — lower LR for fine-tuning to avoid forgetting ─────────────────
learning_rate = 1e-4    # 3× lower than Stage 1 pretrain (3e-4)
min_lr        = 1e-5
beta1         = 0.9
beta2         = 0.99    # higher than pretrain's 0.95 — fine-tuning is more stable
weight_decay  = 0.1
grad_clip     = 1.0

# ── LR schedule ──────────────────────────────────────────────────────────────
decay_lr       = True
warmup_iters   = 50      # short warmup — already pretrained
max_iters      = 2000    # just enough to learn instruction format
lr_decay_iters = 2000
min_lr         = 1e-5

# ── Colab resilience ─────────────────────────────────────────────────────────
ckpt_interval_secs = 900

# ── System ───────────────────────────────────────────────────────────────────
device  = 'cuda'
dtype   = 'bfloat16'
compile = True
