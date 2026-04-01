# nanoGPT — ROCStories Story Generation
### COMP4680/8650 Advanced Topics in Machine Learning — ANU 2026 S1
**Student:** Adithya Rama | **wandb:** `rocstories-nanogpt` (adithyaiyer-anu)

A nanoGPT-based pipeline for training and evaluating small GPT models on the
ROCStories commonsense story corpus. Extended from
[Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) with modern
LLaMA-style architecture improvements, synthetic data augmentation, and
Hugging Face Hub packaging.

---

## Results Summary

| Task | Model | Params | Val PPL | Notes |
|------|-------|--------|---------|-------|
| T1 Baseline | Vanilla 7L/6H/384D | 31.8M | 22.4 | Plain ROCStories, correct hyperparams |
| T2-A | Vanilla | 31.8M | 21.9 | Ablation baseline |
| T2-B | +RoPE | 31.8M | 23.1 | Marginal gain on short sequences |
| T2-C | +RMSNorm+SwiGLU | 31.8M | 21.9 | Best single mod (tied) |
| T2-D | +QK-Norm | 31.8M | 24.6 | Worst with stable hyperparams (sign-flip finding) |
| T2-E | All Modern | 31.8M | 22.6 | All four flags combined |
| **T3** | **All Modern + synthetic** | **31.8M** | **24.9** ✓ | **Full 19,633-story test** |
| T4 Arena | All Modern 12L/12H/768D | 123.6M | — | Arena competition, 2-stage training |

**Key finding:** Fixing hyperparameters (lr=1e-3, dropout=0.2, β₂=0.99, n_layer=7)
dropped PPL from ~32 to 22.4 — more impactful than any architecture modification.

---

## Architecture Modifications (Task 2)

Four LLaMA-style changes tested independently and combined:

| Flag | What it does |
|------|-------------|
| `use_rope` | Rotary positional encoding instead of learned embeddings |
| `use_rmsnorm` | RMSNorm instead of LayerNorm (no bias) |
| `use_swiglu` | SwiGLU FFN with 8/3× hidden dim instead of GELU |
| `use_qk_norm` | RMSNorm on Q and K before attention scores |

All flags default to `False` (vanilla nanoGPT behaviour).

---

## Repository Layout

```
├── model.py                    # GPT, GPTConfig, generate() — all flags implemented
├── train.py                    # Training loop with W&B, time-based checkpoints
├── eval.py                     # Perplexity evaluation (do not modify for grading)
├── sample.py                   # Single-prompt sampling (do not modify for grading)
├── sample_batch.py             # Batch sampling → JSONL (do not modify for grading)
├── configurator.py             # CLI / config-file overrides
├── hf_load.py                  # HuggingFace Hub upload/download
├── preflight.py                # Pre-training sanity checks
├── config/
│   ├── train_t1_baseline.py    # Task 1 vanilla baseline
│   ├── train_t2_ablation_a.py  # A: Vanilla (ablation control)
│   ├── train_t2_ablation_b.py  # B: +RoPE
│   ├── train_t2_ablation_c.py  # C: +RMSNorm+SwiGLU
│   ├── train_t2_ablation_d.py  # D: +QK-Norm
│   ├── train_t2_ablation_e.py  # E: All Modern
│   ├── train_t3_best.py        # Task 3 submission config
│   ├── train_t3_synthetic.py   # Task 3 with synthetic data
│   ├── train_t4_arena.py       # Task 4 Stage 1 pretraining
│   └── train_t4_finetune.py    # Task 4 Stage 2 fine-tuning
├── data/
│   ├── rocstories/             # ROCStories prepare.py + eval files
│   ├── tinystories/            # TinyStories prepare.py
│   ├── mixed/                  # Mixed format experiments
│   ├── combined/               # ROC + TinyStories for T4 pretraining
│   └── rocstories_synthetic/   # Synthetic data pipeline
└── code_v2.ipynb               # Main notebook (Tasks 1–4)
```

---

## Constraints

| Rule | Detail |
|------|--------|
| Parameter budget | Tasks 1–3 and HuggingFace submission must stay **≤ 32M parameters** |
| Frozen scripts | `eval.py`, `sample.py`, `sample_batch.py` — do not modify |
| No pretrained weights | All parameters trained from scratch |
| T4 Arena | No size limit; do not submit arena checkpoint for T3 grading |

---

## Quick Start

```bash
pip install torch tiktoken datasets huggingface_hub wandb
```

**Prepare data:**
```bash
python data/rocstories/prepare.py
```

**Train Task 1 baseline:**
```bash
python train.py config/train_t1_baseline.py
```

**Evaluate:**
```bash
python eval.py --init_from=resume --out_dir=out-t1-baseline \
    --input_file=data/rocstories/eval_stories.txt
```

**Sample stories:**
```bash
python sample_batch.py --init_from=resume --out_dir=out-t1-baseline \
    --start=FILE:data/rocstories/eval_prompts.txt \
    --batch_prompts=True --max_new_tokens=120
```

---

## Task 3 Training Pipeline

```bash
# 1. Prepare synthetic dataset (requires synthetic JSON)
python data/rocstories_synthetic/prepare.py \
    --json_path /path/to/synthetic_stories_gptoss120b.json

# 2. Train from scratch on synthetic corpus
python train.py config/train_t3_synthetic.py

# 3. Micro fine-tune on pure ROCStories (30 steps)
python train.py config/train_t3_synthetic.py \
    --init_from=resume \
    --dataset=rocstories \
    --max_iters=19780 \
    --learning_rate=1e-4 \
    --always_save_checkpoint=True

# 4. Evaluate on full test set
python eval.py --init_from=resume --out_dir=out-t3-synthetic \
    --input_file=data/rocstories/eval_stories_full.txt \
    --max_paragraphs=-1
```

---

## Task 4 Training Pipeline

```bash
# Stage 1: Pretrain 124M on combined corpus
python data/combined/prepare.py
python train.py config/train_t4_arena.py

# Stage 2: Fine-tune on synthetic ROCStories
cp out-t4-pretrain/ckpt_best.pt out-t4-arena/ckpt.pt
python train.py config/train_t4_finetune.py
```

---

## HuggingFace Submission

```bash
# Upload T3
python hf_load.py upload \
    --local-dir submission_hf \
    --repo-id YOUR_USERNAME/nanoGPT_hw \
    --token YOUR_HF_TOKEN

# Upload T4
python hf_load.py upload \
    --local-dir submission_t4_hf \
    --repo-id YOUR_USERNAME/nanoGPT_hw_t4 \
    --token YOUR_HF_TOKEN
```

Submission folder must contain: `ckpt.pt`, `model.py`, `sample_params.json`.

---

## Hyperparameter Key Finding

| Hyperparameter | GPT-2 default | This work | Effect |
|---|---|---|---|
| Learning rate | 6×10⁻⁴ | **1×10⁻³** | −3 PPL |
| β₂ | 0.95 | **0.99** | Stable on small data |
| Dropout | 0.1 | **0.2** | Prevents memorisation |
| n_layer | 6 | **7** | 28.6 → 22.4 PPL |
| Label smoothing | 0 | 0 (tried 0.1, dropped) | +0.3 PPL when on |

---

## References

- Karpathy, A. [nanoGPT](https://github.com/karpathy/nanoGPT) (2022)
- Mostafazadeh et al. ROCStories corpus. NAACL 2016.
- Su et al. RoFormer / RoPE. arXiv:2104.09864 (2021)
- Zhang & Sennrich. RMSNorm. NeurIPS 2019.
- Shazeer. SwiGLU variants. arXiv:2002.05202 (2020)
- Gemma Team. QK-Norm. arXiv:2403.08295 (2024)
- Eldan & Li. TinyStories. arXiv:2305.07759 (2023)