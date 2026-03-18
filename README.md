# nanoGPT — ROCStories Mini-Project (COMP4680/8650)

This repo contains a complete, assignment-compliant nanoGPT pipeline for **ROCStories story generation**.

## Constraints (important)

- **Tasks 1–3 must use ≤ 32M parameters** (including the model uploaded to HuggingFace).
- **Task 1 baseline must follow the official nanoGPT “baby GPT” config**:
  `n_layer=6`, `n_head=6`, `n_embd=384` (≈ **30.2M** params).
- **Task 4 (arena competition)** may use larger models (optional).

## What to run

The recommended entry point is the notebook:

- `code_v2.ipynb`: clean, cell-by-cell workflow for Task 1 → Task 2 → Task 3 (+ optional Task 4)

## Model variants in this repo

### Task 1 baseline (vanilla nanoGPT, ≤32M)

- **Config:** `config/train_t1_baseline.py`
- **Model size:** ~30.2M params (6L/6H/384D)
- **Architecture:** learned positional embeddings, LayerNorm, GELU MLP

### Task 2 ablations (≤32M, controlled comparison)

All at 6L/6H/384D with identical training hyperparameters:

| Config | Change | Output dir |
|---|---|---|
| `config/train_t2_ablation_a.py` | A. Vanilla reference | `out-t2-vanilla` |
| `config/train_t2_ablation_b.py` | B. +RoPE | `out-t2-rope` |
| `config/train_t2_ablation_c.py` | C. +RMSNorm + SwiGLU | `out-t2-ffn` |
| `config/train_t2_ablation_d.py` | D. +QK-Norm (novel) | `out-t2-qknorm` |
| `config/train_t2_ablation_e.py` | E. All modern combined | `out-t2-all-modern` |

### Task 3 best submission (≤32M)

- **Config:** `config/train_t3_best.py`
- **Model size:** ~30.1M params (6L/6H/384D, all-modern)
- **Dataset:** `data/mixed/` (mixed continuation + instruction + structured)
- **Purpose:** best checkpoint to upload to HuggingFace for grading

### Task 4 arena model (optional, >32M allowed)

- **Config:** `config/train_t4_arena.py`
- **Model size:** ~152M params (12L/12H/768D)
- **Dataset:** `data/combined/` (ROCStories + TinyStories)
- **Warning:** do **not** upload this model for Task 3 grading.

## Quick start (CLI)

```bash
# 1) Install dependencies
pip install tiktoken datasets huggingface_hub

# 2) Prepare ROCStories (required)
python data/rocstories/prepare.py

# 3) Train Task 1 baseline (≤32M)
python train.py config/train_t1_baseline.py

# 4) Evaluate perplexity
python eval.py --init_from=resume --out_dir=out-t1-baseline \
    --input_file=data/rocstories/eval_stories.txt

# 5) Task 2 datasets (optional but recommended for experiments)
python data/rocstories/prepare.py --structured
python data/tinystories/prepare.py

# 6) Build mixed instruction dataset (Task 2/3)
python data/mixed/prepare.py

# 7) Train Task 3 best model (≤32M) on mixed dataset
python train.py config/train_t3_best.py

# 8) Evaluate Task 3 perplexity on the public test set
python eval.py --init_from=resume --out_dir=out-t3-best \
    --input_file=data/rocstories/eval_stories.txt

# 9) Upload to HuggingFace (Task 3)
python hf_load.py upload --local-dir submission_hf \
    --repo-id YOUR_USERNAME/nanoGPT_hw --token YOUR_TOKEN
```

## Colab

Use `code_v2.ipynb` as the clean, report-ready Colab notebook.

## Sampling parameters

`sample_params.json` (used by `sample_batch.py` if you pass it through your wrapper):

```json
{"temperature": 0.75, "top_k": 50, "top_p": 0.9, "repetition_penalty": 1.2}
```

## References

- Karpathy, A. nanoGPT. `https://github.com/karpathy/nanoGPT`
- Mostafazadeh et al. (2016). ROCStories Corpus.
- Su et al. (2022). RoFormer / RoPE. arXiv:2104.09864.
- Zhang & Sennrich (2019). RMSNorm. arXiv:1910.07467.
- Shazeer (2020). GLU variants / SwiGLU. arXiv:2002.05202.
- Henry et al. (2020). Query-Key Normalization. arXiv:2010.04245.

