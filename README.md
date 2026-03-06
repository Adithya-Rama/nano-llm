# nanoGPT — Enhanced LLaMA-Style Story Generator

Modern LLaMA-style transformer for ROCStories story generation, built on [nanoGPT](https://github.com/karpathy/nanoGPT).

## Architecture

**~152M parameter** model with modern enhancements:

| Component | Implementation |
|---|---|
| Normalisation | RMSNorm (pre-norm) |
| Positional encoding | Rotary (RoPE) |
| Feed-forward | SwiGLU (gated activation) |
| Attention stability | **QK-Norm** (Gemma 2, 2024) |
| Attention kernel | Flash Attention via `F.scaled_dot_product_attention` |
| Embeddings | Tied input/output |
| Training loss | Label smoothing (0.1) |
| Generation | Top-k + Top-p + Repetition penalty |

**Config:** 12 layers, 12 heads, 768-dim, block_size=256

## Quick Start

```bash
# 1. Install dependencies
pip install tiktoken datasets huggingface_hub

# 2. Prepare datasets
python data/rocstories/prepare.py         # Required
python data/tinystories/prepare.py        # Optional (extra data)
python data/combined/prepare.py           # Optional (merged)

# 3. Train (ROCStories only — Task 1)
python train.py config/train_rocstories.py

# 4. Train (combined data — best for Task 3)
python train.py config/train_rocstories_combined.py

# 5. Resume after crash
python train.py config/train_rocstories.py --init_from=resume

# 6. Evaluate (perplexity)
python eval.py --init_from=resume --out_dir=out-rocstories \
    --input_file=data/rocstories/eval_stories.txt

# 7. Generate samples
python sample_batch.py --init_from=resume --out_dir=out-rocstories \
    --start="FILE:data/rocstories/eval_prompts.txt" \
    --batch_prompts=True --max_new_tokens=200

# 8. Run 5-way ablation (Task 2)
python train.py config/train_rocstories_baseline.py         # A. Vanilla GPT
python train.py config/train_rocstories_rope_only.py        # B. +RoPE
python train.py config/train_rocstories_rmsnorm_swiglu.py   # C. +RMSNorm+SwiGLU
python train.py config/train_rocstories_qknorm.py           # D. +QK-Norm
python train.py config/train_rocstories.py                  # E. All modern

# 9. Upload to HuggingFace (Task 3)
python hf_load.py upload --local-dir submission_hf \
    --repo-id YOUR_USERNAME/nanoGPT_hw --token YOUR_TOKEN
```

## Colab Training

See `colab_setup.py` for a step-by-step Colab guide. Key features:

- **SIGTERM handler**: Saves checkpoint on Colab preemption
- **Time-based saves**: Every 15 minutes regardless of step count
- **Atomic writes**: No checkpoint corruption from partial writes
- **GradScaler state**: Full resume fidelity for float16 training
- **JSONL logs**: Structured training logs for plotting

## Configuration Files

| Config | Architecture | Purpose |
|---|---|---|
| `train_rocstories.py` | All modern (152M) | Task 1 + Task 3 |
| `train_rocstories_combined.py` | All modern (152M) | Task 3 (best) |
| `train_rocstories_baseline.py` | Vanilla GPT | Task 2 Ablation A |
| `train_rocstories_rope_only.py` | +RoPE only | Task 2 Ablation B |
| `train_rocstories_rmsnorm_swiglu.py` | +RMSNorm+SwiGLU | Task 2 Ablation C |
| `train_rocstories_qknorm.py` | +QK-Norm only | Task 2 Ablation D |

## Sampling Parameters

`sample_params.json`:
```json
{"temperature": 0.75, "top_k": 50, "top_p": 0.9, "repetition_penalty": 1.2}
```

## References

- Karpathy, A. (2023). nanoGPT. GitHub.
- Touvron et al. (2023). LLaMA: Open and Efficient Foundation Language Models.
- Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding.
- Shazeer (2020). GLU Variants Improve Transformer.
- Team et al. (2024). Gemma 2: Improving Open Language Models. (QK-Norm)
- Eldan & Li (2023). TinyStories: How Small Can Language Models Be.
- Mostafazadeh et al. (2016). ROCStories Corpus.
