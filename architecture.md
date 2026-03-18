# NanoGPT Architecture & Experiments (COMP4680/8650)

This document is the definitive architecture reference for the ROCStories mini-project.

---

## Parameter Constraint

Per the professor's revised instructions:

> "The above 32M constraint applies to Task 1, Task 2, **including the model you submit to HuggingFace**."

**All graded work (Tasks 1–3) must use ≤ 32M parameters.**
The 152M arena model is kept for **Task 4 (competition only)** where size is unconstrained.

---

## 1. Model Configurations

### 1.1 Task 1 — Official Baseline

**Config:** `config/train_t1_baseline.py`  
**Output dir:** `out-t1-baseline/`

Follows the official nanoGPT "baby GPT" specification (Karpathy, 2022):

| Parameter | Value | Notes |
|---|---|---|
| `n_layer` | 6 | as required by prof |
| `n_head` | 6 | as required by prof |
| `n_embd` | 384 | as required by prof |
| `block_size` | 256 | context window |
| `vocab_size` | 50,304 | GPT-2 BPE, padded to mult of 64 |
| **Total params** | **≈ 30.2M** | ≤ 32M ✓ |
| `dropout` | 0.1 | |
| `bias` | False | |

**Architecture:** vanilla nanoGPT — learned absolute positional embeddings, LayerNorm, GELU MLP.
No modern modifications. This is the required baseline.

**Training settings:**
- `learning_rate` = 6e-4  (standard for ~30M GPT; Kaplan et al., 2020)
- `max_iters` = 10,000  (~88 passes through ROCStories)
- `batch_size` = 32 × `gradient_accumulation_steps` = 4  → 32,768 tokens/step
- Cosine LR decay: 6e-4 → 6e-5
- `label_smoothing` = 0.1

**Estimated training time (A100):** ~8–10 min

---

### 1.2 Task 2 — Architecture Ablation Study

**All five configs:** `config/train_t2_ablation_{a..e}.py`  
**Output dirs:** `out-t2-{vanilla, rope, ffn, qknorm, all-modern}/`

All ablations use **identical hyperparameters** and the **same 6L/6H/384D scale** so that PPL differences are caused only by the architectural change. Total params remain ≈ 30.2M (≤ 32M ✓) in all cases.

| Config | Change from vanilla | Param count |
|---|---|---|
| A — Vanilla | none (reference) | ≈ 30.2M |
| B — +RoPE | replace learned PE with RoPE | ≈ 30.1M (−0.1M, no pos-emb) |
| C — +RMSNorm+SwiGLU | replace LayerNorm/GELU | ≈ 30.2M (SwiGLU 8/3 scaling) |
| D — +QK-Norm ★ | add Q/K normalisation | ≈ 30.2M (+768 negligible params) |
| E — All Modern | B + C + D combined | ≈ 30.1M |

★ **Novel contribution:** QK-Norm's effect on sub-32M narrative-generation models has not been previously studied.

**Scientific narrative (Task 2 report):** we isolate three orthogonal modernisations — (i) positional encoding (RoPE), (ii) FFN/normalisation (RMSNorm+SwiGLU), and (iii) attention stability (QK-Norm) — and measure their individual and combined contributions to perplexity on short story text. QK-Norm has recently been adopted by Gemma 2 (2024) and Cohere Command-R but its impact on small models is unstudied.

---

### 1.3 Task 3 — Best Submission Model (≤ 32M, for HuggingFace)

**Config:** `config/train_t3_best.py`  
**Output dir:** `out-t3-best/`

Uses the best architecture from Task 2 (E — All Modern) plus a mixed instruction/continuation training corpus.

| Parameter | Value |
|---|---|
| `n_layer` | 6 |
| `n_head` | 6 |
| `n_embd` | 384 |
| `use_rope` | True |
| `use_rmsnorm` | True |
| `use_swiglu` | True |
| `use_qk_norm` | True |
| **Total params** | **≈ 30.1M** (≤ 32M ✓) |
| `dataset` | `mixed` |
| `max_iters` | 15,000 |
| `dropout` | 0.05 |

**Dataset:** `data/mixed/` — see §3.2 below.

---

### 1.4 Task 4 — Arena Model (optional, > 32M allowed)

**Config:** `config/train_t4_arena.py`  
**Output dir:** `out-t4-arena/`

> **Warning:** do NOT upload this for Task 3 grading. For the arena competition only.

| Parameter | Value |
|---|---|
| `n_layer` | 12 |
| `n_head` | 12 |
| `n_embd` | 768 |
| **Total params** | **≈ 152M** |
| `dataset` | `combined` |
| `max_iters` | 20,000 |

---

## 2. The Forward Pass (All-Modern 30M Model)

This describes the journey of one batch through the Task 3 model. The Task 1 baseline is the same but uses learned PE, LayerNorm, and GELU instead.

### Step 1 — Input & Embedding

A batch of token indices `idx` of shape `(B, T)` enters `wte` (50,304 × 384), producing dense vectors. Because we use RoPE, **no positional embedding is added here**.

### Step 2 — Transformer Blocks (×6)

Each block runs two sub-layers on the residual stream:

**A. Attention sub-layer**

1. **Pre-Norm (RMSNorm)** on the residual stream.
2. **QKV projection** → Q, K, V tensors shaped `(B, T, n_head, head_dim)` where `head_dim = 64`.
3. **QK-Norm** — each Q and K head is independently RMSNorm'd before the dot product. This prevents attention logit explosion and is the novel architectural contribution.
4. **RoPE injection** — Q and K are rotated in complex-valued space to encode relative position (Su et al., 2022).
5. **Flash Attention** — `softmax((Q·Kᵀ)/√head_dim)·V` with causal mask; O(T) memory.
6. **Output projection** → back to 384-dim, added to residual (skip connection #1).

**B. Feed-forward sub-layer (SwiGLU)**

1. **Pre-Norm (RMSNorm)** on the updated residual.
2. Two parallel projections: `gate(x)` and `up(x)`, each `384 → 1,024` (`8/3 × 384` keeps param count equal to standard 4× MLP).
3. `output = down(SiLU(gate(x)) * up(x))`, projection back to 384.
4. Added to residual (skip connection #2).

### Step 3 — Output & Loss

1. **Final RMSNorm** on the last residual.
2. **LM head** (`384 → 50,304`), weight-tied to `wte`.
3. **Label-smoothed cross-entropy** (smoothing = 0.1) during training.
4. **Generation sampling** — repetition penalty (1.2) → temperature (0.75) → top-k (50) / top-p (0.9) → multinomial sample.

---

## 3. Dataset Design

### 3.1 Task 1 / Ablation Dataset (plain ROCStories)

`data/rocstories/prepare.py` downloads `mintujupally/ROCStories` from HuggingFace and produces:

- **Tokeniser:** tiktoken GPT-2 BPE, vocab = 50,257 (stored as uint16)
- **Separator:** `<|endoftext|>` (id 50256) between stories
- **Split:** 90% train (≈ 3.7M tokens) / 10% val (≈ 410K tokens), seed = 42
- **Format:** plain text — `[story]<|endoftext|>[story]<|endoftext|>…`

### 3.2 Mixed Instruction + Continuation Dataset (Task 3 experiment)

`data/mixed/prepare.py` creates a training corpus in three interleaved formats:

| Format | Share | Template |
|---|---|---|
| A — Plain continuation | 55% | `[raw story text]` |
| B — Instruction-prefixed | 30% | `Write a short story about: [first 6 words].\n[story]` |
| C — Structured XML | 15% | `<story><s1>…</s1>…<s5>…</s5></story>` |

The **validation split is plain ROCStories only** (same as Task 1), so PPL numbers are directly comparable across all experiments.

**Motivation:** InstructGPT (Ouyang et al., 2022) showed that mixed instruction/continuation training lets a model serve both prompt modes without hurting continuation fluency. We test this hypothesis at 30M scale.

### 3.3 Combined Dataset (Task 4 arena)

`data/combined/prepare.py` concatenates ROCStories train + TinyStories (Eldan & Li, 2023) into a single 110M-token training set, giving the 152M arena model broader narrative exposure.

---

## 4. Training Infrastructure

### 4.1 Colab-Resilient Checkpointing

`train.py` has three layers of protection against Colab preemption:

```
Layer 1: Step-based saves   — every eval_interval (250 steps)
Layer 2: Time-based saves   — every 15 minutes regardless of step count
Layer 3: SIGTERM handler    — saves immediately on Colab preemption signal
```

Checkpoints are written **atomically** (`torch.save` to `.tmp`, then `os.replace`) so a partial write never corrupts the checkpoint.

The checkpoint dict includes `scaler.state_dict()` for full bf16/fp16 resume fidelity.

### 4.2 JSONL Training Logs

Every training run appends one JSON line per `log_interval` steps to `{out_dir}/train_log.jsonl`:

```json
{"step": 500, "train_loss": 2.70, "val_loss": 2.80, "lr": 5.8e-4, "mfu": 38.3, "dt_ms": 257}
```

These are read by the analysis cell in `code_v2.ipynb` to plot learning curves and build the ablation bar chart.

---

## 5. How Each Task is Satisfied

### Task 1 (7 marks)

| Requirement | Implementation |
|---|---|
| **(i) Data processing** | `prepare.py`: HuggingFace download → tiktoken GPT-2 BPE → `<\|endoftext\|>` separator → uint16 binary |
| **(ii) Training** | `config/train_t1_baseline.py`: official 6/6/384 config, all hyperparameters documented, Colab-resilient |
| **(iii) Evaluation** | `eval.py` for PPL, `sample_batch.py` for qualitative samples; learning curves from JSONL logs |

### Task 2 (8 marks)

| Criterion | Strategy |
|---|---|
| **Novelty (3 marks)** | QK-Norm at sub-32M scale is unstudied. Instruction-mixed training at this scale is also novel. Goes well beyond "switching architecture". |
| **Comprehensive trials (3 marks)** | 5-way architecture ablation (A–E) + instruction-mixing experiment. Each has hypothesis, PPL result, and qualitative comparison. |
| **Writing clarity (2 marks)** | Tables, learning-curve overlay, bar chart, lexical-diversity metrics, all generated by the analysis cell. |

### Task 3 (15 marks)

| Component | Strategy |
|---|---|
| **PPL** | All-modern 30M + mixed instruction data + 15K steps with cosine decay |
| **Qwen quality** | Repetition penalty (1.2) prevents loops; temperature 0.75, top-p 0.9 for coherent sampling |
| **Submission** | `ckpt.pt` + `model.py` + `sample_params.json` uploaded by §3.2 of `code_v2.ipynb` |

### Task 4 (2 bonus marks)

| Strategy | Details |
|---|---|
| **Model** | 152M all-modern on combined ROCStories+TinyStories |
| **Sampling** | Temperature 0.85, top-k 50, top-p 0.9, rep-penalty 1.3 (more creative for human judging) |

---

## 6. Execution Order

Use `code_v2.ipynb` (recommended). CLI equivalent:

```bash
# Task 1
python data/rocstories/prepare.py
python train.py config/train_t1_baseline.py
python eval.py --init_from=resume --out_dir=out-t1-baseline \
    --input_file=data/rocstories/eval_stories.txt

# Task 2 — ablations
for cfg in a b c d e; do
    python train.py config/train_t2_ablation_${cfg}.py
done

# Task 2 — instruction experiment / Task 3 model
python data/mixed/prepare.py
python train.py config/train_t3_best.py

# Task 3 — upload
python hf_load.py upload --local-dir submission_hf \
    --repo-id YOUR_USERNAME/nanoGPT_hw --token YOUR_TOKEN

# Task 4 (optional arena)
python data/combined/prepare.py
python train.py config/train_t4_arena.py
```

---

## 7. References

- Karpathy, A. (2022). nanoGPT. https://github.com/karpathy/nanoGPT
- Mostafazadeh et al. (2016). ROCStories Corpus. *NAACL*.
- Kaplan et al. (2020). Scaling Laws for Neural Language Models. arXiv:2001.08361.
- Su et al. (2022). RoFormer: Enhanced Transformer with Rotary Position Embedding. arXiv:2104.09864.
- Zhang & Sennrich (2019). Root Mean Square Layer Normalization. arXiv:1910.07467.
- Shazeer (2020). GLU Variants Improve Transformer. arXiv:2002.05202.
- Henry et al. (2020). Query-Key Normalization for Transformers. arXiv:2010.04245.
- Google (2024). Gemma 2: Improving Open Language Models at a Practical Size. arXiv:2408.00118.
- Ouyang et al. (2022). Training Language Models to Follow Instructions. arXiv:2203.02155.
- Eldan & Li (2023). TinyStories. arXiv:2305.07759.
