# NanoGPT ROCStories — LLM context file

**Purpose:** Single document to onboard an AI assistant or model to this repository: goals, layout, conventions, and changes made during development (COMP4680/8650, ANU, March 2026).

**Student:** Adithya Rama  
**Base:** [Karpathy nanoGPT](https://github.com/karpathy/nanoGPT) with substantial extensions for the assignment.

---

## 1. Assignment goals

| Task | What | Constraint |
|------|------|------------|
| **T1** | Vanilla “baby GPT” baseline on ROCStories | ≤32M params |
| **T2** | 5 architecture ablations (A–E) on ROCStories | Same; isolate RoPE, RMSNorm+SwiGLU, QK-Norm |
| **T3** | Best submission checkpoint | ≤32M params; **target val PPL &lt; 25** on professor evaluation (instruction: use mixed data + TinyStories) |
| **T4** | Arena / optional larger model | No param cap; combined corpus |

**Metric:** Validation perplexity from training logs (`train_log.jsonl`); report **best** val PPL (early stopping), not necessarily final step — models often overfit after ~1250–2250 steps on small ROC-only data.

---

## 2. Authoritative configs (use these)

| Config | Role |
|--------|------|
| `config/train_t1_baseline.py` | Task 1 baseline |
| `config/train_t2_ablation_{a,b,c,d,e}.py` | Task 2 A–E |
| `config/train_t3_best.py` | Task 3 submission training |
| `config/train_t4_arena.py` | Task 4 (different scale; not the “baby-GPT” recipe) |

**Legacy / unused:** Original `train_rocstories_*.py`, Shakespeare, GPT-2 eval configs were either marked deprecated, moved under `config/unused/`, and/or listed in `.gitignore`. Do not treat them as the assignment pipeline.

---

## 3. Model code (`model.py`) — high level

- Extended GPT with optional: **RoPE**, **RMSNorm**, **SwiGLU** FFN (hidden dim ≈ `8/3 * n_embd`, rounded to multiple of 64 for param budget), **QK-Norm**.
- SwiGLU and width choices keep Tasks 1–3 near **~31.7–31.8M** params with `n_layer=7`, `n_head=6`, `n_embd=384`.
- RoPE cache `max_seq_len` set to at least `max(block_size, 2048)`.
- Sampling: top-p uses correct cumulative mask `(cumulative_probs - probs_sorted) >= top_p`.

---

## 4. Training recipe (Tasks 1–3 “baby GPT” scale)

Aligned with Karpathy `train_shakespeare_char` style, **not** GPT-2 pretraining defaults:

- `learning_rate = 1e-3`, `min_lr = 1e-4`
- `dropout`: **0.2** on T1/T2; **0.15** on T3 (per `CURSOR_FINAL_PROMPT.md` / memorisation tradeoff)
- `beta2 = 0.99`, `beta1 = 0.9`
- `label_smoothing = 0.0` (template had none; smoothing inflated val CE)
- `batch_size = 64`, `gradient_accumulation_steps = 1` (effective 16,384 tokens/step at `block_size=256`)
- `always_save_checkpoint = False` — only save on improvement unless periodic resume is needed

**Task 3 current schedule (`train_t3_best.py`):** `max_iters = 8000`, `lr_decay_iters = 8000`, `warmup_iters = 150`.

**Task 1 / T2:** `max_iters = 5000`, `lr_decay_iters = 5000`, `warmup_iters = 100` (stop before severe overfit on pure ROCStories).

---

## 5. Checkpointing (`train.py`)

- **Problem fixed:** `always_save_checkpoint=True` was overwriting good checkpoints with worse late-training states.
- **Behaviour:** On new best validation loss: `save_checkpoint(tag='best')` writes `ckpt_best.pt` and (implementation) also refreshes `ckpt.pt` for resume. If `always_save_checkpoint` and not a new best, only `ckpt.pt` updates.
- **Logging:** `train_log.jsonl` via `log_training_step`; prints include best val PPL with overflow guard if `best_val_loss` is still a huge sentinel.
- **HuggingFace / submission:** Package **`ckpt_best.pt`** (copy/rename to `ckpt.pt` in submission folder), not blindly the last `ckpt.pt`.

---

## 6. Data pipelines

### ROCStories (`data/rocstories/prepare.py`)

- Downloads from HuggingFace (`mintujupally/ROCStories`), tokenises with **tiktoken GPT-2 BPE**, stores `uint16` in `.bin`.
- **`train.bin`:** all downloaded stories (plain or structured via flags).
- **`val.bin`:** **only** stories from `data/rocstories/eval_stories.txt` (professor holdout — **no overlap** with train). Older approach that sampled “monitor” val from the train pool caused **optimistic / contaminated** val PPL.

### Mixed (Tasks 2/3) (`data/mixed/prepare.py`)

- Combines ROCStories-derived formats + optional **TinyStories** prefix.
- **Important:** Validation must be **reserved before** splitting formats so train/val don’t leak.
- TinyStories cap was raised (e.g. toward **100M** tokens) so T3 isn’t accidentally trained on ROC-only scale.

### TinyStories (`data/tinystories/prepare.py`)

- Prepare `train.bin` / `val.bin` for optional upsampling into mixed.

### Combined / Task 4 (`data/combined/prepare.py`)

- Larger arena corpus (e.g. streaming, multiple sources, filters) — see script docstring and flags.

**Ignored in git:** `*.bin`, `*.pt`, etc. (see `.gitignore`).

---

## 7. Evaluation & notebooks

- **`eval.py`:** Used for held-out evaluation; paths depend on `data_dir` and checkpoint.
- **`code_v2.ipynb`:** Main Colab-oriented workflow: §0 setup, dataset size verification, Task 1–4 cells, best-PPL summariser from `train_log.jsonl`, T4 evaluation plots, final summary figures, HuggingFace upload using best checkpoint.
- **`preflight.py`:** Local sanity checks for configs, paths, and (when updated) ROCStories split expectations (e.g. `eval_stories.txt` usage).

---

## 8. Operational run order (typical)

1. Prepare ROCStories → optional structured pass → TinyStories if needed → `data/mixed/prepare.py --with_tinystories`.
2. Verify `data/mixed/train.bin` token count (expect **well above ~10M** if TinyStories merged).
3. `python train.py config/train_t1_baseline.py` then T2 configs, then T3.
4. For reporting: read **best** `val_loss` from `train_log.jsonl`, not only the last line.

---

## 9. Known pitfalls (forensics from logs)

1. **Checkpoint quality:** Submit / report **best** checkpoint, not the final step.
2. **T3 without TinyStories:** Mixed dataset size ≈ ROC-only → same convergence speed as T1; need TinyStories (or equivalent) in the mix for Task 3 PPL target.
3. **Small eval set:** `eval_stories.txt` is tiny; noisy PPL — assignment-scale test is larger; trust smoothed val from full `val.bin` training logs for trends.
4. **PPL &lt; 25:** Framed for **T3 submission**, not necessarily T1/T2 on pure ROCStories (often high-20s to low-30s is expected there).

---

## 10. Repo hygiene

- **`.gitignore`:** Excludes checkpoints, bins, `notes/`, `.cursor/`, `results/`, and optionally `config/unused`, `data/unused`, and duplicate legacy config paths — adjust if you need those tracked.
- **`bench.py`:** Benchmarking utility (original nanoGPT).

---

## 11. Quick file map

| Path | Role |
|------|------|
| `train.py` | Training loop, logging, checkpointing |
| `model.py` | GPT + RoPE/RMSNorm/SwiGLU/QK-Norm |
| `config/train_t*.py` | Assignment entry points |
| `data/*/prepare.py` | Dataset binaries |
| `eval.py` | Evaluation |
| `preflight.py` | Config/dataset checks |
| `code_v2.ipynb` | Full experiment notebook |

---

*Generated for context injection into coding assistants. Update this file when configs or data contracts change.*
