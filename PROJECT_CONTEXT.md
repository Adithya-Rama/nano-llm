Here's the complete context:

---

## COMP4680/8650 NanoGPT ROCStories — Complete Context

### Person
Adithya Rama, ANU Master of Computing (Advanced), R&D ML focus. Repo at `/content/drive/MyDrive/COMP8650/Assgn-1/nano-llm/code-vfinal/` in Google Colab. wandb project: `rocstories-nanogpt` (adithyaiyer-anu).

---

### Assignment Structure
Train nanoGPT models for 5-sentence ROCStories story completion. Four tasks: T1 baseline, T2 ablations, T3 best ≤32M checkpoint (HuggingFace submission), T4 arena (124M, no size limit).

---

### Architecture
LLaMA-style nanoGPT with `use_rope=True, use_rmsnorm=True, use_swiglu=True, use_qk_norm=True`. SwiGLU hidden dim = int(8/3 × n_embd) rounded to 64.

---

### T1/T2 — COMPLETED
All configs: 7L/6H/384D, ~31.8M params, 5K steps. T2-E (all modern flags) = best ablation.

---

### T3 — COMPLETED, READY TO SUBMIT

**Final checkpoint:** `out-t3-synthetic/ckpt_best.pt` (step 20001, val PPL **19.0**)
- `ckpt_final.pt` also saved at step 20000
- Architecture: 7L/6H/384D, 31.71M params (≤32M ✓)
- Training run name: `t3-synthetic-v2-31.8M-100k-diverse`

**Dataset used:** `data/rocstories_synthetic/` — built from full 408K merged synthetic JSON at `/content/drive/MyDrive/COMP8650/Assgn-1/nano-llm/synthetic-data/synthetic_stories_gptoss120b.json`
- 408,722 stories passed quality filter
- 28.95M synthetic tokens + 3.70M original ROCStories = **32.65M train tokens**
- Val: 0.41M tokens (original ROCStories only)

**How to build the dataset correctly:**
```bash
python data/rocstories_synthetic/prepare.py \
    --json_path /content/drive/MyDrive/COMP8650/Assgn-1/nano-llm/synthetic-data/synthetic_stories_gptoss120b.json
```
This produces ~32.65M tokens. If you pass the wrong path (the file in `code-vfinal/`) you only get 23.75M tokens — that's wrong.

**Config:** `config/train_t3_synthetic.py`
- `init_from = 'scratch'`
- `out_dir = 'out-t3-synthetic'`
- `max_iters = 20000, lr_decay_iters = 20000`
- `learning_rate = 1e-3, min_lr = 1e-4`
- `batch_size = 64, block_size = 256, gradient_accumulation_steps = 1`
- `always_save_checkpoint = False` (saves only on improvement)
- `wandb_run_name = 't3-synthetic-v2-31.8M-100k-diverse'`

**Sample params** (`out-t3-synthetic/sample_params.json`):
```json
{"temperature": 0.78, "top_k": 50, "top_p": 0.92, "repetition_penalty": 1.03}
```

**Training curve summary:**
| Step | Val PPL |
|------|---------|
| 250 | 87.4 |
| 1000 | 38.6 |
| 5000 | 25.3 |
| 10000 | 22.0 |
| 15000 | 20.0 |
| 17750 | 19.3 |
| 19750 | 19.0 (best) |
| 20000 | 19.17 (final, not best) |

Best checkpoint is `ckpt_best.pt` at step 19750/20001, PPL 19.0.

**Story quality:**
- Emily (umbrella) — coherent 5-sentence arc ✓
- Tom (cooking dinner) — coherent ✓
- Sofia (Rio conference presentation) — coherent, professional ✓ (100K diverse stories fixed this)
- Sarah (lost keys) — location confusion bug persists (architectural ceiling at 31.8M)
- Danish/Anoop (OOD names) — completely broken, unfixable at this size

**Pending action:** Submit to HuggingFace — run Cell 34 (comparison table) then Cell 36 (upload). Set `HF_USERNAME` in Cell 36.

---

### T4 — COMPLETED

**Final checkpoint:** `out-t4-arena/ckpt.pt` (step 55001) and `ckpt_final.pt`
- Architecture: 12L/12H/768D, 123.59M params
- Training run name: `t4-finetune-124M-synthetic-gptoss120b`

**Two-stage training:**

**Stage 1 (pretrain):** 25K steps on combined corpus (ROCStories 5x + TinyStories ~200M tokens). Config: `config/train_t4_arena.py`. Output: `out-t4-pretrain/ckpt_best.pt`. Stage 1 best val loss = 0.5055 (PPL 1.7 — this is stored in `best_val_loss` in the checkpoint but reflects Stage 1 metric on combined val, NOT Stage 2 metric).

**Stage 2 (fine-tune):** Extended to 55K steps total (25K Stage1 + ~30K fine-tune). Config: `config/train_t4_finetune.py`. Trained on `data/rocstories_synthetic/` (32.65M tokens from full 408K merged JSON). Resumed from `out-t4-arena/ckpt.pt`.

**Stage 2 config key settings:**
- `init_from = 'resume'`, `out_dir = 'out-t4-arena'`
- `max_iters = 55000, lr_decay_iters = 55000`
- `learning_rate = 1e-4, min_lr = 1e-5`
- `batch_size = 16, gradient_accumulation_steps = 8, block_size = 512`
- `always_save_checkpoint = True` (critical — val loss never beats Stage 1 best)
- `warmup_iters = 100`

**Stage 2 training curve (val loss rising = intentional TinyStories suppression):**
| Step | Val PPL |
|------|---------|
| 30200 | 1.82 |
| 33000 | 2.50 |
| 37000 | 3.30 |
| 42000 | 4.00 |
| 48000 | 4.59 |
| 55000 | 5.05 |

Val loss rising is NOT overfitting. It reflects the model shifting away from TinyStories distribution onto ROCStories distribution. The `best_val_loss = 0.5055` stored in the checkpoint is the Stage 1 metric and is meaningless for evaluating Stage 2 quality. Rising val loss during Stage 2 is expected and desired.

**MFU:** 43.7% (correct — using batch=16, accum=8, block=512 = 65,536 tokens/step). Previous broken Stage 2 run had 6-7% MFU because wrong batch config.

**Sample params** (`out-t4-arena/sample_params.json`):
```json
{"temperature": 0.80, "top_k": 50, "top_p": 0.92, "repetition_penalty": 1.03}
```
**No `stop_token`** — do not include it.

**Story quality (Cell 49 evaluation — the correct cell):**
All 8 test prompts produce 5/5 sentences with coherent narrative arcs:
- Emily: borrows umbrella from neighbor, catches bus, arrives relieved ✓
- Tom: pasta dish, out of parmesan, substituted mozzarella, turned out well ✓
- Lily: jogging routine, improvement, ankle injury, recovered ✓
- Mark: borrowed bike, rode to park, rainstorm, returned apologizing ✓
- Anna: planted seeds, watered daily, shoots appeared, harvested tomatoes ✓
- Sarah: lost wallet, cancelled cards, got replacement ✓
- Jake: Father's Day breakfast surprise, went shopping, dad touched ✓
- Old dog: reunited with owner ✓ (slight incoherence in middle sentences)

**T4 eval PPL (Cell 49 reports): 26.39** — note this is measured against ROCStories val, not a raw model PPL.

---

### Critical Evaluation Warning

**Cell 48 = BROKEN. Cell 49 = CORRECT.**

Cell 48 strips S1 from output, uses hardcoded broken params with `stop_token=50256`, producing 4-sentence outputs with missing first sentence. This is a TinyStories residue artifact in the old cell.

Cell 49 uses `complete_story()` function which:
1. Generates with EOT token split (no stop_token)
2. Prepends prompt as S1 if missing
3. Deduplicates consecutive identical sentences
4. Retries with higher temperature if fewer than 5 sentences
5. Returns first 5 sentences

The coherence in Cell 49 outputs comes from model weights, not post-processing. Post-processing only fixes formatting edge cases.

---

### Synthetic Data Pipeline

**Merged file location:** `/content/drive/MyDrive/COMP8650/Assgn-1/nano-llm/synthetic-data/synthetic_stories_gptoss120b.json`
- 408,724 total stories
- 300K from llama3:8b via local Ollama (RTX 5070)
- 100K from Groq llama-3.3-70b (diverse cultural categories: South Asian, East Asian, African, Latin American, Middle Eastern, diverse workplace — at 1.5x weight)
- 28 categories total

**Common mistake:** If you pass the wrong JSON path to `prepare.py` (the one inside `code-vfinal/` instead of `synthetic-data/`), you only get 308K stories and 23.75M tokens instead of 408K stories and 32.65M tokens.

---

### Key Bugs Fixed During This Session
1. Wrong JSON path to `prepare.py` — was getting 308K stories instead of 408K
2. T3 previously trained on 23.75M tokens — retrained on 32.65M with full 408K corpus
3. T4 Stage 2 previously ran only 5K effective steps — extended to 30K effective steps
4. `always_save_checkpoint = True` required in T4 finetune because val loss never beats Stage 1 best
5. Cell 48 broken evaluation — must use Cell 49 `complete_story()` function

---

### Architectural Ceiling (Cannot Fix)
- 31.8M params cannot reliably track 4-step causal chains
- OOD names (Anoop, Danish, non-Western) → falls back to training distribution
- Location tracking bugs (Sarah apartment) persist at 31.8M
- Threshold for reliable narrative coherence: ~350M+ params
- T3 PPL measures token fluency, not narrative coherence — explains gap

---

### Pending Actions
1. **IMMEDIATE:** Submit T3 to HuggingFace — Cell 34 (comparison) → Cell 36 (upload). Set `HF_USERNAME` in Cell 36. Checkpoint is `out-t3-synthetic/ckpt_best.pt`, step 19750, PPL 19.0.
2. **VERIFY T4:** Run Cell 49 `complete_story()` on a few more prompts including Sofia (professional conference prompt) to confirm quality. Then confirm `out-t4-arena/sample_params.json` has the correct params.
3. **OPTIONAL SANITY CHECK:** Run raw `sample.py` on T4 to verify coherence without post-processing:
```bash
python sample.py \
    --out_dir=out-t4-arena \
    --start="Emily forgot her umbrella before leaving for work." \
    --max_new_tokens=110 \
    --temperature=0.80 \
    --top_k=50 \
    --top_p=0.92 \
    --repetition_penalty=1.03 \
    --num_samples=3
```

---

### File Locations
- **Repo root:** `/content/drive/MyDrive/COMP8650/Assgn-1/nano-llm/code-vfinal/`
- **T3 checkpoint:** `out-t3-synthetic/ckpt_best.pt` (step 19750/20001, PPL 19.0)
- **T3 final:** `out-t3-synthetic/ckpt_final.pt`
- **T4 checkpoint:** `out-t4-arena/ckpt.pt` (step 55001)
- **T4 final:** `out-t4-arena/ckpt_final.pt`
- **Merged synthetic JSON:** `/content/drive/MyDrive/COMP8650/Assgn-1/nano-llm/synthetic-data/synthetic_stories_gptoss120b.json`
- **T3 config:** `config/train_t3_synthetic.py`
- **T4 finetune config:** `config/train_t4_finetune.py`
- **Synthetic prepare:** `data/rocstories_synthetic/prepare.py`
- **Notebook:** `code_v2.ipynb` — T3 retrain = Cell 25, T4 Stage 2 = Cell 43, T4 correct eval = Cell 49, T4 broken eval = Cell 48 (do not use)