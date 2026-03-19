"""
Pre-flight check — run this ONCE before starting any training in Colab.
Catches every known failure mode without downloading data or using the GPU.

Usage:
    python preflight.py

All checks either print  ✓  ⚠  or  ✗.
Script exits 1 if any ✗ is found so you cannot proceed past a bad state.
Warnings (⚠) are advisory — training will still work but you should know.
"""

import os, sys, math, json, importlib, traceback
PASS, WARN, FAIL = "✓", "⚠", "✗"
results = []          # (icon, label, message)
_REPO   = os.path.dirname(os.path.abspath(__file__))


# ─── helpers ─────────────────────────────────────────────────────────────────

def check(label, fn):
    """Run fn(). PASS on success, FAIL on any exception / AssertionError."""
    try:
        msg = fn()
        results.append((PASS, label, msg or ""))
        print(f"  {PASS}  {label}{(' — ' + msg) if msg else ''}")
    except AssertionError as e:
        results.append((FAIL, label, str(e)))
        print(f"  {FAIL}  {label} — {e}")
    except Exception as e:
        results.append((FAIL, label, f"{type(e).__name__}: {e}"))
        print(f"  {FAIL}  {label} — {type(e).__name__}: {e}")


def warn(label, fn):
    """Run fn(). WARN (advisory) on failure — does NOT block training."""
    try:
        msg = fn()
        results.append((PASS, label, msg or ""))
        print(f"  {PASS}  {label}{(' — ' + msg) if msg else ''}")
    except AssertionError as e:
        results.append((WARN, label, str(e)))
        print(f"  {WARN}  {label} — {e}")
    except Exception as e:
        results.append((WARN, label, f"{type(e).__name__}: {e}"))
        print(f"  {WARN}  {label} — {type(e).__name__}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. PACKAGES
# ══════════════════════════════════════════════════════════════════════════════
print("\n── 1. Packages ──────────────────────────────────────────────────────")

def chk_torch():
    import torch
    v = torch.__version__
    major, minor = (int(x) for x in v.split(".")[:2])
    assert (major, minor) >= (2, 0), f"PyTorch {v} — need ≥2.0 for Flash Attention / torch.compile"
    cuda = torch.cuda.is_available()
    if cuda:
        gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"PyTorch {v}, CUDA {torch.version.cuda}, VRAM={gb:.0f} GB"
    return f"PyTorch {v}, CUDA unavailable (CPU — training will be very slow)"
check("PyTorch ≥ 2.0", chk_torch)

for pkg in ["tiktoken", "numpy", "datasets", "huggingface_hub"]:
    check(f"Package: {pkg}", lambda p=pkg: __import__(p) and None)


# ══════════════════════════════════════════════════════════════════════════════
# 2. REQUIRED FILES PRESENT
# ══════════════════════════════════════════════════════════════════════════════
print("\n── 2. Required files ────────────────────────────────────────────────")

REQUIRED_FILES = [
    # Core training
    "model.py", "train.py", "configurator.py",
    # Evaluation & sampling
    "eval.py", "sample.py", "sample_batch.py",
    # Submission
    "hf_load.py", "sample_params.json",
    # Notebook
    "code_v2.ipynb",
    # Data scripts
    "data/rocstories/prepare.py",
    "data/tinystories/prepare.py",
    "data/mixed/prepare.py",
    "data/combined/prepare.py",
    # Task configs
    "config/train_t1_baseline.py",
    "config/train_t2_ablation_a.py",
    "config/train_t2_ablation_b.py",
    "config/train_t2_ablation_c.py",
    "config/train_t2_ablation_d.py",
    "config/train_t2_ablation_e.py",
    "config/train_t3_best.py",
    "config/train_t4_arena.py",
]

for f in REQUIRED_FILES:
    path = os.path.join(_REPO, f)
    check(f, lambda p=path, fn=f: (None if os.path.exists(p)
          else (_ for _ in ()).throw(AssertionError(f"MISSING — check your repo"))))


# ══════════════════════════════════════════════════════════════════════════════
# 3. model.py CORRECTNESS
# ══════════════════════════════════════════════════════════════════════════════
print("\n── 3. model.py fixes ────────────────────────────────────────────────")

def _src(fname):
    return open(os.path.join(_REPO, fname), encoding="utf-8").read()

def chk_swiglu_dim():
    src = _src("model.py")
    assert ("8 / 3" in src or "8/3" in src), \
        "SwiGLU uses 4× hidden — change to 8/3× to stay within 32M"
    assert "hidden = 4 * config.n_embd" not in src, \
        "Old 4× line still present in SwiGLUMLP — fix model.py"
    return "8/3× hidden confirmed"
check("SwiGLU 8/3× hidden dim", chk_swiglu_dim)

def chk_rope_cache():
    src = _src("model.py")
    assert "max(config.block_size, 2048)" in src, \
        "RoPE cache capped at block_size — fix: max(config.block_size, 2048)"
    return "RoPE max(block_size, 2048) confirmed"
check("RoPE cache size fix", chk_rope_cache)

def chk_topp():
    src = _src("model.py")
    assert ">= top_p" in src, \
        "top-p filter uses > instead of >= — boundary token incorrectly included"
    return ">= top_p confirmed"
check("top-p ≥ boundary fix", chk_topp)

def chk_param_counts():
    # Load model on CPU, no GPU needed
    os.chdir(_REPO)
    for mod in [k for k in sys.modules if "model" in k]:
        del sys.modules[mod]
    from model import GPT, GPTConfig
    base = dict(n_layer=6, n_head=6, n_embd=384, block_size=256,
                vocab_size=50304, bias=False, dropout=0.0)
    configs = {
        "A-Vanilla":   dict(use_rope=False, use_rmsnorm=False, use_swiglu=False, use_qk_norm=False),
        "B-RoPE":      dict(use_rope=True,  use_rmsnorm=False, use_swiglu=False, use_qk_norm=False),
        "C-SwiGLU":    dict(use_rope=False, use_rmsnorm=True,  use_swiglu=True,  use_qk_norm=False),
        "D-QKNorm":    dict(use_rope=False, use_rmsnorm=False, use_swiglu=False, use_qk_norm=True),
        "E-AllModern": dict(use_rope=True,  use_rmsnorm=True,  use_swiglu=True,  use_qk_norm=True),
    }
    over, summary = [], []
    for name, flags in configs.items():
        m = GPT(GPTConfig(**base, **flags))
        n = sum(p.numel() for p in m.parameters()) / 1e6
        summary.append(f"{name}={n:.2f}M")
        if n > 32.0:
            over.append(f"{name}={n:.2f}M")
    assert not over, f"Over 32M limit: {over}"
    return "all ≤ 32M — " + ", ".join(summary)
check("All 5 model configs ≤ 32M params", chk_param_counts)

def chk_forward_pass():
    import torch
    for mod in [k for k in sys.modules if "model" in k]:
        del sys.modules[mod]
    from model import GPT, GPTConfig
    base = dict(n_layer=6, n_head=6, n_embd=384, block_size=64,
                vocab_size=50304, bias=False, dropout=0.0)
    x = torch.randint(0, 50304, (2, 32))
    for label, flags in [
        ("vanilla",    dict(use_rope=False, use_rmsnorm=False, use_swiglu=False, use_qk_norm=False)),
        ("all-modern", dict(use_rope=True,  use_rmsnorm=True,  use_swiglu=True,  use_qk_norm=True)),
    ]:
        m = GPT(GPTConfig(**base, **flags)).eval()
        logits, loss = m(x, x)
        assert not math.isnan(loss.item()), f"NaN loss for {label}"
        assert logits.shape == (2, 32, 50304), f"Wrong logits shape {logits.shape} for {label}"
    return "vanilla + all-modern forward pass clean (CPU)"
check("Forward pass: no NaN, correct shape", chk_forward_pass)


# ══════════════════════════════════════════════════════════════════════════════
# 4. train.py CORRECTNESS
# ══════════════════════════════════════════════════════════════════════════════
print("\n── 4. train.py fixes ────────────────────────────────────────────────")

def chk_lr_autofix():
    src = _src("train.py")
    assert ("Auto-matched lr_decay_iters" in src or
            "lr_decay_iters = max_iters"   in src), \
        "LR auto-correction block missing — cosine decay will never fire"
    return "auto-correction block present"
check("lr_decay_iters auto-correction", chk_lr_autofix)

def chk_warmup_default():
    src = _src("train.py")
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("warmup_iters") and "=" in stripped and not stripped.startswith("#"):
            val = int(stripped.split("=")[1].split("#")[0].strip())
            assert val <= 200, \
                f"warmup_iters default={val} — should be ≤200 (old value was 2000)"
            return f"default warmup_iters={val}"
    return "warmup_iters default not found (set per-config — OK)"
check("warmup_iters default ≤ 200", chk_warmup_default)

def chk_log_interval():
    src = _src("train.py")
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("log_interval") and "=" in stripped and not stripped.startswith("#"):
            val = int(stripped.split("=")[1].split("#")[0].strip())
            assert val >= 5, \
                f"log_interval={val} — every-step logging creates huge .jsonl files, use ≥10"
            return f"default log_interval={val}"
    return "log_interval line not found"
check("log_interval default ≥ 5 (avoids huge logs)", chk_log_interval)

def chk_train_prints_ppl():
    src = _src("train.py")
    assert "val_ppl" in src, \
        "train.py eval step does not print PPL — add: val_ppl = math.exp(losses['val'])"
    assert "ppl" in src.lower(), \
        "PPL not printed during training eval — check train.py line ~397"
    return "val_ppl printed at every eval step"
check("train.py prints PPL during eval", chk_train_prints_ppl)

def chk_train_logs_ppl():
    src = _src("train.py")
    assert '"val_ppl"' in src or "'val_ppl'" in src, \
        'val_ppl not written to train_log.jsonl — add entry["val_ppl"] = ... in log_training_step()'
    return "val_ppl written to train_log.jsonl"
check("train.py logs val_ppl to JSONL", chk_train_logs_ppl)

def chk_label_smoothing():
    src = _src("train.py")
    assert "label_smoothing" in src, \
        "label_smoothing not found in train.py — needed for Task 3 regularisation"
    return "label_smoothing parameter present"
check("label_smoothing in train.py", chk_label_smoothing)


# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAINING CONFIGS
# ══════════════════════════════════════════════════════════════════════════════
print("\n── 5. Training configs ──────────────────────────────────────────────")

# (max_iters, lr_decay_iters, warmup range, must-have n_embd=384)
CONFIGS = {
    "config/train_t1_baseline.py":   (10000, 10000, (50, 200),  True),
    "config/train_t2_ablation_a.py": (10000, 10000, (50, 200),  True),
    "config/train_t2_ablation_b.py": (10000, 10000, (50, 200),  True),
    "config/train_t2_ablation_c.py": (10000, 10000, (50, 200),  True),
    "config/train_t2_ablation_d.py": (10000, 10000, (50, 200),  True),
    "config/train_t2_ablation_e.py": (10000, 10000, (50, 200),  True),
    "config/train_t3_best.py":       (15000, 15000, (50, 300),  True),
    "config/train_t4_arena.py":      (20000, 20000, (50, 400),  False),  # larger model, no 32M limit
}

def _check_config(rel_path, max_iters_exp, ldi_exp, warmup_range, check_32m):
    path = os.path.join(_REPO, rel_path)
    if not os.path.exists(path):
        raise AssertionError("MISSING")
    ns = {}
    exec(open(path, encoding="utf-8").read(), ns)
    problems = []

    # max_iters
    mi = ns.get("max_iters")
    if mi is None:
        problems.append("max_iters not set (will default to 600 000)")
    elif mi != max_iters_exp:
        problems.append(f"max_iters={mi} (expected {max_iters_exp})")

    # lr_decay_iters — must match max_iters (or use auto-fix in train.py)
    ldi = ns.get("lr_decay_iters", 600000)
    if ldi != 600000 and ldi != mi:
        problems.append(f"lr_decay_iters={ldi} ≠ max_iters={mi} — cosine decay won't reach min_lr")

    # warmup_iters
    wi = ns.get("warmup_iters")
    if wi is not None:
        lo, hi = warmup_range
        if not (lo <= wi <= hi):
            problems.append(f"warmup_iters={wi} outside expected range {lo}–{hi}")

    if problems:
        raise AssertionError("; ".join(problems))

    return (f"max={mi}, lr_decay={ldi}, warmup={wi or '(default)'}"
            + (", 32M OK" if check_32m else ", (arena — no size limit)"))

for rel, (mi, ldi, wr, check32) in CONFIGS.items():
    check(os.path.basename(rel),
          lambda r=rel, m=mi, l=ldi, w=wr, c=check32: _check_config(r, m, l, w, c))


# ══════════════════════════════════════════════════════════════════════════════
# 6. DATA PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
print("\n── 6. Data pipeline ─────────────────────────────────────────────────")

def chk_rocstories_split():
    src = _src("data/rocstories/prepare.py")
    assert ("indices[:n_val]" in src or "idx[:n_val]" in src or "val_idx" in src), \
        "Val split not found — verify data/rocstories/prepare.py is correct"
    return "val split pattern found"
check("data/rocstories/prepare.py — val split", chk_rocstories_split)

def chk_mixed_leakage():
    src = _src("data/mixed/prepare.py")
    second_rng = src.count("np.random.default_rng(SEED)") + src.count("np.random.default_rng(42)")
    assert second_rng <= 1, \
        f"{second_rng} RNG instantiations — old leaky code recreates RNG for val selection"
    assert ("val_stories" in src or "val_idx" in src), \
        "Val split variable not found in data/mixed/prepare.py"
    val_pos  = src.find("val_stories") if "val_stories" in src else src.find("val_idx")
    fmt_pos  = src.find("n_plain")
    if fmt_pos != -1:
        assert val_pos < fmt_pos, \
            "Val selection must happen BEFORE format split to prevent data leakage"
    return "val reserved before format split, single RNG"
check("data/mixed/prepare.py — no val/train leakage", chk_mixed_leakage)

def chk_combined_prepare():
    src = _src("data/combined/prepare.py")
    assert "TinyStories" in src or "tinystories" in src, \
        "data/combined/prepare.py does not reference TinyStories — Task 4 needs it"
    assert "--dry_run" in src or "dry_run" in src, \
        "dry_run mode missing from data/combined/prepare.py"
    return "TinyStories + dry_run mode present"
check("data/combined/prepare.py — Task 4 features", chk_combined_prepare)

def chk_existing_bins():
    import numpy as np
    bins = {
        "data/rocstories/train.bin": (500_000, 10_000_000),
        "data/rocstories/val.bin":   (50_000,  1_000_000),
    }
    found = []
    for rel, (lo, hi) in bins.items():
        path = os.path.join(_REPO, rel)
        if not os.path.exists(path):
            raise AssertionError(
                f"{rel} missing — run: python data/rocstories/prepare.py")
        n = len(__import__("numpy").fromfile(path, dtype=__import__("numpy").uint16))
        assert lo <= n <= hi, \
            f"{rel}: {n:,} tokens — outside expected range {lo:,}–{hi:,}"
        found.append(f"{rel.split('/')[-2]}/{rel.split('/')[-1]} = {n/1e6:.2f}M tokens")
    return " | ".join(found)
# Warn (not fail) — bins may not exist yet before the user downloads data
warn("ROCStories .bin files exist and are valid", chk_existing_bins)


# ══════════════════════════════════════════════════════════════════════════════
# 7. SUBMISSION ASSETS
# ══════════════════════════════════════════════════════════════════════════════
print("\n── 7. Submission assets ─────────────────────────────────────────────")

def chk_sample_params():
    path = os.path.join(_REPO, "sample_params.json")
    assert os.path.exists(path), \
        "sample_params.json missing — the graders load this for sampling (not PPL eval)"
    with open(path) as f:
        p = json.load(f)
    required = ["temperature", "top_k", "top_p", "repetition_penalty"]
    missing = [k for k in required if k not in p]
    assert not missing, f"Missing keys: {missing}"
    assert 0.3 <= p["temperature"] <= 1.5, f"temperature={p['temperature']} seems wrong"
    assert 0.5 <= p["top_p"]       <= 1.0, f"top_p={p['top_p']} seems wrong"
    assert p["repetition_penalty"] >= 1.0, f"repetition_penalty < 1 would penalise diversity"
    return (f"temp={p['temperature']}, top_k={p['top_k']}, "
            f"top_p={p['top_p']}, rep_pen={p['repetition_penalty']}")
check("sample_params.json — valid + keys present", chk_sample_params)

def chk_hf_load():
    src = _src("hf_load.py")
    assert "upload" in src, \
        "hf_load.py has no 'upload' command — check the file"
    assert "ckpt.pt" in src or "ckpt" in src, \
        "hf_load.py does not reference ckpt.pt — check submission logic"
    return "upload command + ckpt reference found"
check("hf_load.py — upload command present", chk_hf_load)

def chk_model_py_submission():
    """model.py must be self-contained (graders run it directly per assignment spec)."""
    src = _src("model.py")
    for cls in ["GPTConfig", "GPT"]:
        assert cls in src, f"{cls} class not found in model.py"
    for flag in ["use_rope", "use_rmsnorm", "use_swiglu", "use_qk_norm"]:
        assert flag in src, f"Flag '{flag}' not in GPTConfig — needed for resume eval"
    return "GPTConfig + GPT + all 4 flags present"
check("model.py — self-contained for HF submission", chk_model_py_submission)

def chk_notebook_valid():
    path = os.path.join(_REPO, "code_v2.ipynb")
    assert os.path.exists(path), "code_v2.ipynb missing"
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)   # will raise if invalid JSON
    assert "cells" in nb, "code_v2.ipynb JSON lacks 'cells' key — not a valid notebook"
    src_blocks = [
        "".join(c.get("source", []))
        for c in nb["cells"] if c.get("cell_type") == "code"
    ]
    all_src = "\n".join(src_blocks)
    assert "run_streaming" in all_src, \
        "run_streaming helper not found in code_v2.ipynb — training cells won't stream output"
    # Rough check: has cells for all 4 tasks
    for kw in ["train_t1_baseline", "train_t2_ablation", "train_t3_best"]:
        assert kw in all_src, \
            f"Notebook missing reference to '{kw}' — check training cells"
    return f"{len(nb['cells'])} cells, run_streaming OK, T1/T2/T3 configs found"
check("code_v2.ipynb — valid JSON + structure", chk_notebook_valid)


# ══════════════════════════════════════════════════════════════════════════════
# 8. GPU / VRAM ESTIMATE
# ══════════════════════════════════════════════════════════════════════════════
print("\n── 8. GPU / VRAM ────────────────────────────────────────────────────")

def chk_vram():
    import torch
    assert torch.cuda.is_available(), \
        "No CUDA GPU detected — training will run on CPU (very slow, not recommended)"
    gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    lines = [f"VRAM={gb:.0f} GB"]
    if gb < 10:
        lines.append("⚠ <10 GB — set batch_size=8, gradient_accumulation_steps=16 for T1–T3")
    elif gb < 15:
        lines.append("⚠ <15 GB — reduce batch_size to 16 for T3")
    if gb < 24:
        lines.append("⚠ <24 GB — set use_gradient_checkpointing=True for T4 (152M)")
    # Soft requirement: 10 GB minimum for useful training
    assert gb >= 6, \
        f"Only {gb:.0f} GB VRAM — too low for any reasonable batch size"
    return ", ".join(lines)
# Warn (not fail) — preflight must work without GPU (run locally before Colab)
warn("CUDA GPU with sufficient VRAM", chk_vram)


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
passes  = [r for r in results if r[0] == PASS]
warns   = [r for r in results if r[0] == WARN]
fails   = [r for r in results if r[0] == FAIL]

print(f"\n{'═'*62}")
print(f"  PRE-FLIGHT RESULT:  "
      f"{len(passes)} passed  |  {len(warns)} warnings  |  {len(fails)} failed")
print(f"{'═'*62}")

if warns:
    print(f"\n{WARN}  Warnings (advisory — training can still run):\n")
    for _, label, msg in warns:
        print(f"  {WARN}  {label}")
        if msg:
            print(f"       → {msg}")

if fails:
    print(f"\n{FAIL}  Fix these before running:\n")
    for _, label, msg in fails:
        print(f"  {FAIL}  {label}")
        if msg:
            print(f"       → {msg}")
    print()
    sys.exit(1)

print(f"\n  {PASS}  All hard checks passed — safe to start training.\n")
print("  Recommended order:")
print("    1.  python data/rocstories/prepare.py")
print("    2.  python data/tinystories/prepare.py           # Task 4 only")
print("    3.  python data/mixed/prepare.py --with_tinystories")
print("    4.  python train.py config/train_t1_baseline.py")
print("    5.  for cfg in a b c d e:")
print("            python train.py config/train_t2_ablation_${cfg}.py")
print("    6.  python train.py config/train_t3_best.py")
print("    7.  python eval.py init_from=resume out_dir=out-t3-best \\")
print("            input_file=data/rocstories/eval_stories.txt")
print("    8.  python hf_load.py upload out-t3-best <username>/nanoGPT_hw")
print()
sys.exit(0)
