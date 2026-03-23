"""Run Section 5 checks from notes/task-3-4-synthetic-data.md (no apostrophe in path)."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

checks = {}

# Section 1
f = "data/rocstories_synthetic/prepare.py"
if os.path.exists(f):
    src = open(f, encoding="utf-8").read()
    checks["synthetic prepare: file exists"] = True
    checks["synthetic prepare: quality filter fn"] = "_quality_filter" in src
    checks["synthetic prepare: _split_at_eot fn"] = "_split_at_eot" in src
    checks["synthetic prepare: 10pct val frac"] = "VAL_FRAC = 0.10" in src
    checks["synthetic prepare: val = original only"] = "orig_val" in src
    checks["synthetic prepare: argparse json_path"] = "json_path" in src
    checks["synthetic prepare: bad_phrase filter"] = "bad_phrases" in src
    checks["synthetic prepare: assertion val size"] = "assert" in src and "1024" in src
else:
    for k in [
        "file exists",
        "quality filter fn",
        "_split_at_eot fn",
        "10pct val frac",
        "val = original only",
        "argparse json_path",
        "bad_phrase filter",
        "assertion val size",
    ]:
        checks[f"synthetic prepare: {k}"] = False

# Section 2
f = "config/train_t3_synthetic.py"
if os.path.exists(f):
    src = open(f, encoding="utf-8").read()
    checks["t3_synthetic config: file exists"] = True
    checks["t3_synthetic config: out-t3-synthetic"] = (
        "out_dir               = 'out-t3-synthetic'" in src
    )
    checks["t3_synthetic config: dataset=synthetic"] = (
        "dataset = 'rocstories_synthetic'" in src
    )
    checks["t3_synthetic config: max_iters=12000"] = "max_iters     = 12000" in src
    checks["t3_synthetic config: lr_decay=12000"] = "lr_decay_iters = 12000" in src
    checks["t3_synthetic config: dropout=0.15"] = "dropout  = 0.15" in src
    checks["t3_synthetic config: warmup=200"] = "warmup_iters   = 200" in src
    checks["t3_synthetic config: n_layer=7"] = "n_layer  = 7" in src
    checks["t3_synthetic config: n_embd=384"] = "n_embd   = 384" in src
    checks["t3_synthetic config: scratch"] = "init_from             = 'scratch'" in src
    checks["t3_synthetic config: all modern flags"] = all(
        x in src
        for x in [
            "use_rmsnorm = True",
            "use_rope    = True",
            "use_swiglu  = True",
            "use_qk_norm = True",
        ]
    )
else:
    for k in [
        "file exists",
        "out-t3-synthetic",
        "dataset=synthetic",
        "max_iters=12000",
        "lr_decay=12000",
        "dropout=0.15",
        "warmup=200",
        "n_layer=7",
        "n_embd=384",
        "scratch",
        "all modern flags",
    ]:
        checks[f"t3_synthetic config: {k}"] = False

# Section 3
src = open("config/train_t4_finetune.py", encoding="utf-8").read()
checks["t4_finetune: dataset=synthetic"] = "dataset = 'rocstories_synthetic'" in src
checks["t4_finetune: max_iters=30000"] = "max_iters      = 30000" in src
checks["t4_finetune: lr_decay=30000"] = "lr_decay_iters = 30000" in src
checks["t4_finetune: lr=1e-4"] = "learning_rate = 1e-4" in src
checks["t4_finetune: wandb_run updated"] = "synthetic" in src or "gptoss" in src

# Section 4
nb = json.load(open("code_v2.ipynb", encoding="utf-8"))
all_src = " ".join("".join(c.get("source", [])) for c in nb["cells"])
checks["notebook: synthetic data prep cell"] = "synthetic_stories_gptoss120b.json" in all_src
checks["notebook: t3 synthetic training cell"] = (
    "t3-synthetic" in all_src or "T3_SYN" in all_src
)
checks["notebook: checkpoint comparison cell"] = (
    "Checkpoint Comparison" in all_src or "ppl_orig" in all_src
)
checks["notebook: USE_DIR in HF upload"] = "USE_DIR" in all_src
checks["notebook: t4 stage2 synthetic print"] = "rocstories_synthetic" in all_src
checks["notebook: data prep guard in t4"] = (
    "synthetic dataset verified" in all_src or "Synthetic dataset verified" in all_src
)

all_ok = all(checks.values())
passing = sum(checks.values())
total = len(checks)
print(f"Checks: {passing}/{total} pass")
print()
for k, v in checks.items():
    print(f"  {'OK' if v else 'XX'} {k}")
print()
print(f"RESULT: {'ALL PASS' if all_ok else 'FAILURES — fix before proceeding'}")
sys.exit(0 if all_ok else 1)
