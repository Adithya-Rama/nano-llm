import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
with open(ROOT / "code_v2.ipynb", encoding="utf-8") as f:
    nb = json.load(f)
for i, c in enumerate(nb["cells"]):
    s = "".join(c.get("source", []))
    if "T3_CONFIG" in s and "train_t3_best" in s:
        print("T3 cell", i)
    if "T3_DIR" in s and "SUBMIT_DIR" in s:
        print("HF cell", i)
    if "rocstories_plain" in s and "train_t4_finetune" in s:
        print("T4 stage2", i)
    if "| T3 |" in s or ("T3" in s and "All Modern" in s and "|" in s and "Summary" in s):
        pass
    if "§5" in s and "| T1 |" in s:
        print("summary table", i)
    if "sample_batch" in s and "out-t3-best" in s:
        print("sample eval", i)
    if "HuggingFace" in s and "upload" in s.lower() and "T3_DIR" in s:
        print("HF upload detail", i)
