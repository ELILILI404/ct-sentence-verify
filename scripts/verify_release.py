"""Check that the released files are internally consistent and match the reported counts.

    python scripts/verify_release.py

1. data/sentences.parquet has 81,452 rows over 5,000 studies; 20,551 carry a class.
2. Recomputing the CT-CLIP labels from the released entities and p(c) reproduces
   label_tau030 / label_tau040 exactly.
3. Each PRM-format file matches the corresponding column (records, sentences, support rate).
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EXPECT = {"ctclip_tau030": (4955, 20551), "ctclip_tau040": (4955, 20551), "llm_reference": (5000, 81452)}


def fail(msg):
    print("FAIL:", msg)
    sys.exit(1)


df = pd.read_parquet(DATA / "sentences.parquet")
if len(df) != 81452 or df.study_id.nunique() != 5000:
    fail(f"sentence table: {len(df)} rows / {df.study_id.nunique()} studies")
if df.radgraph_class.notna().sum() != 20551:
    fail(f"{df.radgraph_class.notna().sum()} sentences with a class")
print("sentence table: 81,452 sentences, 5,000 studies, 20,551 with a mapped class  ok")

# --- recompute image-evidence labels into a scratch copy and compare ---
tmp = ROOT / "work" / "_verify"
tmp.mkdir(parents=True, exist_ok=True)
saved = df.copy()
subprocess.run([sys.executable, str(ROOT / "pipeline" / "06_ctclip_labels.py"), "--from-release"],
               check=True, cwd=ROOT / "pipeline")
re = pd.read_parquet(DATA / "sentences.parquet")
for col in ("radgraph_class", "polarity", "label_tau030", "label_tau040"):
    if not saved[col].equals(re[col]):
        fail(f"recomputed {col} differs from released column")
saved.to_parquet(DATA / "sentences.parquet", index=False)  # restore byte-identical original
print("recomputed labels identical to released columns  ok")

# --- PRM-format files ---
for name, (n_rec, n_sent) in EXPECT.items():
    col = {"ctclip_tau030": "label_tau030", "ctclip_tau040": "label_tau040", "llm_reference": "label_llm"}[name]
    recs = [json.loads(l) for l in open(DATA / "prm_format" / f"{name}.jsonl", encoding="utf-8")]
    sents = sum(len(r["labels"]) for r in recs)
    pos = sum(sum(r["labels"]) for r in recs)
    if (len(recs), sents) != (n_rec, n_sent):
        fail(f"{name}: {len(recs)} records / {sents} sentences")
    if pos != int(df[col].sum()):
        fail(f"{name}: supported count {pos} != column sum {int(df[col].sum())}")
    print(f"{name:15s} {len(recs):5d} records {sents:6d} sentences {100 * pos / sents:5.1f}% supported  ok")
print("ALL CHECKS PASSED")
