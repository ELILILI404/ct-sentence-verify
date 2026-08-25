"""Export a label column of data/sentences.parquet in TRL PRMTrainer format.

One record per study: a fixed prompt (no reference text, so no leakage), the
study's labelled sentences as `completions`, and one boolean per sentence.
Sentences whose label is null are dropped; studies left empty are skipped.

Usage:  08_build_prm_dataset.py --label label_tau030 --out data/prm_format/ctclip_tau030.jsonl
        08_build_prm_dataset.py --label label_llm    --out data/prm_format/llm_reference.jsonl
        add --hf-dir <dir> to also save a datasets.Dataset for 09_train_prm.py
"""
import argparse
import json

import pandas as pd

from common import DATA, PROMPT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, choices=["label_tau030", "label_tau040", "label_llm"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--hf-dir", default=None)
    a = ap.parse_args()

    df = pd.read_parquet(DATA / "sentences.parquet")
    rows = []
    with open(a.out, "w", encoding="utf-8") as f:
        for sid, g in df.groupby("study_id", sort=True):
            g = g[g[a.label].notna()].sort_values("sentence_index")
            if not len(g):
                continue
            rec = {"study_id": sid, "prompt": PROMPT, "completions": g["sentence"].tolist(),
                   "labels": [bool(x) for x in g[a.label].tolist()]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rows.append(rec)
    n = sum(len(r["labels"]) for r in rows)
    pos = sum(sum(r["labels"]) for r in rows)
    print(f"{len(rows)} records, {n} sentences, {pos} supported ({100 * pos / n:.1f}%) -> {a.out}")

    if a.hf_dir:
        from datasets import Dataset
        Dataset.from_list([{k: r[k] for k in ("prompt", "completions", "labels")} for r in rows]).save_to_disk(a.hf_dir)
        print("saved HF dataset ->", a.hf_dir)


if __name__ == "__main__":
    main()
