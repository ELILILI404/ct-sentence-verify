"""Comparator labels: an LLM judges each sentence against the CT-RATE reference report.

One request per study (all sentences of that study in one JSON prompt),
temperature 0, model meta-llama/llama-3.1-70b-instruct through an
OpenAI-compatible endpoint. The rule is asymmetric: silence in the reference
is not an error; only a concrete fabricated or contradicted abnormality is.

Requires OPENROUTER_API_KEY in the environment and the CT-RATE reference
reports (train_reports.csv, gated access, see README).

Usage:  07_llm_labels.py --reports train_reports.csv [--limit N]
Input:  data/sentences.parquet
Output: work/llm_labels.jsonl  {study_id, volume_id, labels:[0/1/null], error}
        data/sentences.parquet  (label_llm column filled)
"""
import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

from common import DATA, WORK

MODEL = "meta-llama/llama-3.1-70b-instruct"
URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = """You are an expert radiologist verifying AI-generated chest CT report sentences against a ground-truth reference report.

For EACH numbered candidate sentence, decide if it is SUPPORTED (1) or NOT SUPPORTED (0) by the reference report, using this ASYMMETRIC rule (same principle used for clinical-accuracy grading of full reports):

- Label 1 (supported) if: the sentence describes a finding that IS present in the reference report; OR the sentence states a finding is normal/absent/unremarkable and this does not contradict the reference report; OR the sentence is a generic technical/procedural statement (e.g., imaging technique limitations, recommendations for further evaluation) that makes no specific clinical claim; OR the sentence describes a finding not explicitly mentioned in the reference report but does NOT contradict it (absence of mention is NOT an error — non-existent pathologies are not necessarily required to be mentioned in the reference).
- Label 0 (not supported) ONLY if: the sentence describes a SPECIFIC abnormal finding (a concrete pathology, mass, lesion, measurement, or diagnosis) that is absent from or contradicts the reference report (this is a hallucination); OR the sentence's severity/location/laterality directly contradicts the reference report; OR the sentence appears to describe an entirely different patient's clinical scenario/history (e.g., mentions a specific unrelated diagnosis, prior surgery, or disease history not found anywhere in the reference).

In short: do NOT penalize a sentence merely because the reference report is silent on that detail. ONLY penalize sentences that assert something concrete and clinically significant that contradicts the reference or is fabricated.

Respond with ONLY a JSON object mapping EVERY sentence number (as a string key, "0" through the last index) to 1 (supported) or 0 (not supported). Example for 3 sentences: {"0": 1, "1": 0, "2": 1}. You MUST include an entry for every single sentence number shown, no more and no fewer. No other text, no explanation."""


def build_user(reference: str, sentences: list[str]) -> str:
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences))
    return (f"Reference report:\n{reference}\n\n"
            f"Candidate sentences (indices 0 to {len(sentences) - 1}, {len(sentences)} total):\n{numbered}\n\n"
            f"Return the JSON object with exactly {len(sentences)} entries "
            f"(keys \"0\" through \"{len(sentences) - 1}\") now.")


def extract_labels(text: str, n: int):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj.get("labels"), dict):
            obj = obj["labels"]
        labels, got = [None] * n, 0
        for k, v in obj.items():
            try:
                idx = int(k)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < n:
                labels[idx] = 1 if int(v) else 0
                got += 1
        if got >= max(1, int(0.8 * n)):   # a study is usable only if >=80% of its sentences were labelled
            return labels
    except (ValueError, TypeError):
        pass
    return None


def call_one(rec, api_key, retries=4):
    n = len(rec["sentences"])
    payload = {"model": MODEL, "temperature": 0, "max_tokens": 1536,
               "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": build_user(rec["reference"], rec["sentences"])}]}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    base = {"study_id": rec["study_id"], "volume_id": rec["volume_id"]}
    for attempt in range(retries):
        try:
            resp = requests.post(URL, headers=headers, json=payload, timeout=90)
            if resp.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            resp.raise_for_status()
            labels = extract_labels(resp.json()["choices"][0]["message"]["content"], n)
            if labels is not None:
                return {**base, "labels": labels, "error": None}
            if attempt == retries - 1:
                return {**base, "labels": None, "error": "parse_failed_or_length_mismatch"}
        except requests.RequestException as e:
            if attempt == retries - 1:
                return {**base, "labels": None, "error": str(e)[:200]}
            time.sleep(3 * (attempt + 1))
    return {**base, "labels": None, "error": "exhausted"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", required=True, help="CT-RATE train_reports.csv")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    api_key = os.environ["OPENROUTER_API_KEY"]

    df = pd.read_parquet(DATA / "sentences.parquet").sort_values(["study_id", "sentence_index"])
    reports = pd.read_csv(a.reports)
    reports["volume_id"] = reports["VolumeName"].str.replace(".nii.gz", "", regex=False)
    vol2ref = reports.set_index("volume_id")["Findings_EN"].to_dict()

    records = []
    for sid, g in df.groupby("study_id"):
        vid = g["volume_id"].iloc[0]
        ref = vol2ref.get(vid)
        if isinstance(ref, str) and ref.strip():
            records.append({"study_id": sid, "volume_id": vid, "reference": ref,
                            "sentences": g["sentence"].tolist()})
    if a.limit:
        records = records[: a.limit]

    out = WORK / "llm_labels.jsonl"
    WORK.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["study_id"] for l in open(out, encoding="utf-8")}
    todo = [r for r in records if r["study_id"] not in done]
    print(f"{len(todo)} studies to label ({len(done)} already done)")

    with open(out, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(call_one, r, api_key) for r in todo]), 1):
            f.write(json.dumps(fut.result()) + "\n")
            f.flush()
            if i % 100 == 0:
                print(f"{i}/{len(todo)}", flush=True)

    labels = {}
    for line in open(out, encoding="utf-8"):
        r = json.loads(line)
        if r.get("error") or not r.get("labels"):
            continue
        for i, l in enumerate(r["labels"]):
            labels[(r["study_id"], i)] = None if l is None else bool(l)
    df["label_llm"] = pd.array([labels.get((r.study_id, int(r.sentence_index))) for r in df.itertuples()],
                               dtype="boolean")
    df.to_parquet(DATA / "sentences.parquet", index=False)
    n = df.label_llm.notna().sum()
    print(f"label_llm: {n} sentences, supported {int(df.label_llm.sum())} ({100 * df.label_llm.sum() / n:.1f}%)")


if __name__ == "__main__":
    main()
