"""Select the 5,000-study training pool from CT-RATE (train split only).

Rules: one reconstruction per study (lowest reconstruction index), one study
per patient, every study must have a reference report. Seed 42.

Inputs (from the CT-RATE release, see README):
    <ctrate>/train_metadata.csv
    <ctrate>/radiology_text_reports/train_reports.csv
Output:
    data/study_manifest.csv   (study_id, volume_id, reconstruction)

The released data/study_manifest.csv is the manifest of record; this script
documents the selection rule.
"""
import argparse
import csv
import random
import re

from common import DATA

VOL_RE = re.compile(r"^(?P<split>train|valid)_(?P<pat>\d+)_(?P<study>[a-z]+)_(?P<recon>\d+)\.nii\.gz$")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True, help="CT-RATE train_metadata.csv")
    ap.add_argument("--reports", required=True, help="CT-RATE train_reports.csv")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(DATA / "study_manifest.csv"))
    a = ap.parse_args()

    have_report = {r["VolumeName"] for r in csv.DictReader(open(a.reports, encoding="utf-8"))}
    by_study: dict[str, list] = {}
    for r in csv.DictReader(open(a.metadata, encoding="utf-8")):
        m = VOL_RE.match(r["VolumeName"])
        if not m or m.group("split") != "train" or r["VolumeName"] not in have_report:
            continue
        pat = f"train_{m.group('pat')}"
        sid = f"{pat}_{m.group('study')}"
        by_study.setdefault(sid, []).append((int(m.group("recon")), r["VolumeName"], pat))

    pick = {sid: sorted(v)[0] for sid, v in by_study.items()}
    by_pat: dict[str, list] = {}
    for sid, (_, _, pat) in pick.items():
        by_pat.setdefault(pat, []).append(sid)

    rng = random.Random(a.seed)
    pats = sorted(by_pat)
    rng.shuffle(pats)
    chosen = [sorted(by_pat[p])[0] for p in pats[: a.n]]

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["study_id", "volume_id", "reconstruction"])
        for sid in chosen:
            recon, vol, _ = pick[sid]
            w.writerow([sid, vol[:-7], recon])
    print(f"wrote {len(chosen)} studies -> {a.out}")


if __name__ == "__main__":
    main()
