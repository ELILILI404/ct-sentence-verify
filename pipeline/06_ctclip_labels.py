"""Image-evidence sentence labels: RadGraph polarity vs. thresholded CT-CLIP p(c).

For every sentence with a mapped abnormality class c and definite polarity a,
    y = 1[ a == present  <=>  p(c) >= tau ]
Sentences without a mapped class receive no label and are excluded from the
CT-CLIP-labelled dataset. tau = 0.3 was used for the 0.5B reward model and
tau = 0.4 for the 3B one.

Inputs:  work/sentences.parquet, work/radgraph/shard*.json, data/ctclip_probs.parquet,
         data/class_dictionary.json
Output:  data/sentences.parquet  (adds radgraph_class, polarity, p_class, label_tau030,
         label_tau040, radgraph_entities; label_llm is filled by 07_llm_labels.py)

Running this script on the released data/ inputs (with --from-release) rebuilds the
label columns exactly; scripts/verify_release.py relies on that.
"""
import argparse
import glob
import json

import pandas as pd

from common import CLASSES, DATA, WORK

DICT = json.load(open(DATA / "class_dictionary.json"))
OBS2CLS = DICT["observation_to_class"]
ANAT_HINT = DICT["anatomy_hint"]
NORMAL_OBS = set(DICT["normal_observations"])
ANAT2CLS_NORMAL = DICT["anatomy_to_class_normal"]


def map_sentence(ents):
    """Return (class, polarity) for a sentence's RadGraph entities, or (None, None)."""
    obs = [(e["tokens"].lower(), e["label"]) for e in ents if e["label"].startswith("OBS")]
    anat = " ".join(e["tokens"].lower() for e in ents if e["label"].startswith("ANAT"))
    for tok, lab in obs:
        cls = OBS2CLS.get(tok)
        if cls is None:
            continue
        if cls.startswith("_"):            # site-ambiguous term: needs anatomy in the same sentence
            hit = None
            for keys, target in ANAT_HINT[cls]:
                if any(k in anat for k in keys):
                    hit = target
                    break
            if hit is None:
                continue
            cls = hit
        pol = "present" if lab == "OBS-DP" else ("absent" if lab == "OBS-DA" else None)
        if pol:
            return cls, pol
    for tok, lab in obs:                   # "heart contour is normal" -> Cardiomegaly denied
        if tok in NORMAL_OBS and lab == "OBS-DP":
            for keys, target in ANAT2CLS_NORMAL:
                if any(k in anat for k in keys):
                    return target, "absent"
    return None, None


def label_at(p, pol, tau):
    if p is None or pol is None:
        return None
    return bool(p >= tau) if pol == "present" else bool(p < tau)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--taus", default="0.3,0.4")
    ap.add_argument("--from-release", action="store_true",
                    help="take sentences and entities from data/sentences.parquet instead of work/")
    a = ap.parse_args()
    taus = [float(t) for t in a.taus.split(",")]

    if a.from_release:
        df = pd.read_parquet(DATA / "sentences.parquet")
        ents = {(r.study_id, int(r.sentence_index)): (json.loads(r.radgraph_entities) if r.radgraph_entities else [])
                for r in df.itertuples()}
        df = df[["study_id", "volume_id", "sentence_index", "sentence"] +
                (["label_llm"] if "label_llm" in df.columns else [])]
    else:
        df = pd.read_parquet(WORK / "sentences.parquet")
        ents = {}
        for f in glob.glob(str(WORK / "radgraph" / "shard*.json")):
            for e in json.load(open(f)):
                ents[(e["study_id"], e["sentence_index"])] = e["entities"]

    probs = pd.read_parquet(DATA / "ctclip_probs.parquet").set_index("volume_id")
    cls_col, pol_col, p_col, ent_col = [], [], [], []
    lab_cols = {t: [] for t in taus}
    for r in df.itertuples():
        e = ents.get((r.study_id, int(r.sentence_index)), [])
        cls, pol = map_sentence(e)
        p = float(probs.loc[r.volume_id, cls]) if (cls and r.volume_id in probs.index) else None
        cls_col.append(cls)
        pol_col.append(pol)
        p_col.append(p)
        ent_col.append(json.dumps(e) if e else None)
        for t in taus:
            lab_cols[t].append(label_at(p, pol, t))
    df = df.copy()
    df["radgraph_class"] = cls_col
    df["polarity"] = pol_col
    df["p_class"] = p_col
    for t in taus:
        df["label_tau%03d" % round(t * 100)] = pd.array(lab_cols[t], dtype="boolean")
    df["radgraph_entities"] = ent_col
    df.to_parquet(DATA / "sentences.parquet", index=False)

    n = df.radgraph_class.notna().sum()
    print(f"{len(df)} sentences, {n} with a mapped class")
    for t in taus:
        col = "label_tau%03d" % round(t * 100)
        print(f"  tau={t}: supported {int(df[col].sum())} / {n} ({100 * df[col].sum() / n:.1f}%)")


if __name__ == "__main__":
    main()
