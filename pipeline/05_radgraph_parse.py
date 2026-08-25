"""Parse each sentence with RadGraph to obtain observation / anatomy entities.

Entity labels follow RadGraph: OBS-DP (definitely present), OBS-DA
(definitely absent), OBS-U (uncertain), ANAT-DP. Only the token and label of
each entity are kept.

Usage:  05_radgraph_parse.py <shard> <num_shards>
Input:  work/sentences.parquet
Output: work/radgraph/shard<k>.json  [{study_id, sentence_index, entities:[{tokens,label}]}]
"""
import argparse
import json

import pandas as pd
import torch

from common import WORK


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("shard", type=int)
    ap.add_argument("num_shards", type=int)
    ap.add_argument("--batch", type=int, default=200)
    a = ap.parse_args()

    # radgraph pins CUDA tensors in its checkpoint; force CPU-compatible loading
    _orig = torch.load
    torch.load = lambda *x, **k: _orig(*x, **{**k, "map_location": torch.device("cpu"), "weights_only": False})
    from radgraph import RadGraph

    df = pd.read_parquet(WORK / "sentences.parquet")
    df = df.iloc[a.shard::a.num_shards]
    rg = RadGraph(model_type="radgraph")
    out_dir = WORK / "radgraph"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = []
    recs = df.to_dict("records")
    for i in range(0, len(recs), a.batch):
        chunk = recs[i:i + a.batch]
        o = rg(hyps=[r["sentence"] for r in chunk])
        for j, r in enumerate(chunk):
            ents = [{"tokens": e["tokens"], "label": e["label"]}
                    for e in o.get(str(j), {}).get("entities", {}).values()]
            res.append({"study_id": r["study_id"], "sentence_index": int(r["sentence_index"]),
                        "entities": ents})
        if i % 2000 == 0:
            print(f"shard {a.shard}: {i}/{len(recs)}", flush=True)
    json.dump(res, open(out_dir / f"shard{a.shard}.json", "w"))
    print(f"shard {a.shard} done: {len(res)} sentences")


if __name__ == "__main__":
    main()
