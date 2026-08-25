"""Split every generated report into sentences (one row per sentence).

Input:  work/generated/gen_shard*.jsonl
Output: work/sentences.parquet  (study_id, volume_id, sentence_index, sentence)
"""
import glob
import json

import pandas as pd

from common import WORK, split_report


def main() -> None:
    rows = []
    for f in sorted(glob.glob(str(WORK / "generated" / "gen_shard*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            g = json.loads(line)
            for i, s in enumerate(split_report(g["generated_report"])):
                rows.append({"study_id": g["study_id"], "volume_id": g["volume_id"],
                             "sentence_index": i, "sentence": s})
    df = pd.DataFrame(rows).sort_values(["study_id", "sentence_index"]).reset_index(drop=True)
    df.to_parquet(WORK / "sentences.parquet", index=False)
    print(f"{len(df)} sentences from {df.study_id.nunique()} studies")


if __name__ == "__main__":
    main()
