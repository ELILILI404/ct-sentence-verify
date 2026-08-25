"""Sample one report per study from CT-CHAT (stochastic decoding, seed 42 + shard).

Decoding: temperature 1.0, top-p 0.9, top-k 50, 512 new tokens. Requires the
CT-CHAT checkpoint (LLaVA-style LoRA over Llama-3.1-8B-Instruct) and the
CT-ViT latent of each volume, both obtained through the CT-CHAT / CT-RATE
release (see README).

Usage:
    02_generate_reports.py <shard> <num_shards> --ctchat <repo> --weights <lora>
                           --base <llama-3.1-8b-instruct> --encodings <dir>
Output:
    work/generated/gen_shard<k>.jsonl  with {study_id, volume_id, generated_report}
"""
import argparse
import csv
import json
import random
import sys
import time

import numpy as np
import torch

from common import DATA, WORK

PROMPT = "<image>\nWould you mind generating the radiology report for the specified chest CT scan?<report_generation>"
SEED = 42


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("shard", type=int)
    ap.add_argument("num_shards", type=int)
    ap.add_argument("--ctchat", required=True, help="path to the CT-CHAT repository")
    ap.add_argument("--weights", required=True, help="CT-CHAT LoRA checkpoint dir")
    ap.add_argument("--base", required=True, help="Meta-Llama-3.1-8B-Instruct dir")
    ap.add_argument("--encodings", required=True, help="dir of <volume_id>.npz CT-ViT latents")
    a = ap.parse_args()

    sys.path.insert(0, a.ctchat)
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init

    random.seed(SEED + a.shard)
    np.random.seed(SEED + a.shard)
    torch.manual_seed(SEED + a.shard)
    torch.cuda.manual_seed_all(SEED + a.shard)
    disable_torch_init()
    tok, model, _, _ = load_pretrained_model(a.weights, a.base, "llava-lora", False, False, device="cuda")
    model.eval()

    rows = list(csv.DictReader(open(DATA / "study_manifest.csv", encoding="utf-8")))
    rows = [r for i, r in enumerate(rows) if i % a.num_shards == a.shard]
    out_dir = WORK / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"gen_shard{a.shard}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(l)["study_id"] for l in open(out, encoding="utf-8")}

    t0 = time.time()
    ok = 0
    for i, r in enumerate(rows, 1):
        if r["study_id"] in done:
            continue
        image = np.load(f"{a.encodings}/{r['volume_id']}.npz")["arr"]
        it = torch.tensor(image).to(model.device, dtype=torch.float16)
        conv = conv_templates["llama3"].copy()
        conv.append_message(conv.roles[0], PROMPT)
        conv.append_message(conv.roles[1], None)
        ids = tokenizer_image_token(conv.get_prompt(), tok, IMAGE_TOKEN_INDEX,
                                    return_tensors="pt").unsqueeze(0).to(model.device)
        with torch.inference_mode():
            gen = model.generate(ids, images=it, image_sizes=None,
                                 do_sample=True, temperature=1.0, top_p=0.9, top_k=50,
                                 max_new_tokens=512, use_cache=True)
        txt = tok.decode(gen[0], skip_special_tokens=True).strip()
        if not txt:
            continue
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps({"study_id": r["study_id"], "volume_id": r["volume_id"],
                                "generated_report": txt}, ensure_ascii=False) + "\n")
        ok += 1
        if i % 50 == 0:
            print(f"shard {a.shard}: {i}/{len(rows)} ok={ok} ({(time.time() - t0) / 60:.0f} min)", flush=True)
    print(f"shard {a.shard} done: ok={ok}")


if __name__ == "__main__":
    main()
