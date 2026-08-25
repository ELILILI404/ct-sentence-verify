"""Zero-shot CT-CLIP abnormality probabilities p(c) for the 18 CT-RATE classes.

Runs the released CT-CLIP inference (CT-CLIP_v2 checkpoint, CT-ViT image
tower, CXR-BERT text tower) over the training-pool volumes and writes the
5,000 x 18 probability matrix used by 06_ctclip_labels.py.

Usage:
    03_ctclip_probs.py --ctclip <repo> --ckpt CT-CLIP_v2.pt --volumes <dir>
                       --reports train_reports.csv --labels train_predicted_labels.csv
Output:
    work/ctclip/{predicted_weights.npz, accessions.txt}
    data/ctclip_probs.parquet   (volume_id + one column per class)
"""
import argparse
import sys

import numpy as np
import pandas as pd

from common import CLASSES, DATA, WORK


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctclip", required=True, help="path to the CT-CLIP repository")
    ap.add_argument("--ckpt", required=True, help="CT-CLIP_v2.pt")
    ap.add_argument("--volumes", required=True, help="dir holding the manifest volumes (.nii.gz)")
    ap.add_argument("--reports", required=True, help="CT-RATE train_reports.csv")
    ap.add_argument("--labels", required=True, help="CT-RATE train_predicted_labels.csv")
    a = ap.parse_args()

    for p in (f"{a.ctclip}/scripts", f"{a.ctclip}/CT_CLIP"):
        sys.path.insert(0, p)
    import torch
    import torch.multiprocessing
    from ct_clip import CTCLIP
    from transformer_maskgit import CTViT
    from transformers import BertModel, BertTokenizer
    from zero_shot import CTClipInference

    torch.multiprocessing.set_sharing_strategy("file_system")
    tokenizer = BertTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized", do_lower_case=True)
    text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")
    text_encoder.resize_token_embeddings(len(tokenizer))
    image_encoder = CTViT(dim=512, codebook_size=8192, image_size=480, patch_size=20,
                          temporal_patch_size=10, spatial_depth=4, temporal_depth=4,
                          dim_head=32, heads=8)
    clip = CTCLIP(image_encoder=image_encoder, text_encoder=text_encoder,
                  dim_image=294912, dim_text=768, dim_latent=512,
                  extra_latent_projection=False, use_mlm=False,
                  downsample_image_embeds=False, use_all_token_embeds=False)
    clip.load_state_dict(torch.load(a.ckpt, map_location="cpu"), strict=False)

    out = WORK / "ctclip"
    out.mkdir(parents=True, exist_ok=True)
    CTClipInference(clip, data_folder=a.volumes, reports_file=a.reports,
                    meta_file=str(DATA / "study_manifest.csv"), labels=a.labels,
                    results_folder=str(out) + "/").infer()

    pred = np.load(out / "predicted_weights.npz")["data"]
    acc = open(out / "accessions.txt").read().split()
    df = pd.DataFrame(pred, columns=CLASSES)
    df.insert(0, "volume_id", acc)
    df.to_parquet(DATA / "ctclip_probs.parquet", index=False)
    print("wrote", len(df), "volumes ->", DATA / "ctclip_probs.parquet")


if __name__ == "__main__":
    main()
