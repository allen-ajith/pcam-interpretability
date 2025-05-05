import argparse
import os
import h5py
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from huggingface_hub import hf_hub_download, upload_file
from utils.load_model_from_hf import load_model_from_hf
import torchvision.transforms as T

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SPLIT_TO_FILENAME = {
    "train": ("pcam/camelyonpatch_level_2_split_train_x.h5", "pcam/camelyonpatch_level_2_split_train_y.h5"),
    "val":   ("pcam/camelyonpatch_level_2_split_valid_x.h5", "pcam/camelyonpatch_level_2_split_valid_y.h5"),
    "test":  ("pcam/camelyonpatch_level_2_split_test_x.h5",  "pcam/camelyonpatch_level_2_split_test_y.h5")
}

def extract_cls_attn(attentions):
    last = attentions[-1]  # (B, heads, tokens, tokens)
    cls_attn = last.mean(dim=1)[:, 0, 1:]  # (B, 196)
    return cls_attn.reshape(-1, 1, 14, 14)  # (B, 1, 14, 14)

def load_data(split, max_samples=None):
    x_path, y_path = SPLIT_TO_FILENAME[split]
    x_path = hf_hub_download("allen-ajith/pcam-h5", filename=x_path, repo_type="dataset")
    y_path = hf_hub_download("allen-ajith/pcam-h5", filename=y_path, repo_type="dataset")
    with h5py.File(x_path, "r") as fx, h5py.File(y_path, "r") as fy:
        x = fx["x"][:max_samples]
        y = fy["y"][:max_samples]
    return x, y

def upload_to_hf(local_path, repo_id, split):
    upload_file(
        path_or_fileobj=local_path,
        path_in_repo=f"pcam_attn_{split}_with_logits.h5",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Upload {split} attn maps + logits + labels"
    )
    print(f"Uploaded to: https://huggingface.co/datasets/{repo_id}/blob/main/pcam_attn_{split}_with_logits.h5")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id_model", type=str, required=True)
    parser.add_argument("--repo_id_dataset", type=str, required=True)
    parser.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--normalize_attn", action="store_true")
    parser.add_argument("--smooth_sigma", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()

    model = load_model_from_hf("dino-vits16", args.repo_id_model).to(DEVICE).eval()
    images, labels = load_data(args.split, args.max_samples)
    N = len(images)
    print(f"Loaded {N} {args.split} samples")

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    x_buf = []
    y_buf = []
    label_buf = []
    logit_buf = []

    with torch.no_grad():
        for idx in tqdm(range(N), desc=f"Generating {args.split} attention maps"):
            img_t = transform(images[idx]).unsqueeze(0).to(DEVICE)
            true = labels[idx]

            logits, attns = model(img_t, output_attentions=True)
            attn = extract_cls_attn(attns)
            attn = F.interpolate(attn, size=(224, 224), mode="bilinear", align_corners=False).squeeze().cpu().numpy()

            if args.smooth_sigma > 0:
                attn = gaussian_filter(attn, sigma=args.smooth_sigma)

            if args.normalize_attn:
                attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

            x_buf.append(transform(images[idx]).numpy().astype(np.float32))
            y_buf.append(attn.astype(np.float32))
            label_buf.append(int(true))
            logit_buf.append(logits.squeeze().cpu().numpy())

    out_file = os.path.join(args.output_dir, f"pcam_attn_{args.split}_with_logits.h5")
    with h5py.File(out_file, "w") as f:
        f.create_dataset("x", data=np.stack(x_buf), compression="gzip")
        f.create_dataset("y", data=np.stack(y_buf), compression="gzip")
        f.create_dataset("label", data=np.array(label_buf, dtype=np.int64))
        f.create_dataset("logits", data=np.stack(logit_buf), compression="gzip")

    upload_to_hf(out_file, args.repo_id_dataset, args.split)

    import gc
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()
