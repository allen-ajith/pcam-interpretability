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
    "train": "pcam/camelyonpatch_level_2_split_train_x.h5",
    "val":   "pcam/camelyonpatch_level_2_split_valid_x.h5",
    "test":  "pcam/camelyonpatch_level_2_split_test_x.h5"
}

def extract_cls_attn(attentions):
    last = attentions[-1]  # (B, heads, tokens, tokens)
    cls_attn = last.mean(dim=1)[:, 0, 1:]  # (B, 196)
    return cls_attn.reshape(-1, 1, 14, 14)  # (B, 1, 14, 14)

def load_images(split, max_samples=None):
    h5_path = hf_hub_download(
        repo_id="allen-ajith/pcam-h5",
        filename=SPLIT_TO_FILENAME[split],
        repo_type="dataset"
    )
    with h5py.File(h5_path, "r") as f:
        return f["x"][:max_samples]

def save_h5(inputs, attns, out_path):
    with h5py.File(out_path, "w") as f:
        f.create_dataset("x", data=inputs, compression="gzip")  # (N, 3, 224, 224)
        f.create_dataset("y", data=attns, compression="gzip")   # (N, 224, 224)
    print(f"[✔] Saved dataset to {out_path} (x: {inputs.shape}, y: {attns.shape})")

def upload_to_hf(local_path, repo_id, split):
    upload_file(
        path_or_fileobj=local_path,
        path_in_repo=f"pcam_attn_{split}.h5",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Add {split} images + attention maps for U-Net"
    )
    print(f"[↑] Uploaded {split} data to {repo_id}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id_model", type=str, required=True,
                        help="HF model repo with your .pth checkpoint")
    parser.add_argument("--repo_id_dataset", type=str, required=True,
                        help="HF dataset repo to upload the .h5 files")
    parser.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--normalize_attn", action="store_true")
    parser.add_argument("--smooth_sigma", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()

    model = load_model_from_hf("dino-vits16", args.repo_id_model).to(DEVICE).eval()

    images = load_images(args.split, args.max_samples)
    print(f"[•] Loaded {len(images)} {args.split} images")

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    x_images = []
    y_attn_maps = []

    with torch.no_grad():
        for img in tqdm(images, desc=f"Generating {args.split} dataset"):
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)
            logits, attentions = model(img_tensor, output_attentions=True)

            attn = extract_cls_attn(attentions)
            attn = F.interpolate(attn, size=(224, 224), mode="bilinear", align_corners=False).squeeze().cpu().numpy()

            if args.smooth_sigma > 0:
                attn = gaussian_filter(attn, sigma=args.smooth_sigma)

            if args.normalize_attn:
                attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

            x_images.append(transform(img).numpy().astype(np.float32))   # (3, 224, 224)
            y_attn_maps.append(attn.astype(np.float32))                 # (224, 224)

    x_array = np.stack(x_images)
    y_array = np.stack(y_attn_maps)

    out_file = os.path.join(args.output_dir, f"pcam_attn_{args.split}.h5")
    save_h5(x_array, y_array, out_file)
    upload_to_hf(out_file, args.repo_id_dataset, args.split)

    import gc
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()
