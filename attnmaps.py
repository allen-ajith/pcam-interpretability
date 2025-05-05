import argparse
import os
import h5py
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from huggingface_hub import hf_hub_download, upload_file
import torchvision.transforms as T
from utils.load_model_from_hf import load_model_from_hf

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SPLIT_TO_FILENAME = {
    "train": ("pcam/camelyonpatch_level_2_split_train_x.h5", "pcam/camelyonpatch_level_2_split_train_y.h5"),
    "val":   ("pcam/camelyonpatch_level_2_split_valid_x.h5", "pcam/camelyonpatch_level_2_split_valid_y.h5"),
    "test":  ("pcam/camelyonpatch_level_2_split_test_x.h5",  "pcam/camelyonpatch_level_2_split_test_y.h5")
}

def extract_cls_attn(attentions):
    last = attentions[-1]
    cls_attn = last.mean(dim=1)[:, 0, 1:]
    return cls_attn.reshape(-1, 1, 14, 14)

def fix_vit_classifier_head(model, out_features=1, device=None):
    if hasattr(model, "classifier"):
        in_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_features, out_features)
        if device:
            model.classifier = model.classifier.to(device)
        print(f"✅ Replaced ViT classifier with Linear({in_features}, {out_features})")
    else:
        raise AttributeError("Model does not have a `classifier` attribute.")

def load_data_range(split, start, end):
    x_path, _ = SPLIT_TO_FILENAME[split]
    x_path = hf_hub_download("allen-ajith/pcam-h5", filename=x_path, repo_type="dataset")
    with h5py.File(x_path, "r") as fx:
        x = fx["x"][start:end]
    return x

def upload_to_hf(local_path, repo_id, split, start, end):
    part_name = f"pcam_attn_{split}_part{start}_{end}.h5"
    try:
        upload_file(
            path_or_fileobj=local_path,
            path_in_repo=part_name,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Upload {split} part {start}-{end}"
        )
        print(f"Upload successful: {part_name}")
        os.remove(local_path)
        print(f"Local file deleted: {local_path}")
    except Exception as e:
        print(f"Upload failed: {e}")
        print(f"Keeping local file: {local_path}")

def process_chunk(model, args, start_idx, end_idx):
    x_np = load_data_range(args.split, start_idx, end_idx)
    N = len(x_np)
    print(f"Loaded {N} samples from {args.split} [{start_idx}:{end_idx}]")

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    out_file = os.path.join(
        args.output_dir,
        f"pcam_attn_{args.split}_part{start_idx}_{end_idx}.h5"
    )

    with h5py.File(out_file, "w") as f:
        dset_y = f.create_dataset("y", shape=(N, 224, 224), dtype=np.float32)
        dset_logits = f.create_dataset("logits", shape=(N, 1), dtype=np.float32)

        with torch.no_grad():
            for idx in tqdm(range(N), desc=f"{start_idx}-{end_idx}"):
                img_t = transform(x_np[idx]).unsqueeze(0).to(DEVICE)

                logits, attns = model(img_t, output_attentions=True)
                attn = extract_cls_attn(attns)
                attn = F.interpolate(attn, size=(224, 224), mode="bilinear", align_corners=False).squeeze().cpu().numpy()

                if args.smooth_sigma > 0:
                    attn = gaussian_filter(attn, sigma=args.smooth_sigma)

                if args.normalize_attn:
                    attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

                dset_y[idx] = attn.astype(np.float32)
                dset_logits[idx] = logits.squeeze().cpu().numpy().reshape(1)

                if idx % 100 == 0:
                    torch.cuda.empty_cache()

    print(f"Saved chunk to: {out_file}")
    upload_to_hf(out_file, args.repo_id_dataset, args.split, start_idx, end_idx)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id_model", type=str, required=True)
    parser.add_argument("--repo_id_dataset", type=str, required=True)
    parser.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    parser.add_argument("--start_idx", type=int)
    parser.add_argument("--end_idx", type=int)
    parser.add_argument("--chunk_size", type=int)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--normalize_attn", action="store_true")
    parser.add_argument("--smooth_sigma", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()

    model = load_model_from_hf("dino-vits16", args.repo_id_model).to(DEVICE).eval()
    fix_vit_classifier_head(model, out_features=1, device=DEVICE)

    x_path, _ = SPLIT_TO_FILENAME[args.split]
    x_path = hf_hub_download("allen-ajith/pcam-h5", filename=x_path, repo_type="dataset")
    with h5py.File(x_path, "r") as f:
        total = len(f["x"])

    if args.all:
        print(f"\n=== Processing entire split: {args.split} (0:{total}) ===")
        process_chunk(model, args, 0, total)
    elif args.chunk_size:
        for start in range(0, total, args.chunk_size):
            end = min(start + args.chunk_size, total)
            print(f"\n=== Processing chunk {start}:{end} ===")
            process_chunk(model, args, start, end)
    else:
        if args.start_idx is None or args.end_idx is None:
            raise ValueError("Provide --start_idx and --end_idx when not using --chunk_size or --all")
        print(f"\n=== Processing range {args.start_idx}:{args.end_idx} ===")
        process_chunk(model, args, args.start_idx, args.end_idx)

if __name__ == "__main__":
    main()
