import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torchvision.transforms import ToPILImage
from utils.pcam_dataloader import get_pcam_loaders
from utils.load_model_from_hf import load_model_from_hf
from huggingface_hub import upload_file
import os

def attention_rollout(attentions):
    result = torch.eye(attentions[0].size(-1)).to(attentions[0].device)
    for attn in attentions:
        attn_heads = attn.mean(dim=1)
        attn_heads = attn_heads / attn_heads.sum(dim=-1, keepdim=True)
        result = torch.bmm(attn_heads, result)
    return result[:, 0, 1:]

def show_side_by_side_grid(images, attn_roll, samples_per_row=2, save_path=None, max_samples=None):
    n = len(images) if max_samples is None else min(len(images), max_samples)
    ncols = samples_per_row * 2
    nrows = (n + samples_per_row - 1) // samples_per_row
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axs = axs.reshape(nrows, ncols)

    for idx in range(n):
        img = images[idx].permute(1, 2, 0).cpu().numpy()
        attn = attn_roll[idx].cpu().numpy()
        attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
        attn = np.uint8(255 * attn).reshape(14, 14)
        attn = np.kron(attn, np.ones((16, 16)))

        row = idx // samples_per_row
        col_base = (idx % samples_per_row) * 2

        axs[row, col_base].imshow(img)
        axs[row, col_base].set_title(f"Image {idx}")
        axs[row, col_base].axis("off")

        axs[row, col_base + 1].imshow(img)
        axs[row, col_base + 1].imshow(attn, cmap='jet', alpha=0.5)
        axs[row, col_base + 1].set_title("Attention Overlay")
        axs[row, col_base + 1].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved grid to {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=1)
    parser.add_argument("--samples_per_row", type=int, default=2)
    parser.add_argument("--full_rollout", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--push_to_hf", action="store_true")
    parser.add_argument("--hf_dataset_repo", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_hf("dino-vits16", args.repo_id).to(device).eval()

    train_loader, val_loader, test_loader = get_pcam_loaders(batch_size=args.batch_size, model_type="vit")
    if args.split == "train":
        loader = train_loader
    elif args.split == "val":
        loader = val_loader
    else:
        loader = test_loader

    torch.set_grad_enabled(False)

    for b_idx, (images, _) in enumerate(tqdm(loader, desc=f"{args.split} attention rollout")):
        if b_idx >= args.max_batches:
            break

        images = images.to(device)

        with torch.cuda.amp.autocast(enabled=args.amp):
            _, attn_list = model(images, output_attentions=True)

        if args.full_rollout:
            attn_roll = attention_rollout(attn_list)
        else:
            attn_last = attn_list[-1].mean(dim=1)
            attn_last = attn_last / attn_last.sum(dim=-1, keepdim=True)
            attn_roll = attn_last[:, 0, 1:]

        show_side_by_side_grid(images, attn_roll, samples_per_row=args.samples_per_row, save_path=args.save_path, max_samples=args.max_samples)

        if args.push_to_hf and args.save_path and args.hf_dataset_repo:
            upload_file(
                path_or_fileobj=args.save_path,
                path_in_repo=os.path.basename(args.save_path),
                repo_id=args.hf_dataset_repo,
                repo_type="dataset",
                commit_message="Upload attention overlay grid"
            )
            print(f"✅ Uploaded to HF: {args.hf_dataset_repo}/{os.path.basename(args.save_path)}")

if __name__ == "__main__":
    main()