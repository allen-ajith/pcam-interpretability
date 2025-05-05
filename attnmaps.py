import argparse
import h5py
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from utils.pcam_dataloader import get_pcam_loaders
from utils.load_model_from_hf import load_model_from_hf

def attention_rollout(attentions):
    result = torch.eye(attentions[0].size(-1)).to(attentions[0].device)
    for attn in attentions:
        attn_heads = attn.mean(dim=1)
        attn_heads = attn_heads / attn_heads.sum(dim=-1, keepdim=True)
        result = torch.bmm(attn_heads, result)
    return result[:, 0, 1:]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_batches", type=int, default=999)
    parser.add_argument("--output_file", type=str, default="/kaggle/working/pcam_attn_for_unet.h5")
    parser.add_argument("--rollout_type", type=str, default="last", choices=["last", "full"])
    parser.add_argument("--upsample_size", type=int, default=224)
    parser.add_argument("--normalize_attn", action="store_true")
    parser.add_argument("--smooth_sigma", type=float, default=0.0)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_hf("dino-vits16", args.repo_id).to(device).eval()
    loaders = get_pcam_loaders(batch_size=args.batch_size, model_type="vit")
    loader = {"train": loaders[0], "val": loaders[1], "test": loaders[2]}[args.split]

    all_images = []
    all_attn_maps = []

    torch.set_grad_enabled(False)
    for b_idx, (images, _) in enumerate(tqdm(loader, desc=f"{args.split} attention rollout")):
        if b_idx >= args.max_batches:
            break
        images = images.to(device)

        with torch.cuda.amp.autocast(enabled=args.amp):
            _, attn_list = model(images, output_attentions=True)

        if args.rollout_type == "full":
            attn_roll = attention_rollout(attn_list)
        else:
            attn_last = attn_list[-1].mean(dim=1)
            attn_last = attn_last / attn_last.sum(dim=-1, keepdim=True)
            attn_roll = attn_last[:, 0, 1:]

        for i in range(images.size(0)):
            img = images[i].detach().cpu()
            attn = attn_roll[i].detach().cpu().view(14, 14)
            attn = F.interpolate(
                attn.unsqueeze(0).unsqueeze(0),
                size=(args.upsample_size, args.upsample_size),
                mode="bilinear",
                align_corners=False
            ).squeeze().numpy()

            if args.smooth_sigma > 0:
                attn = gaussian_filter(attn, sigma=args.smooth_sigma)

            if args.normalize_attn:
                attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

            img_np = img.permute(1, 2, 0).numpy()
            all_images.append(img_np.astype(np.float32))
            all_attn_maps.append(attn[..., np.newaxis].astype(np.float32))

    with h5py.File(args.output_file, "w") as f:
        f.create_dataset("x", data=np.stack(all_images))
        f.create_dataset("y", data=np.stack(all_attn_maps))
    print(f"Saved: {args.output_file}")

if __name__ == "__main__":
    main()