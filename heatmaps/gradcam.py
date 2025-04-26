import torch
import os
import argparse
from tqdm import tqdm
from torchvision.utils import save_image
from torchvision import transforms
from torchcam.methods import SmoothGradCAMpp
from utils.load_model_from_hf import load_model_from_hf
from utils.pcam_dataloader import get_pcam_loaders
from scipy.ndimage import gaussian_filter
import numpy as np

def normalize_heatmap(heatmap):
    return (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

def process_heatmap(heatmap, blur_sigma=None, threshold=None):
    heatmap_np = heatmap.cpu().numpy()
    if blur_sigma:
        heatmap_np = gaussian_filter(heatmap_np, sigma=blur_sigma)
    if threshold:
        heatmap_np = np.where(heatmap_np < threshold, 0, heatmap_np)
    return torch.from_numpy(heatmap_np)

def save_heatmap(heatmap, idx, save_dir, save_format="png"):
    os.makedirs(save_dir, exist_ok=True)
    fname = f"{idx}.{save_format}"
    if save_format == "png":
        save_image(heatmap.unsqueeze(0), os.path.join(save_dir, fname))
    elif save_format == "pt":
        torch.save(heatmap, os.path.join(save_dir, fname))
    else:
        raise ValueError("Unsupported save_format. Choose 'png' or 'pt'.")

def generate_heatmaps(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = load_model_from_hf(args.model_name, args.repo_id, device=device)


    cam_extractor = SmoothGradCAMpp(model, target_layer=args.target_layer)


    train_loader, val_loader, test_loader = get_pcam_loaders(
        batch_size=args.batch_size, model_type="resnet", seed=42
    )
    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    loader = loaders[args.split]

    model.eval()
    idx_counter = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"Generating Heatmaps [{args.split}]"):
            images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)

            high_conf_mask = (probs > args.confidence_threshold).squeeze(1) & (labels.squeeze(1) == 1)
            if high_conf_mask.sum() == 0:
                continue

            selected_images = images[high_conf_mask]
            selected_outputs = outputs[high_conf_mask]

            for idx_in_batch in range(selected_images.size(0)):
                image = selected_images[idx_in_batch].unsqueeze(0)
                output = selected_outputs[idx_in_batch].unsqueeze(0)

                # Get Grad-CAM++ heatmap
                heatmap = cam_extractor(output.squeeze(0), class_idx=0)[0]
                heatmap = torch.from_numpy(heatmap)

                # Normalize and post-process
                heatmap = normalize_heatmap(heatmap)
                heatmap = process_heatmap(heatmap, blur_sigma=args.blur_sigma, threshold=args.threshold)

                # Save as PNG and/or PT
                save_heatmap(
                    heatmap, idx_counter,
                    save_dir=os.path.join(args.save_dir, args.split, args.model_name, "masks"),
                    save_format=args.save_format
                )

                if args.save_images:
                    save_image(
                        image.cpu(),
                        os.path.join(args.save_dir, args.split, args.model_name, "images", f"{idx_counter}.png")
                    )

                idx_counter += 1

            cam_extractor.clear_hooks()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Grad-CAM++ Heatmaps for PCam")
    parser.add_argument("--model_name", type=str, required=True, choices=["resnet50", "swin-tiny", "dino-vits16"])
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--target_layer", type=str, default="backbone.layer4[-1]", help="Target layer for Grad-CAM++")
    parser.add_argument("--confidence_threshold", type=float, default=0.9)
    parser.add_argument("--save_dir", type=str, default="outputs/heatmaps")
    parser.add_argument("--save_format", type=str, default="png", choices=["png", "pt"])
    parser.add_argument("--save_images", action="store_true", help="Also save the original input images")
    parser.add_argument("--blur_sigma", type=float, default=None, help="Gaussian blur sigma for heatmaps")
    parser.add_argument("--threshold", type=float, default=None, help="Threshold below which activations are zeroed")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    generate_heatmaps(args)
