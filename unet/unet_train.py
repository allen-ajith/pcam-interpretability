import h5py
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import os

# ----------------------------
# Load Datasets from Hugging Face
# ----------------------------
def load_data():
    orig_path = hf_hub_download(
        repo_id="allen-ajith/pcam-h5",
        filename="pcam/camelyonpatch_level_2_split_train_x.h5",
        repo_type="dataset"
    )
    cam_path = hf_hub_download(
        repo_id="pcam-interpretability/pcam_heatmaps",
        filename="pcam_heatmaps_train_1.h5",
        repo_type="dataset"
    )

    with h5py.File(orig_path, "r") as f:
        orig_images = f["x"][:]
        print('original')

    with h5py.File(cam_path, "r") as f:
        cam_images = f["x"][:]
        print('gradcam')

    return orig_images, cam_images

# ----------------------------
# Convert Heatmaps to Binary Masks
# ----------------------------
def rgb_to_grayscale(images):
    print('rgb')
    return np.dot(images[..., :3], [0.2989, 0.5870, 0.1140])

def convert_to_binary_masks(cam_images, threshold=0.5):
    print('binary masks')
    grayscale = rgb_to_grayscale(cam_images) / 255.0
    return (grayscale > threshold).astype(np.uint8)

def convert_to_soft_masks(cam_images):
    print('soft masks')
    grayscale = rgb_to_grayscale(cam_images) / 255.0
    return grayscale.astype(np.float32)

# ----------------------------
# Custom Dataset
# ----------------------------
class PCamSegmentationDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        mask = Image.fromarray((self.masks[idx] * 255).astype(np.uint8))

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        return img, mask

# ----------------------------
# U-Net Model
# ----------------------------
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
            )

        self.enc1 = CBR(3, 64)
        self.enc2 = CBR(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.dec1 = CBR(128, 64)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        d1 = self.up(self.dec1(e2))
        out = self.final(d1)
        return torch.sigmoid(out)

# ----------------------------
# Mask Conversion with Ignore
# ----------------------------
def soft_mask_to_hard_labels(mask, low_thresh=0.3, high_thresh=0.7):
    labels = torch.full_like(mask, -1)  # -1 means ignore
    labels[mask < low_thresh] = 0
    labels[mask > high_thresh] = 1
    return labels

# ----------------------------
# Metrics
# ----------------------------
def compute_metrics(preds, targets, threshold=0.5, eps=1e-7):
    preds = (preds > threshold).float()
    targets = (targets > threshold).float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = (preds + targets).sum(dim=(1, 2, 3)) - intersection
    dice = (2 * intersection + eps) / (preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + eps)
    iou = (intersection + eps) / (union + eps)
    return dice.mean().item(), iou.mean().item()

# ----------------------------
# Training Loop
# ----------------------------
def train_unet(orig_images, cam_images, epochs=5, batch_size=32, lr=1e-5, save_path="unet.pth"):
    print('train 1')
    grayscale_masks = rgb_to_grayscale(cam_images) / 255.0

    transform = T.Compose([T.ToTensor()])
    dataset = PCamSegmentationDataset(orig_images, grayscale_masks, transform=transform)

    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss(reduction='none')

    os.makedirs("checkpoints", exist_ok=True)
    print('train 3')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_pixels = 0
        total_dice = 0
        total_iou = 0

        print(f"\nEpoch {epoch+1}/{epochs}")
        for imgs, masks in tqdm(train_loader, desc="Training", leave=False):
            imgs = imgs.to(device)
            masks = masks.to(device).float() / 255.0
            labels = soft_mask_to_hard_labels(masks)
            valid_mask = (labels != -1).float()

            preds = model(imgs)
            loss = loss_fn(preds, labels.float())
            loss = (loss * valid_mask).sum() / valid_mask.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            correct = ((preds > 0.5) == (labels > 0.5)).float() * valid_mask
            total_correct += correct.sum().item()
            total_pixels += valid_mask.sum().item()

            dice, iou = compute_metrics(preds * valid_mask, labels * valid_mask)
            total_dice += dice
            total_iou += iou

        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_pixels
        avg_dice = total_dice / len(train_loader)
        avg_iou = total_iou / len(train_loader)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}")
        epoch_path = f"checkpoints/unet_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), epoch_path)
        print(f"Model checkpoint saved: {epoch_path}")

        # Optional: Validation pass can be added here

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# ----------------------------
# Main Entry Point
# ----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--save_path", type=str, default="unet.pth")
    args = parser.parse_args()

    orig_images, cam_images = load_data()
    train_unet(orig_images, cam_images, args.epochs, args.batch_size, args.lr, args.save_path)

# ----------------------------
# Optional: Soft label training (commented out)
# ----------------------------
# def train_with_soft_labels(...):
#     masks = convert_to_soft_masks(cam_images)
#     ...
#     loss_fn = nn.BCELoss()
#     loss = loss_fn(preds, masks)
