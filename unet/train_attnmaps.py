import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from PIL import Image
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import os

# ----------------------------
# Load Original Images
# ----------------------------
def load_orig_images():
    orig_path = hf_hub_download(
        repo_id="allen-ajith/pcam-h5",
        filename="pcam/camelyonpatch_level_2_split_train_x.h5",
        repo_type="dataset"
    )
    with h5py.File(orig_path, "r") as f:
        orig_images = f["x"][:]
    return orig_images

# ----------------------------
# Load and Concatenate Attention Maps from Multiple Parts
# ----------------------------
def load_attention_maps():
    attn_maps, logits_list = [], []
    file_list = [
        "pcam_attn_train_part0_87381.h5",
        "pcam_attn_train_part174762_262144.h5",
        "pcam_attn_train_part87381_174762.h5"
    ]
    for file in file_list:
        path = hf_hub_download(
            repo_id="pcam-interpretability/dino_vit_attnmaps",
            filename=file,
            repo_type="dataset"
        )
        with h5py.File(path, "r") as f:
            attn_maps.append(f["y"][:])
            logits_list.append(f["logits"][:].squeeze())
    return np.concatenate(attn_maps), np.concatenate(logits_list)

# ----------------------------
# Dataset
# ----------------------------
class PCamAttnDataset(Dataset):
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
# Training
# ----------------------------
def train():
    orig_images = load_orig_images()
    attn_maps, logits = load_attention_maps()

    labels = (logits > 0.5).astype(int)
    correct_idx = np.where((labels == 1) | (labels == 0))[0]
    correct_idx = correct_idx[:10000]
    orig_images = orig_images[correct_idx]
    attn_maps = attn_maps[correct_idx]

    attn_maps = attn_maps / attn_maps.max(axis=(1,2), keepdims=True)

    transform = T.Compose([T.ToTensor()])
    dataset = PCamAttnDataset(orig_images, attn_maps, transform=transform)

    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.BCELoss()

    for epoch in range(5):
        model.train()
        total_loss, total_dice, total_iou = 0, 0, 0
        for imgs, masks in tqdm(train_loader):
            imgs, masks = imgs.to(device), masks.to(device).float() / 255.0
            preds = model(imgs)
            loss = loss_fn(preds, masks)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
            dice, iou = compute_metrics(preds, masks)
            total_dice += dice; total_iou += iou

        print(f"Epoch {epoch+1}: Loss={total_loss:.4f}, Dice={total_dice/len(train_loader):.4f}, IoU={total_iou/len(train_loader):.4f}")
        torch.save(model.state_dict(), f"checkpoints/unet_attn_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    train()
