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
# Load Original Images and Grad-CAM Heatmaps
# ----------------------------
def load_orig_and_gradcam():
    orig_path = hf_hub_download(
        repo_id="allen-ajith/pcam-h5",
        filename="pcam/camelyonpatch_level_2_split_train_x.h5",
        repo_type="dataset"
    )
    gradcam_path = hf_hub_download(
        repo_id="pcam-interpretability/pcam_heatmaps",
        filename="pcam_heatmaps_train_1.h5",
        repo_type="dataset"
    )

    with h5py.File(orig_path, "r") as f:
        orig_images = f["x"][:]

    with h5py.File(gradcam_path, "r") as f:
        gradcam_maps = f["x"][:]

    return orig_images, gradcam_maps

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
            attn_maps.append(f["y"][:])  # attention masks
            logits_list.append(f["logits"][:].squeeze())  # predicted probabilities

    return np.concatenate(attn_maps), np.concatenate(logits_list)

# ----------------------------
# Combine Grad-CAM and Attention Masks
# ----------------------------
def combine_heatmaps(gradcam, attention, alpha=0.5):
    grad_gray = np.dot(gradcam[...,:3], [0.2989, 0.5870, 0.1140]) / 255.0
    attention_norm = attention / attention.max(axis=(1,2), keepdims=True)
    min_len = min(len(grad_gray), len(attention_norm))
    combined = alpha * grad_gray[:min_len] + (1 - alpha) * attention_norm[:min_len]
    return combined[:min_len]

# ----------------------------
# Dataset
# ----------------------------
class PCamCombinedDataset(Dataset):
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
# Visualization
# ----------------------------
def visualize_predictions(model, dataloader, device):
    import matplotlib.pyplot as plt
    model.eval()
    imgs, masks = next(iter(dataloader))
    imgs, masks = imgs.to(device), masks.to(device)
    with torch.no_grad():
        preds = model(imgs)

    for i in range(min(5, len(imgs))):
        img = imgs[i].permute(1, 2, 0).cpu().numpy()
        gt = masks[i][0].cpu().numpy()
        pred = preds[i][0].cpu().numpy()
        plt.figure(figsize=(10,3))
        plt.subplot(1,3,1); plt.imshow(img); plt.title("Image"); plt.axis('off')
        plt.subplot(1,3,2); plt.imshow(gt, cmap='gray'); plt.title("GT Mask")
        plt.subplot(1,3,3); plt.imshow(pred, cmap='gray'); plt.title("Predicted")
        plt.show()

# ----------------------------
# Training
# ----------------------------
def train():
    orig_images, gradcam_maps = load_orig_and_gradcam()
    attn_maps, logits = load_attention_maps()

    # Keep only correctly classified examples (logit > 0.5 for class=1, < 0.5 for class=0)
    labels = (logits > 0.5).astype(int)
    correct_idx = np.where((labels == 1) | (labels == 0))[0]

    orig_images = orig_images[correct_idx]
    gradcam_maps = gradcam_maps[correct_idx]
    attn_maps = attn_maps[correct_idx]

    masks = combine_heatmaps(gradcam_maps, attn_maps)

    transform = T.Compose([T.ToTensor()])
    dataset = PCamCombinedDataset(orig_images, masks, transform=transform)

    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
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
        torch.save(model.state_dict(), f"checkpoints/unet_epoch_{epoch+1}.pth")

    visualize_predictions(model, val_loader, device)

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    train()
