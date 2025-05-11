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
# Dataset
# ----------------------------
class PCamDataset(Dataset):
    def __init__(self, images, masks, img_transform=None, mask_transform=None):
        self.images = images
        self.masks = masks
        self.img_transform = img_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        mask = Image.fromarray((self.masks[idx] * 255).astype(np.uint8))

        if self.img_transform:
            img = self.img_transform(img)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        # Ensure mask has only 1 channel
        if mask.shape[0] == 3:  # If mask has 3 channels, take just the first one
            mask = mask[0].unsqueeze(0)
            
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

        # self.enc1 = CBR(3, 64)
        # self.enc2 = CBR(64, 128)
        # self.pool = nn.MaxPool2d(2)
        # self.dec1 = CBR(128, 64)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.final = nn.Conv2d(64, 1, 1)  # Output is single channel
        self.enc1 = CBR(3, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.enc4 = CBR(256, 512)
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = CBR(512, 1024)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec1 = CBR(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = CBR(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = CBR(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = CBR(128, 64)
        
        # Output layer
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # e1 = self.enc1(x)
        # e2 = self.enc2(self.pool(e1))
        # d1 = self.up(self.dec1(e2))
        # out = self.final(d1)
        # return torch.sigmoid(out)
        # Encoding
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoding with skip connections
        dec1 = self.up1(bottleneck)
        dec1 = torch.cat([dec1, enc4], dim=1)
        dec1 = self.dec1(dec1)
        
        dec2 = self.up2(dec1)
        dec2 = torch.cat([dec2, enc3], dim=1)
        dec2 = self.dec2(dec2)
        
        dec3 = self.up3(dec2)
        dec3 = torch.cat([dec3, enc2], dim=1)
        dec3 = self.dec3(dec3)
        
        dec4 = self.up4(dec3)
        dec4 = torch.cat([dec4, enc1], dim=1)
        dec4 = self.dec4(dec4)
        
        out = self.out(dec4)
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
    acc = (preds == targets).float().mean(dim=(1, 2, 3))
    return dice.mean().item(), iou.mean().item(), acc.mean().item()

# ----------------------------
# Training
# ----------------------------
def train():
    print("Loading datasets...")
    orig_images, gradcam_maps = load_orig_and_gradcam()

    orig_images = orig_images[:10000]  # optional subset
    gradcam_maps = gradcam_maps[:10000]

    print(f"Original image shape: {orig_images.shape}")
    print(f"Gradcam maps shape: {gradcam_maps.shape}")

    # Resize & Normalize Images
    img_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    # Resize Masks and ensure single channel
    mask_transform = T.Compose([
        T.Resize((224, 224)),
        T.Grayscale(num_output_channels=1),  # Convert to single channel
        T.ToTensor()
    ])

    dataset = PCamDataset(orig_images, gradcam_maps, img_transform=img_transform, mask_transform=mask_transform)
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32)

    # Check shapes of first batch
    imgs, masks = next(iter(train_loader))
    print(f"Batch image shape: {imgs.shape}")
    print(f"Batch mask shape: {masks.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    loss_fn = nn.BCELoss()

    os.makedirs("checkpoint", exist_ok=True)
    
    for epoch in range(14):
        model.train()
        total_loss, total_dice, total_iou, total_acc = 0, 0, 0, 0
        for imgs, masks in tqdm(train_loader):
            imgs, masks = imgs.to(device), masks.to(device).float()
            preds = model(imgs)
            
            # Double-check shapes before loss calculation
            if preds.shape != masks.shape:
                print(f"Warning: Shape mismatch - pred: {preds.shape}, mask: {masks.shape}")
                # Ensure masks match prediction shape if needed
                masks = masks[:, :1, :, :]  # Take just first channel if multi-channel
                
            loss = loss_fn(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            dice, iou, acc = compute_metrics(preds, masks)
            total_dice += dice
            total_iou += iou
            total_acc += acc

        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Dice={total_dice/len(train_loader):.4f}, IoU={total_iou/len(train_loader):.4f}, Acc={total_acc/len(train_loader):.4f}")
        
        # Validation step
        model.eval()
        val_loss, val_dice, val_iou, val_acc = 0, 0, 0, 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device).float()
                preds = model(imgs)
                val_loss += loss_fn(preds, masks).item()
                dice, iou, acc = compute_metrics(preds, masks)
                val_dice += dice
                val_iou += iou
                val_acc += acc
        
        print(f"Validation: Loss={val_loss/len(val_loader):.4f}, Dice={val_dice/len(val_loader):.4f}, IoU={val_iou/len(val_loader):.4f}, Acc={val_acc/len(val_loader):.4f}")
        torch.save(model.state_dict(), f"checkpoint/unet_gradcam_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()
