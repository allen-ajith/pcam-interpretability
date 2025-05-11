import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        print('file')
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
# Analysis functions
# ----------------------------
def analyze_masks(masks):
    """Analyze the class distribution in masks"""
    total_pixels = masks.size
    ignore_pixels = np.sum(masks == 255)
    foreground_pixels = np.sum(masks == 1)
    background_pixels = np.sum(masks == 0)
    
    print(f"Total pixels: {total_pixels}")
    print(f"Ignore pixels: {ignore_pixels} ({100*ignore_pixels/total_pixels:.2f}%)")
    print(f"Foreground pixels: {foreground_pixels} ({100*foreground_pixels/total_pixels:.2f}%)")
    print(f"Background pixels: {background_pixels} ({100*background_pixels/total_pixels:.2f}%)")
    
    if foreground_pixels > 0:
        bg_fg_ratio = background_pixels / foreground_pixels
        print(f"Background to foreground ratio: {bg_fg_ratio:.2f}:1")
    else:
        print("Warning: No foreground pixels found!")
    
    return foreground_pixels, background_pixels, ignore_pixels

def threshold_with_ignore(masks, low=0.2, high=0.8, ignore_value=255):
    """
    Thresholds attention maps with three levels:
    - Below low: background (0)
    - Above high: foreground (1)
    - Between low and high: ignore (255)
    """
    masks_bin = np.full_like(masks, ignore_value, dtype=np.uint8)
    masks_bin[masks <= low] = 0
    masks_bin[masks >= high] = 1
    return masks_bin

# ----------------------------
# Dataset
# ----------------------------
class PCamAttentionDataset(Dataset):
    def __init__(self, images, attn_masks, transform=None):
        self.images = images
        self.attn_masks = attn_masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        mask = self.attn_masks[idx]

        if self.transform:
            img = self.transform(img)

        mask = Image.fromarray(mask)
        mask = T.Resize((224, 224), interpolation=Image.NEAREST)(mask)
        mask = torch.from_numpy(np.array(mask)).long()

        return img, mask

# ----------------------------
# U-Net Model
# ----------------------------
class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()
        
        # Encoder
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
            
        self.enc1 = conv_block(n_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = conv_block(512, 1024)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec1 = conv_block(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = conv_block(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = conv_block(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = conv_block(128, 64)
        
        # Output layer
        self.out = nn.Conv2d(64, n_classes, kernel_size=1)
        
    def forward(self, x):
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
# Loss functions
# ----------------------------
class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    - Reduces weight for well-classified examples
    - Focuses on hard examples
    """
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets, ignore_index=255):
        # Create mask for ignored pixels
        mask = (targets != ignore_index).float()
        
        # Convert targets to binary
        targets_binary = torch.zeros_like(targets, dtype=torch.float32)
        targets_binary[targets == 1] = 1.0
        
        # Calculate BCE
        bce = F.binary_cross_entropy(inputs, targets_binary, reduction='none')
        
        # Calculate focal weights
        pt = torch.exp(-bce)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha for positives and (1-alpha) for negatives
        alpha_factor = torch.ones_like(targets_binary) * self.alpha
        alpha_factor[targets_binary == 0] = 1 - self.alpha
        
        # Calculate focal loss
        loss = alpha_factor * focal_weight * bce
        
        # Apply ignore mask
        loss = loss * mask
        
        # Reduce according to strategy
        if self.reduction == 'mean':
            if mask.sum() > 0:
                return loss.sum() / mask.sum()
            return torch.tensor(0.0, device=loss.device)
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class DiceLoss(nn.Module):
    """
    Dice Loss for handling class imbalance in segmentation
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, inputs, targets, ignore_index=255):
        # Create mask for non-ignored pixels
        mask = (targets != ignore_index).float()
        
        # Convert targets to binary
        targets = (targets == 1).float()
        
        # Apply mask to both inputs and targets
        inputs = inputs * mask
        targets = targets * mask
        
        # Flatten
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        # Calculate Dice coefficient
        intersection = (inputs_flat * targets_flat).sum()
        union = inputs_flat.sum() + targets_flat.sum()
        
        # Calculate Dice loss
        if union > 0:
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            return 1 - dice
        return torch.tensor(0.0, device=inputs.device)

class CombinedLoss(nn.Module):
    """
    Combined loss: Focal Loss + Dice Loss
    """
    def __init__(self, focal_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2)
        self.dice_loss = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        
    def forward(self, inputs, targets, ignore_index=255):
        focal = self.focal_loss(inputs, targets, ignore_index)
        dice = self.dice_loss(inputs, targets, ignore_index)
        return self.focal_weight * focal + self.dice_weight * dice

# ----------------------------
# Metrics
# ----------------------------
def compute_metrics(preds, targets, threshold=0.5, eps=1e-7):
    """
    Compute segmentation metrics.
    
    Args:
        preds: prediction tensor of any shape
        targets: target tensor of the same shape as preds
        threshold: threshold for binarizing predictions
        eps: small constant to avoid division by zero
        
    Returns:
        dice, iou, accuracy, precision, recall metrics
    """
    # Ensure tensors are flat for binary metrics calculations
    preds_flat = (preds > threshold).float().view(-1)
    targets_flat = (targets > threshold).float().view(-1)
    
    # Calculate metrics
    tp = (preds_flat * targets_flat).sum()
    fp = preds_flat.sum() - tp
    fn = targets_flat.sum() - tp
    tn = preds_flat.shape[0] - tp - fp - fn
    
    # Dice coefficient
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    
    # IoU / Jaccard index
    iou = (tp + eps) / (tp + fp + fn + eps)
    
    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    # Precision
    precision = (tp + eps) / (tp + fp + eps)
    
    # Recall / Sensitivity
    recall = (tp + eps) / (tp + fn + eps)
    
    return dice.item(), iou.item(), accuracy.item(), precision.item(), recall.item()

# ----------------------------
# Training
# ----------------------------
def train():
    print("Loading original images...")
    orig_images = load_orig_images()
    
    print("Loading attention maps...")
    attn_maps, logits = load_attention_maps()

    print(f"Original data shapes - Images: {orig_images.shape}, Attention maps: {attn_maps.shape}, Logits: {logits.shape}")
    
    labels = (logits > 0.5).astype(int)
    correct_idx = np.where((labels == 1) | (labels == 0))[0]
    correct_idx = correct_idx[:10000]  # subset for testing
    
    orig_images = orig_images[correct_idx]
    attn_maps = attn_maps[correct_idx]

    print(f"Using {len(correct_idx)} samples with valid labels")
    print(f"Working data shapes - Images: {orig_images.shape}, Attention maps: {attn_maps.shape}")
    
    # Normalize attention maps
    print("Processing attention maps...")
    attn_maps = attn_maps / np.maximum(attn_maps.max(axis=(1, 2), keepdims=True), 1e-10)
    
    # Try different thresholds to find better class balance
    print("\nTesting different thresholding values:")
    thresholds = [(0.2, 0.8), (0.3, 0.7), (0.4, 0.6), (0.5, 0.9)]
    
    for low, high in thresholds:
        print(f"\nTesting threshold: low={low}, high={high}")
        test_masks = threshold_with_ignore(attn_maps[:100], low=low, high=high)
        fg, bg, ig = analyze_masks(test_masks)
    
    # Select the best threshold based on analysis above
    # Let's assume we've found a good threshold after analysis
    best_low, best_high = 0.3, 0.7  # Adjust based on analysis results
    print(f"\nUsing selected thresholds: low={best_low}, high={best_high}")
    
    attn_maps = threshold_with_ignore(attn_maps, low=best_low, high=best_high)
    print("\nFinal mask statistics:")
    analyze_masks(attn_maps)
    
    # Setup transforms
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    dataset = PCamAttentionDataset(orig_images, attn_maps, transform=transform)

    # Check a sample
    sample_img, sample_mask = dataset[0]
    print(f"\nSample image shape: {sample_img.shape}, Sample mask shape: {sample_mask.shape}")
    print(f"Sample mask unique values: {torch.unique(sample_mask)}")
    
    # Calculate class weights for weighted sampling
    foreground_count = (attn_maps == 1).sum()
    background_count = (attn_maps == 0).sum()
    total_count = foreground_count + background_count
    
    # Prepare the split
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Use the improved UNet model
    model = UNet(n_channels=3, n_classes=1).to(device)
    
    # Use Adam optimizer with learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Use combined loss for better handling of class imbalance
    criterion = CombinedLoss(focal_weight=0.5, dice_weight=0.5)

    os.makedirs("checkpoint", exist_ok=True)
    
    # For early stopping
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    for epoch in range(14):  # More epochs for better learning
        # Training phase
        model.train()
        total_loss = 0
        metrics = {"dice": 0, "iou": 0, "acc": 0, "prec": 0, "rec": 0}
        batch_count = 0
        
        print(f"\nTraining epoch {epoch+1}...")
        for imgs, masks in tqdm(train_loader):
            imgs = imgs.to(device)
            masks = masks.to(device)

            preds = model(imgs).squeeze(1)  # Remove channel dimension to match masks

            loss = criterion(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            # Calculate metrics only on valid regions (not ignore_value)
            valid_masks = []
            valid_preds = []
            
            for i in range(masks.size(0)):
                valid = masks[i] != 255
                if valid.sum() > 0:
                    valid_masks.append(masks[i][valid])
                    valid_preds.append(preds[i][valid])
            
            if valid_masks:
                # Concatenate all valid regions for batch
                batch_valid_masks = torch.cat(valid_masks)
                batch_valid_preds = torch.cat(valid_preds)
                
                # Calculate metrics
                dice, iou, acc, prec, rec = compute_metrics(batch_valid_preds, batch_valid_masks)
                metrics["dice"] += dice
                metrics["iou"] += iou
                metrics["acc"] += acc
                metrics["prec"] += prec
                metrics["rec"] += rec
                batch_count += 1
        
        avg_metrics_divisor = max(batch_count, 1)  # Avoid division by zero
        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, "
              f"Dice={metrics['dice']/avg_metrics_divisor:.4f}, "
              f"IoU={metrics['iou']/avg_metrics_divisor:.4f}, "
              f"Accuracy={metrics['acc']/avg_metrics_divisor:.4f}, "
              f"Precision={metrics['prec']/avg_metrics_divisor:.4f}, "
              f"Recall={metrics['rec']/avg_metrics_divisor:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_metrics = {"dice": 0, "iou": 0, "acc": 0, "prec": 0, "rec": 0}
        val_batch_count = 0
        
        print("Validating...")
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader):
                imgs = imgs.to(device)
                masks = masks.to(device)
                
                preds = model(imgs).squeeze(1)
                batch_loss = criterion(preds, masks).item()
                val_loss += batch_loss
                
                # Calculate metrics only on valid regions
                valid_masks = []
                valid_preds = []
                
                for i in range(masks.size(0)):
                    valid = masks[i] != 255
                    if valid.sum() > 0:
                        valid_masks.append(masks[i][valid])
                        valid_preds.append(preds[i][valid])
                
                if valid_masks:
                    batch_valid_masks = torch.cat(valid_masks)
                    batch_valid_preds = torch.cat(valid_preds)
                    
                    dice, iou, acc, prec, rec = compute_metrics(batch_valid_preds, batch_valid_masks)
                    val_metrics["dice"] += dice
                    val_metrics["iou"] += iou
                    val_metrics["acc"] += acc 
                    val_metrics["prec"] += prec
                    val_metrics["rec"] += rec
                    val_batch_count += 1
        
        val_metrics_divisor = max(val_batch_count, 1)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Validation: Loss={avg_val_loss:.4f}, "
              f"Dice={val_metrics['dice']/val_metrics_divisor:.4f}, "
              f"IoU={val_metrics['iou']/val_metrics_divisor:.4f}, "
              f"Accuracy={val_metrics['acc']/val_metrics_divisor:.4f}, "
              f"Precision={val_metrics['prec']/val_metrics_divisor:.4f}, "
              f"Recall={val_metrics['rec']/val_metrics_divisor:.4f}")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save model checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_val_loss,
            'dice': val_metrics['dice']/val_metrics_divisor,
        }, f"checkpoint/unet_attnmaps_epoch_{epoch+1}.pth")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                'dice': val_metrics['dice']/val_metrics_divisor,
            }, "checkpoint/unet_attnmaps_best.pth")
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

if __name__ == "__main__":
    train()
