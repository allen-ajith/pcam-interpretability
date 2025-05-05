import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from PIL import Image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# ----------------------------
# Load Grad-CAM and Attention Data
# ----------------------------
def load_attention_maps(paths):
    all_attns, all_logits = [], []
    for path in paths:
        with h5py.File(path, 'r') as f:
            all_attns.append(f['y'][:])
            all_logits.append(f['logits'][:])
    attns = np.concatenate(all_attns, axis=0)
    logits = np.concatenate(all_logits, axis=0)
    return attns, logits.squeeze()

def load_gradcam_heatmaps(path):
    with h5py.File(path, 'r') as f:
        return f['x'][:]

# ----------------------------
# Filter Correct Predictions
# ----------------------------
def filter_correct_predictions(attn_maps, gradcam_maps, logits, labels, threshold=0.5):
    preds = (logits >= threshold).astype(int)
    correct_idx = np.where(preds == labels)[0]
    attn_maps = attn_maps[correct_idx]
    gradcam_maps = gradcam_maps[correct_idx]
    return attn_maps, gradcam_maps

# ----------------------------
# Custom Dataset
# ----------------------------
class PCamDualInputDataset(Dataset):
    def __init__(self, gradcams, attn_maps, transform=None):
        self.gradcams = gradcams
        self.attn_maps = attn_maps
        self.transform = transform

    def __len__(self):
        return len(self.gradcams)

    def __getitem__(self, idx):
        gradcam = Image.fromarray(self.gradcams[idx].astype(np.uint8))
        attn = Image.fromarray((self.attn_maps[idx] * 255).astype(np.uint8))
        if self.transform:
            gradcam = self.transform(gradcam)
            attn = self.transform(attn)
        return gradcam, attn

# ----------------------------
# U-Net (Single Channel Input)
# ----------------------------
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.enc1 = CBR(1, 64)
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
# Training Function
# ----------------------------
def train():
    gradcam_path = "pcam_heatmaps_train_1.h5"
    attn_paths = [
        "pcam_attn_train_part0_87381.h5",
        "pcam_attn_train_part87381_174762.h5",
        "pcam_attn_train_part174762_262144.h5"
    ]

    gradcams = load_gradcam_heatmaps(gradcam_path)
    attn_maps, logits = load_attention_maps(attn_paths)

    labels = np.load("pcam_labels_train.npy")  # Assumes you have saved binary labels from classifier
    attn_maps, gradcams = filter_correct_predictions(attn_maps, gradcams, logits, labels)

    transform = T.Compose([T.Grayscale(), T.Resize((96, 96)), T.ToTensor()])
    dataset = PCamDualInputDataset(gradcams, attn_maps, transform)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    for epoch in range(5):
        model.train()
        total_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1} - Train"):
            x, y = x.to(device), y.to(device).float() / 255.0
            preds = model(x)
            loss = criterion(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Train Loss = {total_loss/len(train_loader):.4f}")

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for x, y in val_loader:
                x, y = x.to(device), y.to(device).float() / 255.0
                preds = model(x)
                loss = criterion(preds, y)
                val_loss += loss.item()
            print(f"Epoch {epoch+1}: Val Loss = {val_loss/len(val_loader):.4f}")

        torch.save(model.state_dict(), f"checkpoints/unet_epoch{epoch+1}.pth")
        print(f"Model saved for epoch {epoch+1}")

        # Visualize predictions
        x, y = next(iter(val_loader))
        x = x.to(device)
        preds = model(x).cpu().detach().numpy()
        for i in range(3):
            fig, axs = plt.subplots(1, 3, figsize=(10, 3))
            axs[0].imshow(x[i].cpu().squeeze(), cmap='gray')
            axs[0].set_title('Input')
            axs[1].imshow(y[i].cpu().squeeze(), cmap='gray')
            axs[1].set_title('Target')
            axs[2].imshow(preds[i].squeeze(), cmap='hot')
            axs[2].set_title('Predicted')
            plt.show()

if __name__ == "__main__":
    train()
