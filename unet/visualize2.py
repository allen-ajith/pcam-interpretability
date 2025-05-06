import os
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as T
from tqdm import tqdm
import h5py
from huggingface_hub import hf_hub_download
from PIL import Image

# ----------------------------
# U-Net Definition (3-channel input)
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

        self.enc1 = CBR(3, 64)  # 3-channel input
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
# Load model
# ----------------------------
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = UNet().to(device)

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint
    model.load_state_dict(state)
    model.eval()
    print(f"Model loaded from {model_path}")
    return model

# ----------------------------
# Load attention maps
# ----------------------------
def load_attention_maps():
    attn_path = hf_hub_download(
        repo_id="pcam-interpretability/dino_vit_attnmaps",
        filename="pcam_attn_test_part0_32768.h5",
        repo_type="dataset"
    )
    with h5py.File(attn_path, "r") as f:
        attn_maps = f["y"][:]
    print(f"Loaded {len(attn_maps)} attention maps")
    return attn_maps

# ----------------------------
# Preprocess attention map
# ----------------------------
def preprocess_attention_map(attn_map):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    # Resize to 224x224
    attn_resized = Image.fromarray((attn_map * 255).astype(np.uint8)).resize((224, 224))
    attn_rgb = np.stack([np.array(attn_resized)] * 3, axis=-1)
    attn_image = Image.fromarray(attn_rgb)

    attn_tensor = transform(attn_image).unsqueeze(0)  # (1, 3, 224, 224)
    return attn_tensor, attn_image

# ----------------------------
# Visualization
# ----------------------------
def visualize_prediction(attn_img, prediction, save_path=None):
    plt.figure(figsize=(12, 5))

    # Original attention map
    plt.subplot(1, 3, 1)
    plt.imshow(attn_img)
    plt.title("Input Attention Map")
    plt.axis('off')

    # Prediction mask
    plt.subplot(1, 3, 2)
    plt.imshow(prediction, cmap='viridis')
    plt.title("U-net Predicted Mask")
    plt.axis('off')

    # Overlay
    plt.subplot(1, 3, 3)
    plt.imshow(attn_img)
    plt.imshow(prediction, cmap='viridis', alpha=0.5)
    plt.title("Overlay")
    plt.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved to {save_path}")
    else:
        plt.show()
    plt.close()

# ----------------------------
# Main function
# ----------------------------
def main():
    model_path = "checkpoints/unet_attnmaps_epoch_10.pth"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return

    print("Loading model...")
    model = load_model(model_path)
    device = next(model.parameters()).device

    print("Loading attention maps...")
    attn_maps = load_attention_maps()

    os.makedirs("attn_output", exist_ok=True)

    for i, attn_map in enumerate(tqdm(attn_maps[:50], desc="Visualizing attention predictions")):
        attn_tensor, attn_image = preprocess_attention_map(attn_map)

        with torch.no_grad():
            pred = model(attn_tensor.to(device))
            pred_np = pred.squeeze().cpu().numpy()

        save_path = os.path.join("attn_output", f"attn_pred_{i:03d}.png")
        visualize_prediction(attn_image, pred_np, save_path)

    print("Visualization complete. Saved to 'attn_output/'.")

if __name__ == "__main__":
    main()
