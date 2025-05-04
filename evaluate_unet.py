import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import h5py
from huggingface_hub import hf_hub_download
import os

from test_unet_pcam import UNet, convert_to_binary_masks  # Make sure to import from your training script

# ----------------------------
# Set Paths
# ----------------------------
MODEL_PATH = "checkpoints/unet_epoch_1.pth"  # trained model
SAVE_DIR = "eval_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

# ----------------------------
# Load Test Data
# ----------------------------
print("Loading test data...")

# Load original test images
test_path = hf_hub_download(
    repo_id="allen-ajith/pcam-h5",
    filename="pcam/camelyonpatch_level_2_split_test_x.h5",
    repo_type="dataset"
)
with h5py.File(test_path, "r") as f:
    test_images = f["x"][:]
print("Test images loaded:", test_images.shape)

# Load test Grad-CAM++ heatmaps
try:
    heatmap_path = hf_hub_download(
        repo_id="pcam-interpretability/pcam_heatmaps",
        filename="pcam_heatmaps_test_best.h5",
        repo_type="dataset"
    )
    with h5py.File(heatmap_path, "r") as f:
        cam_images = f["x"][:]
    binary_masks = convert_to_binary_masks(cam_images)
    print("Test heatmaps loaded:", binary_masks.shape)
except Exception as e:
    binary_masks = None
    print("Skipping ground truth mask loading:", str(e))


# ----------------------------
# Load Model
# ----------------------------
print("Loading model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ----------------------------
# Inference + Save Visuals
# ----------------------------
transform = T.ToTensor()

for idx in range(10):  # Evaluate first 10 samples (change as needed)
    img = test_images[idx]
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(input_tensor)
        pred = pred.squeeze().cpu().numpy()

    # Prepare visuals
    fig, axs = plt.subplots(1, 3 if binary_masks is not None else 2, figsize=(12, 4))

    axs[0].imshow(img)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(pred > 0.5, cmap="gray")
    axs[1].set_title("Predicted Mask")
    axs[1].axis("off")

    if binary_masks is not None:
        axs[2].imshow(binary_masks[idx], cmap="gray")
        axs[2].set_title("Pseudo Ground Truth")
        axs[2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"sample_{idx}.png"))
    plt.close()

print(f"Saved results to: {SAVE_DIR}")
