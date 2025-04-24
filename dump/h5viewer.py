import torch
from torchvision.datasets import PCAM
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Label map
label_map = {0: 'Normal', 1: 'Metastatic'}

# Transform: just convert to tensor (values in [0, 1])
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load PCAM dataset (adjust path if needed)
dataset = PCAM(root='data', split='train', transform=transform, download=False)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Show a single image with normalization for visibility
def show_image(img_tensor, label, index=None):
    img = img_tensor.permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-5)  # normalize to [0, 1]

    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    title = f"Label: {label_map[label.item()]} ({label.item()})"
    if index is not None:
        title = f"Image {index} - " + title
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Load first batch
images, labels = next(iter(dataloader))
print(f"Loaded batch with {len(images)} images.")

# Show all images in a grid
def show_batch(images, labels):
    plt.figure(figsize=(16, 4))
    for i in range(len(images)):
        img = images[i].permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)
        plt.subplot(2, 8, i + 1)
        plt.imshow(img)
        plt.title(label_map[labels[i].item()])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Display the whole batch first
show_batch(images, labels)

# Interactive image selector
while True:
    try:
        idx = input(f"\nEnter image index (0 to {len(images)-1}) to view individually, or 'q' to quit: ")
        if idx.lower() == 'q':
            break
        idx = int(idx)
        if 0 <= idx < len(images):
            show_image(images[idx], labels[idx], index=idx)
        else:
            print("Index out of range.")
    except Exception as e:
        print(f"Invalid input: {e}")
