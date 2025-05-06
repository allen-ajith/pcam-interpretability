import os
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image
import glob
from tqdm import tqdm
import h5py
from huggingface_hub import hf_hub_download

# ----------------------------
# U-Net Model from your training script
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
        self.final = nn.Conv2d(64, 1, 1)  # Output is single channel

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        d1 = self.up(self.dec1(e2))
        out = self.final(d1)
        return torch.sigmoid(out)

# ----------------------------
# Visualization Functions
# ----------------------------
def load_model(model_path):
    """Load the trained U-Net model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = UNet().to(device)
    
    try:
        # Try loading state dict directly
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Direct loading failed: {e}")
        try:
            # Try loading from a checkpoint with more complex structure
            checkpoint = torch.load(model_path, map_location=device)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            elif "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                print("Could not find state_dict in checkpoint")
                return None
        except Exception as e2:
            print(f"Loading checkpoint failed: {e2}")
            return None
    
    model.eval()
    print(f"Successfully loaded model from {model_path}")
    return model

def preprocess_image(image_path, size=(224, 224)):
    """Preprocess an image for model input."""
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    return img, img_tensor

def visualize_prediction(original_img, prediction, save_path=None):
    """Visualize the model prediction alongside the original image."""
    plt.figure(figsize=(12, 5))
    
    # Display original image
    plt.subplot(1, 3, 1)
    plt.imshow(original_img)
    plt.title("Original Image")
    plt.axis('off')
    
    # Display prediction heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(prediction, cmap='viridis')
    plt.title("U-net Prediction")
    plt.axis('off')
    
    # Display overlay
    plt.subplot(1, 3, 3)
    plt.imshow(original_img)
    plt.imshow(prediction, alpha=0.6, cmap='viridis')
    plt.title("Overlay")
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()

def process_images(model, image_paths, output_dir="output", batch_size=4):
    """Process multiple images and save the visualizations."""
    device = next(model.parameters()).device
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process images in small batches to improve efficiency
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        batch_tensors = []
        
        # Load and preprocess batch
        for image_path in batch_paths:
            try:
                img, img_tensor = preprocess_image(image_path)
                batch_images.append(img)
                batch_tensors.append(img_tensor)
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
        
        if not batch_tensors:
            continue
            
        # Stack tensors and run inference
        batch_tensor = torch.cat(batch_tensors, dim=0).to(device)
        with torch.no_grad():
            batch_predictions = model(batch_tensor)
        
        # Process each prediction
        for j, (img, img_path, pred) in enumerate(zip(batch_images, batch_paths, batch_predictions)):
            try:
                # Convert prediction to numpy
                pred_np = pred.squeeze().cpu().numpy()
                
                # Create output filename
                base_filename = os.path.splitext(os.path.basename(img_path))[0]
                save_path = os.path.join(output_dir, f"{base_filename}_gradcam_viz.png")
                
                # Visualize and save
                visualize_prediction(img, pred_np, save_path)
            except Exception as e:
                print(f"Error processing result for {img_path}: {e}")

def load_pcam_test_data(num_samples=50):
    """Load test images from PCam dataset."""
    print("Downloading PCam test data from Hugging Face...")
    test_path = hf_hub_download(
        repo_id="allen-ajith/pcam-h5",
        filename="pcam/camelyonpatch_level_2_split_test_x.h5",
        repo_type="dataset"
    )
    
    with h5py.File(test_path, "r") as f:
        # Get a subset of test images
        test_images = f["x"][:num_samples]
    
    print(f"Loaded {len(test_images)} test images from PCam dataset")
    return test_images

def save_pcam_images(test_images, output_dir="pcam_images"):
    """Save PCam test images as individual files for processing."""
    os.makedirs(output_dir, exist_ok=True)
    image_paths = []
    
    for i, img_array in enumerate(tqdm(test_images, desc="Saving PCam images")):
        img_path = os.path.join(output_dir, f"pcam_test_{i:03d}.png")
        Image.fromarray(img_array).save(img_path)
        image_paths.append(img_path)
    
    return image_paths

def main():
    # Path to model weights
    model_path = "checkpoints/unet_gradcam_epoch_10.pth"
    
    # Check if model exists
    if not os.path.exists(model_path):
        available_models = glob.glob("unet_gradcam_epoch_*.pth")
        if available_models:
            model_path = available_models[0]
            print(f"Using available model: {model_path}")
        else:
            print(f"Error: Model file {model_path} not found.")
            return
    
    # Load model
    model = load_model(model_path)
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    print("\nSelect data source:")
    print("1. Download PCam test data")
    print("2. Use local image files")
    choice = input("Enter your choice (1 or 2): ")
    
    image_paths = []
    
    if choice == "1":
        # Download and use PCam test images
        num_samples = input("Enter number of PCam test samples to visualize (default 50): ")
        num_samples = int(num_samples) if num_samples.isdigit() else 50
        
        # Load test images from PCam
        test_images = load_pcam_test_data(num_samples)
        
        # Save as PNG files for processing
        temp_dir = "pcam_temp_images"
        image_paths = save_pcam_images(test_images, temp_dir)
        
    else:
        # Use local image files
        image_dir = input("Enter directory containing images to process (leave empty for current directory): ")
        if not image_dir:
            image_dir = "."
        
        if not os.path.isdir(image_dir):
            print(f"Error: {image_dir} is not a valid directory")
            return
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(image_dir, f"*{ext}")))
            image_paths.extend(glob.glob(os.path.join(image_dir, f"*{ext.upper()}")))
        
        if not image_paths:
            print(f"No image files found in {image_dir}")
            return
    
    print(f"Found {len(image_paths)} images to process.")
    
    # Create output directory
    output_dir = input("Enter output directory (leave empty for 'gradcam_output'): ")
    if not output_dir:
        output_dir = "gradcam_output"
    
    # Process images
    process_images(model, image_paths, output_dir)
    
    print(f"Finished processing. Visualizations saved in {output_dir}")

if __name__ == "__main__":
    main()