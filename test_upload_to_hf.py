import torch
import os
from models.resnet50 import create_resnet50
from models.swin_tiny import create_swin_tiny
from models.dino_vit import create_dino_vit
from utils.upload_to_hf import upload_model_to_hf

def save_dummy_model_weights(model, model_name, checkpoint_dir="checkpoints"):
    save_path = os.path.join(checkpoint_dir, model_name, "best_model.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f" Saved pretrained weights (no training) for {model_name} at {save_path}")
    return save_path

def test_upload(model_name, val_acc=0.9123, epochs=0, batch_size=64, lr=1e-4):
    checkpoint_dir = "checkpoints"
    
    # Load pretrained models
    if model_name == "resnet50":
        model = create_resnet50(pretrained=True)
    elif model_name == "swin-tiny":
        model = create_swin_tiny(pretrained=True)
    elif model_name == "dino-vitb16":
        model = create_dino_vit(pretrained=True)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Save the weights (mimicking trained checkpoint)
    save_dummy_model_weights(model, model_name, checkpoint_dir=checkpoint_dir)

    # Use your existing upload function
    upload_model_to_hf(
        model_name=model_name,
        val_acc=val_acc,  # Dummy val_acc
        checkpoint_dir=checkpoint_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        commit_message="Test upload with pretrained weights (no training)"
    )

if __name__ == "__main__":
    # Test all three models
    for model in ["resnet50", "swin-tiny", "dino-vitb16"]:
        print(f"\n=== Testing upload for: {model} ===")
        test_upload(model_name=model)
