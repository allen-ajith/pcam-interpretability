import torch
import os
from huggingface_hub import hf_hub_download
from models.resnet50 import create_resnet50
from models.swin_tiny import create_swin_tiny
from models.dino_vit import create_dino_vit

def load_model_from_hf(model_name, repo_id, checkpoint_filename="best_model.pth", device=None):
    """
    Loads the trained model weights from HF Hub and returns the model ready for inference.

    Args:
        model_name (str): One of "resnet50", "swin-tiny", "dino-vits16".
        repo_id (str): Hugging Face repo ID (e.g., "pcam-interpretability/resnet50-val8800").
        checkpoint_filename (str): The name of the checkpoint file.
        device (torch.device or str): Device to load the model onto.

    Returns:
        model (torch.nn.Module): Loaded model with weights.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = hf_hub_download(repo_id=repo_id, filename=checkpoint_filename, repo_type="model")
    print(f"Checkpoint downloaded to: {checkpoint_path}")

    if model_name == "resnet50":
        model = create_resnet50(pretrained=False, freeze_early=False) 
    elif model_name == "swin-tiny":
        model = create_swin_tiny(pretrained=False)
    elif model_name == "dino-vits16":
        model = create_dino_vit(pretrained=False)
    else:
        raise ValueError(f"Unsupported model_name '{model_name}'. Must be one of 'resnet50', 'swin-tiny', or 'dino-vits16'.")

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Model '{model_name}' loaded from HF and ready on device: {device}")
    return model

if __name__ == "__main__":

    model_name = "resnet50"
    repo_id = "pcam-interpretability/resnet50-val8800"  
    model = load_model_from_hf(model_name, repo_id)
