import torch
import argparse
from utils.load_model_from_hf import load_model_from_hf

def test_load_and_forward(model_name, repo_id):
    print(f"\n=== Testing load from HF for model: {model_name} ===")
    
    # Load model from HF
    model = load_model_from_hf(model_name=model_name, repo_id=repo_id)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dummy input based on model type
    dummy_input = torch.randn(1, 3, 224, 224).to(device)  # Both ResNet-50 and ViT-S/16 use 224x224

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    print("Forward pass successful. Output shape:", output.shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test loading a model from Hugging Face Hub and run a dummy forward pass.")
    parser.add_argument("--model_name", type=str, required=True, choices=["resnet50", "dino-vits16"],
                        help="Model name: 'resnet50' or 'dino-vits16'")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face repo ID (e.g., pcam-interpretability/resnet50-val08809)")
    args = parser.parse_args()

    test_load_and_forward(args.model_name, args.repo_id)
