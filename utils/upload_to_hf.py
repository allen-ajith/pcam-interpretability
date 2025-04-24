from huggingface_hub import create_repo, upload_file, upload_folder
import os

def write_readme(model_name, val_acc, epochs=None, batch_size=None, learning_rate=None, save_dir="tmp_hf_upload"):
    """
    Creates a README.md file summarizing the training details.
    """
    val_acc_str = f"{val_acc:.4f}"
    os.makedirs(save_dir, exist_ok=True)
    readme_path = os.path.join(save_dir, "README.md")

    with open(readme_path, "w") as f:
        f.write(f"# {model_name}\n\n")
        f.write(f"**Best Validation Accuracy:** {val_acc_str}\n\n")
        f.write("## Training Configuration:\n")
        if epochs is not None:
            f.write(f"- Epochs: {epochs}\n")
        if batch_size is not None:
            f.write(f"- Batch size: {batch_size}\n")
        if learning_rate is not None:
            f.write(f"- Learning rate: {learning_rate}\n")

    return readme_path

def upload_model_to_hf(model_name, val_acc, checkpoint_dir="checkpoints",
                       epochs=None, batch_size=None, learning_rate=None,
                       commit_message="Upload best model weights"):
    """
    Uploads the best model checkpoint and README to Hugging Face Hub.

    Args:
        model_name (str): Model name (e.g., 'resnet50').
        val_acc (float): Best validation accuracy (e.g., 0.9367).
        checkpoint_dir (str): Directory where the checkpoint is saved.
        epochs (int): Training epochs (optional).
        batch_size (int): Batch size (optional).
        learning_rate (float): Learning rate (optional).
        commit_message (str): Commit message for the upload.
    """
    val_acc_str = f"{val_acc:.4f}".replace('.', '')
    repo_name = f"{model_name}-val{val_acc_str}"

    weights_path = os.path.join(checkpoint_dir, model_name, "best_model.pth")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint not found at {weights_path}")

    # Create repo if it doesn't exist
    create_repo(repo_name, exist_ok=True)

    # Prepare README.md
    temp_dir = "tmp_hf_upload"
    readme_path = write_readme(model_name, val_acc, epochs, batch_size, learning_rate, save_dir=temp_dir)

    # Upload model weights
    upload_file(
        path_or_fileobj=weights_path,
        path_in_repo=os.path.basename(weights_path),
        repo_id=repo_name,
        commit_message=commit_message
    )
    print(f"Uploaded model weights to: https://huggingface.co/{repo_name}")

    # Upload README.md
    upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo_name,
        commit_message="Add training summary"
    )
    print(f"Uploaded README.md to: https://huggingface.co/{repo_name}")

# if __name__ == "__main__":
#     # Example usage
#     model_name = "resnet50"
#     val_acc = 0.9367  

#     upload_model_to_hf(
#         model_name=model_name,
#         val_acc=val_acc,
#         epochs=20,
#         batch_size=64,
#         learning_rate=1e-4
#     )
