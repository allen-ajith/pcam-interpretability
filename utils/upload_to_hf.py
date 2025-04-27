from huggingface_hub import HfApi, upload_file
import os

def write_readme(model_name, val_acc, epochs=None, batch_size=None, learning_rate=None,
                 logs=None, metadata=None, save_dir="tmp_hf_upload"):
    val_acc_str = f"{val_acc:.4f}"
    os.makedirs(save_dir, exist_ok=True)
    readme_path = os.path.join(save_dir, "README.md")

    with open(readme_path, "w") as f:
        f.write(f"# {model_name}\n\n")
        f.write(f"**Best Validation Accuracy:** `{val_acc_str}`\n\n")

        if metadata:
            f.write("## Metadata\n")
            for key, value in metadata.items():
                f.write(f"- **{key.replace('_', ' ').title()}**: `{value}`\n")
            f.write("\n")

        f.write("## Training Configuration\n")
        if epochs is not None:
            f.write(f"- Epochs: `{epochs}`\n")
        if batch_size is not None:
            f.write(f"- Batch size: `{batch_size}`\n")
        if learning_rate is not None:
            f.write(f"- Learning rate (initial): `{learning_rate}`\n")

        if logs:
            f.write("\n## Training Logs (Per Epoch)\n")
            f.write("| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | LR |\n")
            f.write("|-------|------------|-----------|----------|---------|----|\n")
            for log in logs:
                f.write(f"| {log['epoch']} | {log['train_loss']:.4f} | {log['train_acc']:.4f} "
                        f"| {log['val_loss']:.4f} | {log['val_acc']:.4f} | {log['lr']:.6f} |\n")

    return readme_path

def upload_model_to_hf(model_name, val_acc, checkpoint_dir="checkpoints",
                       epochs=None, batch_size=None, learning_rate=None,
                       logs=None, metadata=None,
                       commit_message="Upload best model weights"):
    val_acc_str = f"{val_acc:.4f}".replace('.', '')
    org_name = "pcam-interpretability"
    run_id = metadata["run_id"] if metadata and "run_id" in metadata else "default"
    repo_name = f"{org_name}/{model_name}-val{val_acc_str}-{run_id}"

    weights_path = os.path.join(checkpoint_dir, "best_model.pth")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint not found at {weights_path}")

    api = HfApi()
    try:
        api.create_repo(
            repo_id=repo_name,
            repo_type="model",
            exist_ok=True,
        )
        print(f"Repo '{repo_name}' created or already exists.")
    except Exception as e:
        print(f"Repo creation error: {e}")
        return

    temp_dir = "tmp_hf_upload"
    readme_path = write_readme(
        model_name=model_name,
        val_acc=val_acc,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        logs=logs,
        metadata=metadata,
        save_dir=temp_dir
    )

    try:
        upload_file(
            path_or_fileobj=weights_path,
            path_in_repo=os.path.basename(weights_path),
            repo_id=repo_name,
            commit_message=commit_message,
            repo_type="model"
        )
        print(f"Uploaded model weights to: https://huggingface.co/{repo_name}")
    except Exception as e:
        print(f"Error uploading model weights: {e}")

    try:
        upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_name,
            commit_message="Add training summary",
            repo_type="model"
        )
        print(f"Uploaded README.md to: https://huggingface.co/{repo_name}")
    except Exception as e:
        print(f"Error uploading README.md: {e}")
