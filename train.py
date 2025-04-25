import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse

from utils.pcam_dataloader import get_pcam_loaders
from models.resnet50 import create_resnet50
from models.swin_tiny import create_swin_tiny
from models.dino_vit import create_dino_vit
from utils.upload_to_hf import upload_model_to_hf


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def get_optimizer(model, optimizer_name, lr):
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr)
    elif optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def get_scheduler(optimizer, scheduler_name, epochs):
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif scheduler_name == "none":
        return None
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

def train_model(model, train_loader, val_loader, model_name, epochs, lr, optimizer_name, scheduler_name, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = get_optimizer(model, optimizer_name, lr)
    scheduler = get_scheduler(optimizer, scheduler_name, epochs)
    criterion = nn.BCEWithLogitsLoss()

    save_dir = os.path.join("checkpoints", model_name)
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_model.pth")

    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}")

        if scheduler:
            scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model to {best_model_path} with Val Acc: {val_acc:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs. Best Val Acc: {best_val_acc:.4f}")
            break

    print("Training complete.")
    return best_val_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCam Training Script with HF Upload")
    parser.add_argument('--model_name', type=str, required=True, choices=["resnet50", "swin-tiny", "dino-vits16"])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='adamw', choices=["adam", "adamw", "sgd"])
    parser.add_argument('--scheduler', type=str, default='cosine', choices=["cosine", "step", "none"])
    parser.add_argument('--patience', type=int, default=5, help="Early stopping patience")
    args = parser.parse_args()

    train_loader, val_loader, test_loader = get_pcam_loaders(batch_size=args.batch_size)
    print(f"Train set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(val_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")

    if args.model_name == "resnet50":
        model = create_resnet50(pretrained=True)
    elif args.model_name == "swin-tiny":
        model = create_swin_tiny(pretrained=True)
    elif args.model_name == "dino-vits16":
        model = create_dino_vit(pretrained=True)

    best_val_acc = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_name=args.model_name,
        epochs=args.epochs,
        lr=args.lr,
        optimizer_name=args.optimizer,
        scheduler_name=args.scheduler,
        patience=args.patience
    )

    upload_model_to_hf(
        model_name=args.model_name,
        val_acc=best_val_acc,
        checkpoint_dir="checkpoints",
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
