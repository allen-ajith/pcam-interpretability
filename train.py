import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
import time
from datetime import datetime

from utils.pcam_dataloader import get_pcam_loaders
from models.resnet50 import create_resnet50
from models.swin_tiny import create_swin_tiny
from models.dino_vit import create_dino_vit
from utils.upload_to_hf import upload_model_to_hf

def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Seed set to {seed} for reproducibility.")

def train_one_epoch(model, loader, criterion, optimizer, device, scaler, use_amp, model_name=None):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="Training", leave=True)
    for images, labels in pbar:
        images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        if model_name is not None and "vit" in model_name.lower():
            scaler.unscale_(optimizer) 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device, use_amp):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="Validation", leave=True)
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def get_optimizer(model, optimizer_name, lr, weight_decay):
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(model, train_loader, val_loader, model_name, epochs, lr, optimizer_name, scheduler_name, weight_decay, warmup_epochs, patience=5, use_amp=False, run_name=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = get_optimizer(model, optimizer_name, lr, weight_decay)
    scheduler = get_scheduler(optimizer, scheduler_name, epochs)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # === Checkpoint Directory Logic ===
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = run_name if run_name else timestamp
    save_dir = os.path.join("checkpoints", f"{model_name}_{run_id}")
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_model.pth")

    best_val_acc = 0.0
    epochs_no_improve = 0
    logs = []
    start_time = time.time()

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        if warmup_epochs > 0 and epoch < warmup_epochs:
            warmup_lr = lr * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            print(f"Warmup LR: {warmup_lr:.6f}")
        else:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current LR: {current_lr:.6f}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp)
        val_loss, val_acc = validate(model, val_loader, criterion, device, use_amp)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}")

        logs.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]['lr']
        })

        if scheduler and epoch >= warmup_epochs:
            scheduler.step()
            print(f"LR after scheduler step: {optimizer.param_groups[0]['lr']:.6f}")

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

    total_time = time.time() - start_time

    metadata = {
        "model_name": model_name,
        "optimizer": optimizer_name,
        "scheduler": scheduler_name,
        "weight_decay": weight_decay,
        "warmup_epochs": warmup_epochs,
        "patience": patience,
        "amp": use_amp,
        "seed": 42,
        "batch_size": train_loader.batch_size,
        "initial_lr": lr,
        "total_epochs_ran": epoch + 1,
        "early_stopped": epochs_no_improve >= patience,
        "training_time_seconds": total_time,
        "num_parameters": count_parameters(model),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "run_id": run_id
    }

    return best_val_acc, epoch + 1, logs, metadata, save_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCam Training Script with HF Upload")
    parser.add_argument('--model_name', type=str, required=True, choices=["resnet50", "swin-tiny", "dino-vits16"])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='adamw', choices=["adam", "adamw", "sgd"])
    parser.add_argument('--scheduler', type=str, default='cosine', choices=["cosine", "step", "none"])
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--warmup_epochs', type=int, default=0)
    parser.add_argument('--amp', action='store_true', help="Enable mixed precision training")
    parser.add_argument('--run_name', type=str, default=None, help="Optional run name for checkpoint directory")
    args = parser.parse_args()

    set_seed(42)

    if args.model_name == "resnet50":
        model = create_resnet50(pretrained=True)
        model_type = "resnet"
    elif args.model_name == "swin-tiny":
        model = create_swin_tiny(pretrained=True)
        model_type = "vit"
    elif args.model_name == "dino-vits16":
        model = create_dino_vit(pretrained=True)
        model_type = "vit"

    train_loader, val_loader, test_loader = get_pcam_loaders(
        batch_size=args.batch_size,
        model_type=model_type,
        seed=42
    )

    print(f"Train set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(val_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")

    best_val_acc, actual_epochs, logs, metadata, save_dir = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_name=args.model_name,
        epochs=args.epochs,
        lr=args.lr,
        optimizer_name=args.optimizer,
        scheduler_name=args.scheduler,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        use_amp=args.amp,
        run_name=args.run_name
    )

    upload_model_to_hf(
        model_name=args.model_name,
        val_acc=best_val_acc,
        checkpoint_dir=save_dir,
        epochs=actual_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        logs=logs,
        metadata=metadata
    )
