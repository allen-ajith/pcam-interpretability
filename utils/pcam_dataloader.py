import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.pcam_dataset import PCamHFDataset  # your new dataset class

# ImageNet normalization
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

# Data augmentation for training
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize from 96x96 to 224x224
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

# Validation/test transforms (no augmentation)
transform_val_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

def get_pcam_loaders(repo_id="allen-ajith/pcam-h5", batch_size=64, num_workers=4):
    """
    Creates DataLoaders for the PCam dataset (.h5 version hosted on Hugging Face).

    Args:
        repo_id (str): Hugging Face repo ID where .h5 files are hosted.
        batch_size (int): Number of images per batch.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        train_loader, val_loader, test_loader (torch.utils.data.DataLoader)
    """
    print(f"Loading PCam dataset from Hugging Face repo: {repo_id}")

    # Use PCamHFDataset from pcam_dataset.py
    train_dataset = PCamHFDataset(repo_id=repo_id, split='train', transform=transform_train)
    val_dataset = PCamHFDataset(repo_id=repo_id, split='val', transform=transform_val_test)
    test_dataset = PCamHFDataset(repo_id=repo_id, split='test', transform=transform_val_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"Train set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(val_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")
    print(f"Batch size: {batch_size}")

    return train_loader, val_loader, test_loader
