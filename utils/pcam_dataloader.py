import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]


transform_train = transforms.Compose([
    transforms.Resize((224, 224)),                       # Resize from 96x96 to 224x224
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

transform_val_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

def get_pcam_loaders(root_dir='./data/pcam', batch_size=64, num_workers=4, download=True):
    """
    Creates DataLoaders for the PCam dataset (.h5 version) using torchvision.datasets.PCAM.

    Args:
        root_dir (str): Directory where PCam files will be stored or downloaded.
        batch_size (int): Number of images per batch.
        num_workers (int): Number of subprocesses for data loading.
        download (bool): Whether to download the PCam .h5 files if not found.

    Returns:
        train_loader, val_loader, test_loader (torch.utils.data.DataLoader): DataLoaders for each split.
    """
    print(f"Loading PCam dataset from: {root_dir} (download={download})")

    train_dataset = datasets.PCAM(root=root_dir, split='train', transform=transform_train, download=download)
    val_dataset = datasets.PCAM(root=root_dir, split='val', transform=transform_val_test, download=download)
    test_dataset = datasets.PCAM(root=root_dir, split='test', transform=transform_val_test, download=download)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"Train set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(val_loader.dataset)}")    
    print(f"Test set size: {len(test_loader.dataset)}")
    print(f"Batch size: {batch_size}")

    return train_loader, val_loader, test_loader
