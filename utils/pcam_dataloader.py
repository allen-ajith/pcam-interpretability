import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import RandAugment
from utils.pcam_dataset import PCamHFDataset  

# ImageNet normalization
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

def get_pcam_loaders(
    repo_id="allen-ajith/pcam-h5",
    batch_size=64,
    num_workers=4,
    repo_type="dataset",
    model_type="resnet",
    seed=None
):
    """
    Creates DataLoaders for the PCam dataset (.h5 version hosted on Hugging Face) with model-aware augmentations.

    Args:
        repo_id (str): Hugging Face repo ID where .h5 files are hosted.
        batch_size (int): Number of images per batch.
        num_workers (int): Number of subprocesses for data loading.
        repo_type (str): Hugging Face repo type (default 'dataset').
        model_type (str): "resnet" or "vit". Controls augmentation strength.
        seed (int or None): Optional seed for reproducibility.

    Returns:
        train_loader, val_loader, test_loader (torch.utils.data.DataLoader)
    """
    print(f"Loading PCam dataset from Hugging Face repo: {repo_id}")
    print(f"Model type for augmentation: {model_type}")

    if model_type not in ["resnet", "vit"]:
        raise ValueError(f"Invalid model_type '{model_type}'. Choose 'resnet' or 'vit'.")

    # Set global reproducibility if seed is provided
    generator = torch.Generator()
    if seed is not None:
        print(f"Using seed {seed} for reproducibility.")
        torch.manual_seed(seed)
        generator.manual_seed(seed)

    # Model-aware augmentations
    if model_type == "vit":
            transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),  
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomApply([RandAugment()], p=0.5),      
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3))  
        ])

    else:  # ResNet or CNN-like
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3)) 
        ])

    # Validation/test transforms (no augmentation)
    transform_val_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    # Use PCamHFDataset from pcam_dataset.py
    train_dataset = PCamHFDataset(repo_id=repo_id, split='train', transform=transform_train)
    val_dataset = PCamHFDataset(repo_id=repo_id, split='val', transform=transform_val_test)
    test_dataset = PCamHFDataset(repo_id=repo_id, split='test', transform=transform_val_test)

    common_loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "persistent_workers": (num_workers > 0)
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator if seed is not None else None,
        **common_loader_args
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **common_loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_loader_args)

    print(f"Train set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(val_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")
    print(f"Batch size: {batch_size}")

    return train_loader, val_loader, test_loader
