import h5py
import torch
from torch.utils.data import Dataset
from huggingface_hub import hf_hub_download
from PIL import Image
import numpy as np

class PCamHFDataset(Dataset):
    def __init__(self, repo_id: str, split: str = "train", transform=None):
        """
        PyTorch Dataset for PatchCamelyon .h5 files hosted on Hugging Face Hub.

        Args:
            repo_id (str): Hugging Face dataset repo (e.g., 'allen-ajith/pcam-h5')
            split (str): One of 'train', 'val', or 'test'
            transform (callable, optional): Optional transform to apply on the image
        """
        self.repo_id = repo_id
        self.split = split
        self.transform = transform

        # Map 'val' → 'valid' for compatibility with Hugging Face dataset
        split_map = {'train': 'train', 'val': 'valid', 'test': 'test'}
        split_key = split_map[self.split]

        x_file = f"pcam/camelyonpatch_level_2_split_{split_key}_x.h5"
        y_file = f"pcam/camelyonpatch_level_2_split_{split_key}_y.h5"

        # Download .h5 files from HF Hub 
        self.x_path = hf_hub_download(repo_id=self.repo_id, filename=x_file, repo_type="dataset")
        self.y_path = hf_hub_download(repo_id=self.repo_id, filename=y_file, repo_type="dataset")

        # Open HDF5 datasets
        self.x_data = h5py.File(self.x_path, "r")["x"]
        self.y_data = h5py.File(self.y_path, "r")["y"]

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        image = self.x_data[idx]  # shape: (96, 96, 3), NumPy array
        label = int(self.y_data[idx][0])  # label stored as array([0]) or array([1])

        if self.transform:
            # Convert NumPy array to PIL Image before applying transforms
            image = Image.fromarray(image.astype('uint8'))
            image = self.transform(image)
        else:
            # Default transform: normalize to [0, 1] and permute HWC → CHW
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        return image, torch.tensor(label, dtype=torch.long)
