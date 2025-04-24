# pcam_dataset.py

import h5py
import torch
from torch.utils.data import Dataset
from huggingface_hub import hf_hub_download

class PCamHFDataset(Dataset):
    def __init__(self, repo_id: str, split: str = "train", transform=None):
        """
        PyTorch Dataset for PatchCamelyon .h5 files hosted on Hugging Face Hub.

        Args:
            repo_id (str): Hugging Face dataset repo (e.g., 'allen-ajith/pcam-h5')
            split (str): One of 'train', 'val', or 'test'
            transform (callable, optional): Optional transform to apply on the image
        """
        self.transform = transform
        self.split = split
        self.repo_id = repo_id

        # Correct filenames with the subfolder 'pcam/'
        x_file = f"pcam/camelyonpatch_level_2_split_{split}_x.h5"
        y_file = f"pcam/camelyonpatch_level_2_split_{split}_y.h5"

        # FIX: Added repo_type="dataset" here:
        self.x_path = hf_hub_download(repo_id=repo_id, filename=x_file, repo_type="dataset")
        self.y_path = hf_hub_download(repo_id=repo_id, filename=y_file, repo_type="dataset")

        # Open HDF5 datasets
        self.x_data = h5py.File(self.x_path, "r")["x"]
        self.y_data = h5py.File(self.y_path, "r")["y"]

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        image = self.x_data[idx]  # shape: (96, 96, 3)
        label = int(self.y_data[idx][0])  # stored as array([0]) or array([1])

        if self.transform:
            image = self.transform(image)
        else:
            # Default: Normalize [0, 255] → [0, 1], permute HWC → CHW
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        return image, torch.tensor(label, dtype=torch.long)
