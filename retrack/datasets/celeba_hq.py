from __future__ import annotations

import os
from time import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .common import identity_transform, log_knn_time
from .utils import find_knn_indices, load_knn_cache, resolve_knn_cache_path, save_knn_cache


class CelebAHQDataset(Dataset):
    def __init__(self, data_dir, filter, remove_img_name=None, transform=None, K=None):
        self.data_dir = data_dir
        self.transform = transform or identity_transform()
        self.is_knn = False

        image_files = sorted([name for name in os.listdir(data_dir) if name.endswith(".jpg")])
        if filter == "all":
            self.image_files = image_files
            return
        if filter == "deletion":
            if remove_img_name is None:
                raise ValueError("Deletion filter requires remove_img_name.")
            self.image_files = [remove_img_name]
            return
        if filter == "remain":
            if remove_img_name is None:
                raise ValueError("Remain filter requires remove_img_name.")
            self.image_files = [name for name in image_files if name != remove_img_name]
            return
        if filter != "knn":
            raise ValueError(f"Invalid filter: {filter}")
        if K is None or remove_img_name is None:
            raise ValueError("KNN filter requires both K and remove_img_name.")

        self.is_knn = True
        self.image_files_deletion = remove_img_name
        self.image_files_remain = [name for name in image_files if name != remove_img_name]
        cache_path = resolve_knn_cache_path(data_dir, "celeba_hq", remove_img_name, K, self.transform)
        cached = load_knn_cache(cache_path)
        if cached is not None:
            self.knn_indices = cached["knn_indices"]
            deletion_image = Image.open(os.path.join(data_dir, remove_img_name))
            self.deletion_image = self.transform(deletion_image)
            return

        deletion_image = Image.open(os.path.join(data_dir, remove_img_name))
        self.deletion_image = self.transform(deletion_image)
        data_deletion = torch.stack([self.deletion_image])
        data_remain = np.array([self.transform(Image.open(os.path.join(data_dir, name))) for name in self.image_files_remain])
        start_time = time()
        self.knn_indices = find_knn_indices(data_remain, data_deletion, K)
        log_knn_time(start_time)
        save_knn_cache(cache_path, {"knn_indices": self.knn_indices})

    def __len__(self):
        return len(self.image_files) if not self.is_knn else self.knn_indices.shape[0]

    def __getitem__(self, idx):
        if not self.is_knn:
            image = Image.open(os.path.join(self.data_dir, self.image_files[idx]))
            return self.transform(image)
        knn_neighbors = [Image.open(os.path.join(self.data_dir, self.image_files_remain[i])) for i in self.knn_indices[idx]]
        return self.deletion_image, torch.stack([self.transform(neighbor) for neighbor in knn_neighbors])
