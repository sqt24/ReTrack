from __future__ import annotations

from time import time

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from .common import identity_transform, log_knn_time
from .utils import find_knn_indices, load_knn_cache, resolve_knn_cache_path, save_knn_cache


class Cifar10Dataset(Dataset):
    def __init__(self, data_dir, filter, remove_img_name=None, transform=None, K=None):
        self.data = load_dataset(data_dir, split="train")
        self.transform = transform or identity_transform()
        self.is_knn = False

        if filter == "all":
            return
        if filter == "deletion":
            if remove_img_name is None:
                raise ValueError("Deletion filter requires removal index.")
            self.data = self.data.select([remove_img_name])
            return
        if filter == "remain":
            if remove_img_name is None:
                raise ValueError("Remain filter requires removal index.")
            self.data = self.data.select([i for i in range(len(self.data)) if i != remove_img_name])
            return
        if filter != "knn":
            raise ValueError(f"Invalid filter: {filter}")
        if K is None or remove_img_name is None:
            raise ValueError("KNN filter requires both K and remove_img_name.")

        self.is_knn = True
        cache_path = resolve_knn_cache_path(data_dir, "cifar10", remove_img_name, K, self.transform)
        cached = load_knn_cache(cache_path)
        if cached is not None:
            self.data_remain = cached["data_remain"]
            self.data_deletion = cached["data_deletion"]
            self.knn_indices = cached["knn_indices"]
            return

        remain = self.data.select([i for i in range(len(self.data)) if i != remove_img_name])
        deletion = self.data.select([remove_img_name])
        self.data_remain = torch.stack([self.transform(remain[i]["img"]) for i in range(len(remain))])
        self.data_deletion = torch.stack([self.transform(deletion[i]["img"]) for i in range(len(deletion))])
        start_time = time()
        self.knn_indices = find_knn_indices(self.data_remain, self.data_deletion, K)
        log_knn_time(start_time)
        save_knn_cache(
            cache_path,
            {
                "data_remain": self.data_remain,
                "data_deletion": self.data_deletion,
                "knn_indices": self.knn_indices,
            },
        )

    def __len__(self):
        return len(self.data) if not self.is_knn else self.knn_indices.shape[0]

    def __getitem__(self, idx):
        if not self.is_knn:
            return self.transform(self.data[int(idx)]["img"])
        return self.data_deletion[idx], self.data_remain[self.knn_indices[idx]]
