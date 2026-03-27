from __future__ import annotations

from time import time

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from .common import identity_transform, log_knn_time
from .utils import find_knn_indices, load_knn_cache, resolve_knn_cache_path, save_knn_cache


class MnistWithTshirtDataset(Dataset):
    def __init__(self, data_dir, filter, remove_img_name=None, transform=None, K=None):
        remove_token = remove_img_name or "tshirt"
        self.data = load_dataset(data_dir, split="train")
        self.transform = transform or identity_transform()
        self.is_knn = False

        if filter == "all":
            return
        if filter == "deletion":
            self.data = self.data.filter(lambda item: item["label"] == 10)
            return
        if filter == "remain":
            self.data = self.data.filter(lambda item: item["label"] != 10)
            return
        if filter != "knn":
            raise ValueError(f"Invalid filter: {filter}")
        if K is None:
            raise ValueError("KNN filter requires K.")

        self.is_knn = True
        cache_path = resolve_knn_cache_path(data_dir, "mnist_with_tshirt", remove_token, K, self.transform)
        cached = load_knn_cache(cache_path)
        if cached is not None:
            self.data_remain = cached["data_remain"]
            self.data_deletion = cached["data_deletion"]
            self.knn_indices = cached["knn_indices"]
            return

        labels = self.data["label"]
        remain_indices = [index for index, label in enumerate(labels) if label != 10]
        deletion_indices = [index for index, label in enumerate(labels) if label == 10]
        remain = self.data.select(remain_indices)
        deletion = self.data.select(deletion_indices)
        self.data_remain = torch.stack([self.transform(remain[i]["image"]) for i in range(len(remain))])
        self.data_deletion = torch.stack([self.transform(deletion[i]["image"]) for i in range(len(deletion))])
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
            return self.transform(self.data[int(idx)]["image"])
        return self.data_deletion[idx], self.data_remain[self.knn_indices[idx]]
