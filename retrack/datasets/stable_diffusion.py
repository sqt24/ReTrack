from __future__ import annotations

import json
import os
from time import time

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

from .common import identity_transform, log_knn_time
from .utils import find_knn_indices, load_knn_cache, resolve_knn_cache_path, save_knn_cache


class SDDataset(Dataset):
    def __init__(self, data_dir, filter, vae=None, remove_img_name=None, transform=None, K=None):
        if vae is None:
            raise ValueError("VAE must be provided for SDDataset.")
        if remove_img_name is None:
            raise ValueError("remove_img_name must be provided for SDDataset.")
        self.images_dir = os.path.join(data_dir, remove_img_name, "images")
        self.transform = transform or identity_transform()
        self.vae = vae
        self.is_knn = False

        labels_path = os.path.join(data_dir, remove_img_name, "kmeans_labels.json")
        with open(labels_path, "r", encoding="utf-8") as file:
            labels = json.load(file)

        all_names = list(labels.keys())
        all_labels = torch.tensor(list(labels.values()))

        if filter == "all":
            index_filter = torch.arange(all_labels.shape[0])
            self.img_names = [all_names[i] for i in index_filter.tolist()]
            return
        if filter == "deletion":
            index_filter = torch.where(all_labels == 1)[0]
            self.img_names = [all_names[i] for i in index_filter.tolist()]
            return
        if filter == "remain":
            index_filter = torch.where(all_labels == 0)[0]
            self.img_names = [all_names[i] for i in index_filter.tolist()]
            return
        if filter != "knn":
            raise ValueError(f"Invalid filter: {filter}")
        if K is None:
            raise ValueError("KNN filter requires K.")

        self.is_knn = True
        cache_path = resolve_knn_cache_path(os.path.join(data_dir, remove_img_name), "stable_diffusion", remove_img_name, K, self.transform)
        cached = load_knn_cache(cache_path)
        if cached is not None:
            self.imgs_deletion = cached["imgs_deletion"]
            self.imgs_remain = cached["imgs_remain"]
            self.knn_indices = cached["knn_indices"]
            return

        deletion_index = torch.where(all_labels == 1)[0]
        remain_index = torch.where(all_labels == 0)[0]
        deletion_names = [all_names[i] for i in deletion_index.tolist()]
        remain_names = [all_names[i] for i in remain_index.tolist()]
        deletion_images = torch.stack(
            [self.transform(read_image(os.path.join(self.images_dir, name)).to(torch.float)) for name in deletion_names]
        )
        remain_images = torch.stack(
            [self.transform(read_image(os.path.join(self.images_dir, name)).to(torch.float)) for name in remain_names]
        )
        self.imgs_deletion = self._vae_encode(deletion_images)
        self.imgs_remain = self._vae_encode(remain_images)
        start_time = time()
        self.knn_indices = find_knn_indices(self.imgs_remain, self.imgs_deletion, K)
        log_knn_time(start_time)
        save_knn_cache(
            cache_path,
            {
                "imgs_deletion": self.imgs_deletion,
                "imgs_remain": self.imgs_remain,
                "knn_indices": self.knn_indices,
            },
        )

    def __len__(self):
        return len(self.img_names) if not self.is_knn else self.knn_indices.shape[0]

    def __getitem__(self, idx):
        if not self.is_knn:
            image = read_image(os.path.join(self.images_dir, self.img_names[idx])).to(torch.float)
            image = self.transform(image)
            image = image.to(device=self.vae.device, dtype=self.vae.dtype).unsqueeze(0)
            return self.vae.encode(image).latent_dist.sample()[0] * self.vae.config.scaling_factor
        return self.imgs_deletion[idx], self.imgs_remain[self.knn_indices[idx]]

    def _vae_encode(self, images, batch_size=4):
        latents = []
        for index in range(0, images.size(0), batch_size):
            batch = images[index : index + batch_size].to(device=self.vae.device, dtype=self.vae.dtype)
            with torch.no_grad():
                latent = self.vae.encode(batch).latent_dist.sample() * self.vae.config.scaling_factor
            latents.append(latent.cpu())
        return torch.cat(latents, dim=0)
