from __future__ import annotations

import hashlib
import os

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, window_size=0.5):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rng = None
        window = 0
        if self.shuffle:
            rng = np.random.RandomState(self.seed)
            rng.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        index = 0
        while True:
            current = index % order.size
            if index % self.num_replicas == self.rank:
                yield order[current]
            if window >= 2:
                target = (current - rng.randint(window)) % order.size
                order[current], order[target] = order[target], order[current]
            index += 1


def find_knn_indices(data_remain, data_deletion, num_neighbors):
    remain = _flatten_for_knn(data_remain)
    deletion = _flatten_for_knn(data_deletion)
    knn = NearestNeighbors(n_neighbors=num_neighbors, algorithm="auto")
    knn.fit(remain)
    _, indices = knn.kneighbors(deletion)
    return torch.from_numpy(indices).long()


def resolve_knn_cache_path(data_dir, dataset_name, remove_img_name, num_neighbors, transform) -> str:
    remove_token = str(remove_img_name if remove_img_name is not None else "none")
    transform_key = hashlib.md5(repr(transform).encode("utf-8")).hexdigest()[:12]
    filename = f"{dataset_name}_{_sanitize_token(remove_token)}_k{num_neighbors}_{transform_key}.pt"
    return os.path.join(data_dir, ".cache", "knn", filename)


def load_knn_cache(cache_path):
    if not os.path.isfile(cache_path):
        return None
    return torch.load(cache_path, map_location="cpu")


def save_knn_cache(cache_path, payload) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(payload, cache_path)


def _flatten_for_knn(data):
    if isinstance(data, torch.Tensor):
        return data.reshape(data.shape[0], -1).cpu().numpy()
    return data.reshape(data.shape[0], -1)


def _sanitize_token(token: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in token)
