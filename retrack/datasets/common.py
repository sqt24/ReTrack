from __future__ import annotations

from time import time


def identity_transform():
    return lambda x: x


def log_knn_time(start_time: float):
    elapsed = time() - start_time
    print(f"\n\nKNN search completed in {elapsed:.2f} seconds.\n\n")

