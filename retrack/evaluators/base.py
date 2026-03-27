from __future__ import annotations

import os
from abc import ABC, abstractmethod

import numpy as np
import torch
import torchvision.utils as utils


class Evaluator(ABC):
    def __init__(self, cfg, sampler, logger):
        self.cfg = cfg
        self.sampler = sampler
        self.logger = logger
        self.summary = {}

    def __call__(self, remove_img_name, random_seed, global_step, *args, result_dir=None, saved_samples=None, **kwargs):
        torch.cuda.empty_cache()
        self.metrics = {}
        self.remove_img_name = remove_img_name
        self.random_seed = random_seed
        self.global_step = global_step
        self.result_dir = result_dir
        self.load_model(*args, **kwargs)
        if saved_samples is None:
            self.sample_images()
        else:
            self.set_samples(saved_samples)
        metrics = self.evaluate()
        self.reset_model()
        torch.cuda.empty_cache()
        return metrics

    @abstractmethod
    def load_model(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def reset_model(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError

    @abstractmethod
    def sample_images(self):
        raise NotImplementedError

    @abstractmethod
    def set_samples(self, samples):
        raise NotImplementedError

    def get_summary(self):
        return self.summary

    def log_eval_context(self):
        self.logger.info(f"random seed: {self.random_seed}")
        self.logger.info(f"step: {self.global_step}")

    def ensure_summary_keys(self, metric_names):
        if self.global_step not in self.summary:
            self.summary[self.global_step] = {name: [] for name in metric_names}

    def sample_seed_schedule(self):
        eval_times = self.cfg.eval.num_images // self.cfg.eval.batch_size + 1
        last_batch = self.cfg.eval.num_images % self.cfg.eval.batch_size
        if last_batch == 0:
            eval_times -= 1
            last_batch = self.cfg.eval.batch_size
        return eval_times, last_batch, list(range(eval_times))

    def maybe_save_images(self, images):
        if self.cfg.eval.save_images:
            os.makedirs(self.result_dir, exist_ok=True)
            path = os.path.join(self.result_dir, f"images_step_{self.global_step}.npy")
            np.save(path, images)

    def log_sample_grid(self, key, images, caption, nrow=None):
        del key, caption
        grid = self.make_grid_from_images(images[: self.cfg.eval.num_visualize] if nrow is None else images, nrow=nrow)
        if self.cfg.eval.save_images:
            os.makedirs(self.result_dir, exist_ok=True)
            np.save(os.path.join(self.result_dir, f"grid_step_{self.global_step}.npy"), grid)

    @staticmethod
    def make_grid_from_images(images, nrow=None):
        num_samples = images.shape[0]
        num_channels = images.shape[-1]
        grid = utils.make_grid(
            torch.from_numpy(images).permute(0, 3, 1, 2),
            nrow=nrow if nrow is not None else int(np.sqrt(num_samples)),
        )
        if num_channels == 1:
            grid = np.array(grid.permute(1, 2, 0)[:, :, :1])
        else:
            grid = np.array(grid.permute(1, 2, 0))
        return grid

    @staticmethod
    def metric(func):
        def wrapper(self, *args, **kwargs):
            self.logger.info(f"Calculating {func.__name__} ...")
            result = func(self, *args, **kwargs)
            self.logger.info(f"{func.__name__}: {result:.3f}")
            self.summary[self.global_step][func.__name__].append(result)
            self.metrics[func.__name__] = result
            torch.cuda.empty_cache()
        return wrapper
