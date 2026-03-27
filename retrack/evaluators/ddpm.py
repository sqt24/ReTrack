from __future__ import annotations

import os

import hydra
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchvision import transforms
from tqdm import tqdm

from retrack.utils.song_likelihood.likelihood import get_likelihood_fn
from retrack.utils.song_likelihood.sde_lib import VPSDE

from .base import Evaluator


class BaseDDPMEvaluator(Evaluator):
    metric_names = ()
    sampling_disabled_metric_names = ()

    def __init__(self, cfg, sampler, logger):
        super().__init__(cfg, sampler, logger)
        self.model = None
        self.transform = hydra.utils.instantiate(cfg.transform)
        self.likelihood_fn = get_likelihood_fn(VPSDE())
        self.sscd_model = None
        self.sscd_transform = None
        if hasattr(cfg.metrics, "sscd"):
            self.sscd_model = torch.jit.load(cfg.metrics.sscd.model_path).to(cfg.device)
            self.sscd_transform = hydra.utils.instantiate(cfg.metrics.sscd.transform)

    def load_model(self, unet, noise_scheduler):
        self.model = unet
        self.sampler.load_model(unet, noise_scheduler)
        self.load_targets()

    def reset_model(self):
        self.model = None
        self.images = None
        self.sampler.reset_model()
        self.reset_targets()

    def evaluate(self):
        metric_names = self.get_metric_names()
        self.log_eval_context()
        self.ensure_summary_keys(metric_names)
        for metric_name in metric_names:
            getattr(self, metric_name)()
        return self.metrics

    def sample_images(self):
        image_batches = []
        eval_times, last_batch, generator_seeds = self.sample_seed_schedule()
        for index in tqdm(range(eval_times), desc="evaluator sampling"):
            batch_size = self.cfg.eval.batch_size if index < eval_times - 1 else last_batch
            image_batches.append(
                self.sampler.sample_images(
                    num_samples=batch_size,
                    generator_seed=generator_seeds[index],
                    disable_tqdm=True,
                )
            )
            if self.cfg.debug.disable_sampling and (index + 1) * self.cfg.eval.batch_size >= self.cfg.eval.num_visualize:
                break
        images = np.concatenate(image_batches, axis=0)
        self.maybe_save_images(images)
        self.log_sample_grid("sampled_images", images, caption=f"Step {self.global_step}")
        self.images = torch.from_numpy(images).permute(0, 3, 1, 2).to(self.cfg.device)

    def set_samples(self, samples):
        self.images = torch.from_numpy(samples).permute(0, 3, 1, 2).to(self.cfg.device)
        self.log_sample_grid("sampled_images", samples, caption=f"Step {self.global_step}")

    def load_targets(self):
        pass

    def reset_targets(self):
        pass

    def get_metric_names(self):
        return self.sampling_disabled_metric_names if self.cfg.debug.disable_sampling else self.metric_names

    def init_fid(self, dataset_config, repeat_grayscale=False):
        self.fid_calculator = FrechetInceptionDistance(normalize=True, reset_real_features=False).to(self.cfg.device)
        real_dataset = hydra.utils.instantiate(dataset_config, transform=transforms.ToTensor())
        real_dataloader = DataLoader(real_dataset, batch_size=self.cfg.metrics.fid.batch_size, shuffle=False)
        for batch_real_images in tqdm(real_dataloader, desc="Loading real data for FID"):
            batch_real_images = batch_real_images.to(self.cfg.device)
            if repeat_grayscale and batch_real_images.shape[1] == 1:
                batch_real_images = batch_real_images.repeat(1, 3, 1, 1)
            self.fid_calculator.update(batch_real_images, real=True)
        torch.cuda.empty_cache()

    def compute_fid(self, images, repeat_grayscale=False):
        for index in range(0, len(images), self.cfg.metrics.fid.batch_size):
            batch = images[index : index + self.cfg.metrics.fid.batch_size]
            if repeat_grayscale and batch.shape[1] == 1:
                batch = batch.repeat(1, 3, 1, 1)
            self.fid_calculator.update(batch, real=False)
        fid_score = self.fid_calculator.compute()
        self.fid_calculator.reset()
        return fid_score.item()

    def compute_sscd_from_injection(self, target_image, target_sscd_image):
        injection_samples = self.sampler.inject_and_denoise_images(target_image, generator_seed=self.random_seed)
        injection_samples = np.concatenate(injection_samples, axis=0)
        self.log_sample_grid(
            "injection_samples",
            injection_samples,
            caption=f"Step {self.global_step} (t={self.cfg.metrics.sscd.denoising_injection_timestep})",
            nrow=3,
        )
        denoised_image = Image.fromarray((injection_samples[2] * 255).astype(np.uint8))
        denoised_image = self.sscd_transform(denoised_image).unsqueeze(0).to(self.cfg.device)
        embeddings = self.sscd_model(torch.cat([target_sscd_image, denoised_image], dim=0))
        return torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1], dim=0).item()


class Cifar10Evaluator(BaseDDPMEvaluator):
    metric_names = ("negative_likelihood", "SSCD", "FID", "IS")
    sampling_disabled_metric_names = ("negative_likelihood", "SSCD")

    def __init__(self, cfg, sampler, logger):
        super().__init__(cfg, sampler, logger)
        if not cfg.debug.disable_sampling:
            self.init_fid(cfg.dataset.all, repeat_grayscale=True)
            self.inception_calculator = InceptionScore(normalize=True).to(cfg.device)

    def load_targets(self):
        target_image_raw = hydra.utils.instantiate(
            self.cfg.dataset.deletion,
            remove_img_name=self.remove_img_name,
            transform=None,
        )[0]
        self.target_image = self.transform(target_image_raw).unsqueeze(0).to(self.cfg.device)
        self.sscd_target_image = self.sscd_transform(target_image_raw).unsqueeze(0).to(self.cfg.device)

    def reset_targets(self):
        self.target_image = None
        self.sscd_target_image = None

    @Evaluator.metric
    def negative_likelihood(self):
        return self.likelihood_fn(self.model, self.target_image)[0][0].item()

    @Evaluator.metric
    def FID(self):
        return self.compute_fid(self.images)

    @Evaluator.metric
    def IS(self):
        for index in range(0, len(self.images), self.cfg.metrics.inception_score.batch_size):
            self.inception_calculator.update(self.images[index : index + self.cfg.metrics.inception_score.batch_size])
        is_mean, _ = self.inception_calculator.compute()
        self.inception_calculator.reset()
        return is_mean.item()

    @Evaluator.metric
    def SSCD(self):
        return self.compute_sscd_from_injection(self.target_image, self.sscd_target_image)


class CelebAHQEvaluator(BaseDDPMEvaluator):
    metric_names = ("negative_likelihood", "SSCD", "FID")
    sampling_disabled_metric_names = ("negative_likelihood", "SSCD")

    def __init__(self, cfg, sampler, logger):
        super().__init__(cfg, sampler, logger)
        if not cfg.debug.disable_sampling:
            self.init_fid(cfg.dataset.all)

    def load_targets(self):
        target_image_raw = Image.open(os.path.join(self.cfg.data_dir, self.remove_img_name))
        self.target_image = self.transform(target_image_raw).unsqueeze(0).to(self.cfg.device)
        self.sscd_target_image = self.sscd_transform(target_image_raw).unsqueeze(0).to(self.cfg.device)

    def reset_targets(self):
        self.target_image = None
        self.sscd_target_image = None

    @Evaluator.metric
    def negative_likelihood(self):
        return self.likelihood_fn(self.model, self.target_image)[0][0].item()

    @Evaluator.metric
    def FID(self):
        return self.compute_fid(self.images)

    @Evaluator.metric
    def SSCD(self):
        return self.compute_sscd_from_injection(self.target_image, self.sscd_target_image)


class MnistWithTshirtEvaluator(BaseDDPMEvaluator):
    metric_names = ("negative_likelihood", "frequency", "inception_score", "FID")
    sampling_disabled_metric_names = ("negative_likelihood",)

    def __init__(self, cfg, sampler, logger):
        super().__init__(cfg, sampler, logger)
        transform = hydra.utils.instantiate(cfg.transform)
        self.tshirt_img_no_norm = transforms.ToTensor()(Image.open(cfg.metrics.frequency.tshirt_path)).to(cfg.device)
        self.tshirt_img_with_norm = transform(Image.open(cfg.metrics.frequency.tshirt_path)).to(cfg.device)
        if not cfg.debug.disable_sampling:
            self.init_fid(cfg.dataset.remain, repeat_grayscale=True)
            classifier = hydra.utils.instantiate(cfg.metrics.inception_score.classifier)
            classifier.load_state_dict(torch.load(cfg.metrics.inception_score.classifier_ckpt))
            self.inception_calculator = InceptionScore(feature=classifier).to(cfg.device)

    @Evaluator.metric
    def negative_likelihood(self):
        return self.likelihood_fn(self.model, self.tshirt_img_with_norm.unsqueeze(0))[0][0].item()

    @Evaluator.metric
    def FID(self):
        return self.compute_fid(self.images_no_tshirts, repeat_grayscale=True)

    @Evaluator.metric
    def inception_score(self):
        for index in range(0, len(self.images_no_tshirts), self.cfg.metrics.inception_score.batch_size):
            self.inception_calculator.update(self.images_no_tshirts[index : index + self.cfg.metrics.inception_score.batch_size])
        is_mean, _ = self.inception_calculator.compute()
        self.inception_calculator.reset()
        return is_mean.item()

    @Evaluator.metric
    def frequency(self):
        tshirt_flattened = self.tshirt_img_no_norm.view(-1)
        images_flattened = self.images.view(self.images.size(0), -1)
        distances = torch.norm(images_flattened - tshirt_flattened, dim=1)
        matches = distances < self.cfg.metrics.frequency.threshold
        self.images_no_tshirts = self.images[~matches]
        return matches.float().mean().item()
