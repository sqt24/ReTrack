from __future__ import annotations

import json
import os

import hydra
import joblib
import numpy as np
import torch
from PIL import Image
from torchmetrics.multimodal import CLIPImageQualityAssessment
from torchvision import transforms
from tqdm import tqdm

from .base import Evaluator


class SDEvaluator(Evaluator):
    def __init__(self, cfg, sampler, logger):
        super().__init__(cfg, sampler, logger)
        self.clip_iqa_computer = CLIPImageQualityAssessment().to(cfg.device)
        self.sscd_model = torch.jit.load(cfg.metrics.sscd.model_path).to(cfg.device)
        self.sscd_transform = hydra.utils.instantiate(cfg.metrics.sscd.transform)

    def load_model(self, unet, noise_scheduler, vae, text_encoder, tokenizer, prompts):
        self.sampler.load_model(unet, noise_scheduler, vae, text_encoder, tokenizer, prompts)
        self.kmeans_classifier = joblib.load(os.path.join(self.cfg.data_dir, self.remove_img_name, "kmeans_classifier.joblib"))
        clustering_info_path = os.path.join(self.cfg.data_dir, self.remove_img_name, "clustering_info.json")
        with open(clustering_info_path, "r", encoding="utf-8") as file:
            clustering_info = json.load(file)
        image_name = f"{self.remove_img_name}_{str(clustering_info['mem_idx']).zfill(3)}.png"
        image_path = os.path.join(self.cfg.data_dir, self.remove_img_name, "images", image_name)
        self.sscd_img = self.sscd_transform(Image.open(image_path)).unsqueeze(0).to(self.cfg.device)

    def reset_model(self):
        self.images = None
        self.kmeans_classifier = None
        self.sscd_img = None
        self.sampler.reset_model()

    def evaluate(self):
        self.log_eval_context()
        self.ensure_summary_keys(
            (
                "CLIP_IQA_original",
                "SSCD_original",
                "success_rate_original",
                "CLIP_IQA_modified",
                "SSCD_modified",
                "success_rate_modified",
            )
        )
        self.CLIP_IQA_original()
        self.SSCD_original()
        self.success_rate_original()
        self.CLIP_IQA_modified()
        self.SSCD_modified()
        self.success_rate_modified()
        return self.metrics

    def sample_images(self):
        eval_times, last_batch, generator_seeds = self.sample_seed_schedule()
        original_batches = []
        modified_batches = []
        for index in tqdm(range(eval_times), desc="evaluator sampling"):
            batch_size = self.cfg.eval.batch_size if index < eval_times - 1 else last_batch
            samples = self.sampler.sample_images(batch_size, generator_seed=generator_seeds[index], disable_tqdm=True)
            original_batches.append(samples[0])
            modified_batches.append(samples[1])
        images = [np.concatenate(original_batches, axis=0), np.concatenate(modified_batches, axis=0)]
        self.maybe_save_images(np.array(images))
        self.log_sample_grid("sampled_images_original", images[0], caption=f"Step {self.global_step}")
        self.log_sample_grid("sampled_images_modified", images[1], caption=f"Step {self.global_step}")
        self.images = [torch.from_numpy(batch).permute(0, 3, 1, 2).to(self.cfg.device) for batch in images]

    def set_samples(self, samples):
        self.log_sample_grid("sampled_images_original", samples[0], caption=f"Step {self.global_step}")
        self.log_sample_grid("sampled_images_modified", samples[1], caption=f"Step {self.global_step}")
        self.images = [torch.from_numpy(batch).permute(0, 3, 1, 2).to(self.cfg.device) for batch in samples]

    def _predict_success_rate(self, image_index):
        preds = self.kmeans_classifier.predict(255 * self.images[image_index].cpu().permute(0, 2, 3, 1).flatten(start_dim=1))
        return 1 if np.all(preds == 0) else 0

    def _compute_clip_iqa(self, image_index):
        return self.clip_iqa_computer(self.images[image_index]).mean().item()

    def _compute_sscd(self, image_index):
        images = [
            self.sscd_transform(transforms.ToPILImage()(image.cpu())).unsqueeze(0).to(self.sscd_img.device)
            for image in self.images[image_index]
        ]
        embeddings = self.sscd_model(torch.cat([self.sscd_img, *images], dim=0))
        scores = torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1:], dim=1).detach().cpu().numpy()
        return np.mean(scores).item()

    @Evaluator.metric
    def success_rate_original(self):
        return self._predict_success_rate(0)

    @Evaluator.metric
    def success_rate_modified(self):
        return self._predict_success_rate(1)

    @Evaluator.metric
    def CLIP_IQA_original(self):
        return self._compute_clip_iqa(0)

    @Evaluator.metric
    def CLIP_IQA_modified(self):
        return self._compute_clip_iqa(1)

    @Evaluator.metric
    def SSCD_original(self):
        return self._compute_sscd(0)

    @Evaluator.metric
    def SSCD_modified(self):
        return self._compute_sscd(1)
