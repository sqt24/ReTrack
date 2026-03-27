from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from diffusers import DDPMPipeline, StableDiffusionPipeline
from tqdm import tqdm


class Sampler(ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        self.pipeline = None

    @abstractmethod
    def load_model(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def reset_model(self):
        raise NotImplementedError

    @abstractmethod
    def sample_images(self, num_samples, generator_seed=None, disable_tqdm=False, tqdm_desc=None):
        raise NotImplementedError


class DDPMSampler(Sampler):
    def load_model(self, unet, noise_scheduler):
        self.unet = unet
        self.noise_scheduler = noise_scheduler
        self.pipeline = DDPMPipeline(unet=unet, scheduler=noise_scheduler).to(self.cfg.device)

    def reset_model(self):
        self.unet = None
        self.noise_scheduler = None
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        torch.cuda.empty_cache()

    def sample_images(self, num_samples, generator_seed=None, disable_tqdm=False, tqdm_desc=None):
        self.pipeline.set_progress_bar_config(disable=disable_tqdm, desc=tqdm_desc)
        generator = torch.Generator(device=self.cfg.device).manual_seed(generator_seed) if generator_seed is not None else None
        with torch.no_grad():
            return self.pipeline(
                generator=generator,
                batch_size=num_samples,
                num_inference_steps=self.cfg.eval.num_inference_steps,
                output_type="numpy",
            ).images

    def inject_and_denoise_images(self, target_image, generator_seed=None):
        generator = torch.Generator(device=self.cfg.device).manual_seed(generator_seed) if generator_seed is not None else None
        self.noise_scheduler.set_timesteps(1000)
        timestep = self.cfg.metrics.sscd.denoising_injection_timestep
        noise = torch.randn(target_image.shape, generator=generator, device=self.cfg.device)
        timestep_batch = torch.full((target_image.shape[0],), timestep, device=self.cfg.device)
        noisy_image = self.pipeline.scheduler.add_noise(target_image, noise, timestep_batch)
        denoised_image = noisy_image.clone()
        with torch.no_grad():
            for step in tqdm(range(timestep, -1, -1), desc="Denoising from injection"):
                model_output = self.unet(denoised_image, step)["sample"]
                denoised_image = self.noise_scheduler.step(model_output, step, denoised_image, generator=generator)["prev_sample"]
        return self.to_numpy_image(target_image), self.to_numpy_image(noisy_image), self.to_numpy_image(denoised_image)

    @staticmethod
    def to_numpy_image(image):
        image = (image + 1) / 2
        return image.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()


class SDSampler(Sampler):
    def load_model(self, unet, noise_scheduler, vae, text_encoder, tokenizer, prompts):
        self.pipeline = StableDiffusionPipeline(
            unet=unet,
            scheduler=noise_scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        ).to(self.cfg.device)
        self.prompts = prompts

    def reset_model(self):
        self.prompts = None
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        torch.cuda.empty_cache()

    def sample_images(self, num_samples, generator_seed=None, disable_tqdm=False, tqdm_desc=None):
        self.pipeline.set_progress_bar_config(disable=disable_tqdm, desc=tqdm_desc)
        generator = torch.Generator(device=self.cfg.device).manual_seed(generator_seed) if generator_seed is not None else None
        with torch.no_grad():
            images = []
            for prompt in self.prompts:
                images.append(
                    self.pipeline(
                        prompt,
                        generator=generator,
                        num_images_per_prompt=num_samples,
                        num_inference_steps=self.cfg.eval.num_inference_steps,
                        output_type="np",
                    ).images
                )
        torch.cuda.empty_cache()
        return images
