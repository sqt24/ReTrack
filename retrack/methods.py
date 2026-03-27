from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DatasetRequirements:
    remain: bool
    deletion: bool
    knn: bool


class UnlearningLoss:
    """Encapsulates the training objective for each unlearning method."""

    _REQUIREMENTS = {
        "pretrain": DatasetRequirements(remain=False, deletion=False, knn=False),
        "vanilla": DatasetRequirements(remain=True, deletion=False, knn=False),
        "neggrad": DatasetRequirements(remain=False, deletion=True, knn=False),
        "erasediff": DatasetRequirements(remain=True, deletion=True, knn=False),
        "siss": DatasetRequirements(remain=True, deletion=True, knn=False),
        "retrack": DatasetRequirements(remain=True, deletion=False, knn=True),
    }

    def __init__(self, name: str, params: dict, noise_scheduler, device: str):
        if name not in self._REQUIREMENTS:
            raise ValueError(f"Unknown loss name: {name}")
        self.name = name
        self.params = params
        self.requirements = self._REQUIREMENTS[name]
        self.need_rescale = name in {"siss", "erasediff"}
        self.all_gamma = (noise_scheduler.alphas_cumprod ** 0.5).to(device)
        self.all_sigma = ((1 - noise_scheduler.alphas_cumprod) ** 0.5).to(device)
        self.device = device

    @property
    def need_dataset(self) -> dict[str, bool]:
        return {
            "remain": self.requirements.remain,
            "deletion": self.requirements.deletion,
            "knn": self.requirements.knn,
        }

    def __call__(self, *args, **kwargs):
        return getattr(self, self.name)(*args, **kwargs)

    def retrack(self, model_outputs, images, noises, noisy_images, timesteps, knn_neighbors):
        timesteps_knn = timesteps["knn"]
        gamma = self.all_gamma[timesteps_knn].view(-1, 1, 1, 1, 1)
        sigma = self.all_sigma[timesteps_knn].view(-1, 1, 1, 1, 1)
        xt = noisy_images["knn"].float().unsqueeze(1)
        model_output_knn = model_outputs["knn"].float().unsqueeze(1)
        model_output_remain = model_outputs["remain"].float()
        noise_remain = noises["remain"].float()

        v = (xt - gamma * knn_neighbors) / sigma
        dists = -0.5 * torch.sum(v ** 2, dim=(2, 3, 4))
        pn = torch.softmax(dists, dim=1)
        norms = torch.sum((model_output_knn - v) ** 2, dim=(2, 3, 4))
        loss_unlearn = torch.mean(torch.sum(pn * norms, dim=1))

        batch_size = model_output_remain.shape[0]
        loss_remain = F.mse_loss(model_output_remain, noise_remain, reduction="sum") / batch_size

        retrack_lambda = self.params["lambda"]
        return retrack_lambda * loss_unlearn + (1 - retrack_lambda) * loss_remain

    def vanilla(self, model_outputs, _images, noises, *_args, **_kwargs):
        batch_size = model_outputs["remain"].shape[0]
        return F.mse_loss(model_outputs["remain"].float(), noises["remain"].float(), reduction="sum") / batch_size

    def neggrad(self, model_outputs, _images, noises, *_args, **_kwargs):
        batch_size = model_outputs["deletion"].shape[0]
        return -F.mse_loss(model_outputs["deletion"].float(), noises["deletion"].float(), reduction="sum") / batch_size

    def erasediff(self, model_outputs, _images, noises, *_args, **_kwargs):
        batch_size = model_outputs["deletion"].shape[0]
        loss_main = F.mse_loss(model_outputs["remain"].float(), noises["remain"].float(), reduction="sum") / batch_size
        loss_correction = F.mse_loss(
            model_outputs["deletion"].float(),
            torch.rand_like(model_outputs["deletion"].float()),
            reduction="sum",
        ) / batch_size
        return loss_main, loss_correction

    def siss(self, model_outputs, images, _noises, noisy_images, timesteps, *_args, **_kwargs):
        timesteps_deletion = timesteps["deletion"]
        timesteps_remain = timesteps["remain"]
        gamma_deletion = self.all_gamma[timesteps_deletion]
        sigma_deletion = self.all_sigma[timesteps_deletion]
        gamma_remain = self.all_gamma[timesteps_remain]
        sigma_remain = self.all_sigma[timesteps_remain]
        images_deletion = images["deletion"]
        images_remain = images["remain"]
        noisy_images_deletion = noisy_images["deletion"]
        noisy_images_remain = noisy_images["remain"]
        model_output_deletion = model_outputs["deletion"]
        model_output_remain = model_outputs["remain"]
        siss_lambda = self.params["lambda"]
        batch_size = noisy_images_deletion.shape[0]
        image_shape = noisy_images_deletion.shape
        reduce_dims = list(range(1, len(image_shape)))

        remain_mask = torch.rand(batch_size, device=self.device) > siss_lambda
        deletion_mask = ~remain_mask

        mixture_noisy_images = torch.empty(image_shape, device=self.device)
        mixture_noisy_images[remain_mask] = noisy_images_remain[remain_mask]
        mixture_noisy_images[deletion_mask] = noisy_images_deletion[deletion_mask]

        mixture_model_output = torch.empty(image_shape, device=self.device)
        mixture_model_output[remain_mask] = model_output_remain[remain_mask]
        mixture_model_output[deletion_mask] = model_output_deletion[deletion_mask]

        mixture_gamma = torch.empty(batch_size, device=self.device)
        mixture_gamma[remain_mask] = gamma_remain[remain_mask]
        mixture_gamma[deletion_mask] = gamma_deletion[deletion_mask]
        mixture_sigma = torch.empty(batch_size, device=self.device)
        mixture_sigma[remain_mask] = sigma_remain[remain_mask]
        mixture_sigma[deletion_mask] = sigma_deletion[deletion_mask]

        view_shape = (-1,) + (1,) * (len(image_shape) - 1)
        eps_remain = (mixture_noisy_images - mixture_gamma.view(view_shape) * images_remain) / mixture_sigma.view(view_shape)
        eps_deletion = (mixture_noisy_images - mixture_gamma.view(view_shape) * images_deletion) / mixture_sigma.view(view_shape)

        unweighted_loss_remain = torch.sum((mixture_model_output - eps_remain) ** 2, dim=reduce_dims)
        unweighted_loss_deletion = torch.sum((mixture_model_output - eps_deletion) ** 2, dim=reduce_dims)

        dist_remain = torch.sum((mixture_noisy_images - mixture_gamma.view(view_shape) * images_remain) ** 2, dim=reduce_dims)
        dist_remain /= 2 * (mixture_sigma ** 2)
        dist_deletion = torch.sum((mixture_noisy_images - mixture_gamma.view(view_shape) * images_deletion) ** 2, dim=reduce_dims)
        dist_deletion /= 2 * (mixture_sigma ** 2)

        ratio_deletion_remain = torch.exp(dist_remain - dist_deletion)
        ratio_remain_deletion = torch.exp(dist_deletion - dist_remain)

        importance_weight_remain = 1 / ((1 - siss_lambda) + siss_lambda * ratio_deletion_remain)
        importance_weight_deletion = 1 / ((1 - siss_lambda) * ratio_remain_deletion + siss_lambda)

        loss_main = torch.mean(importance_weight_remain * unweighted_loss_remain)
        loss_correction = torch.mean(importance_weight_deletion * unweighted_loss_deletion)
        return loss_main, loss_correction


def create_loss(name: str, params: dict, noise_scheduler, device: str) -> UnlearningLoss:
    return UnlearningLoss(name=name, params=params, noise_scheduler=noise_scheduler, device=device)
