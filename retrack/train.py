from __future__ import annotations

import gc
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import accelerate
import datasets
import diffusers
import hydra
import numpy as np
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import UNet2DModel
from diffusers.optimization import get_scheduler
from packaging import version
from tqdm.auto import tqdm
from transformers.utils import ContextManagers

from retrack.artifacts import build_artifact_paths, ensure_run_directories, save_json, save_run_metadata
from retrack.methods import create_loss
from retrack.datasets import InfiniteSampler


@dataclass
class SDComponents:
    original_prompts: dict
    modified_prompts: dict
    tokenizer: object
    text_encoder: object
    vae: object


@dataclass(frozen=True)
class ExperimentSchedule:
    random_seeds: list
    remove_img_names: list
    num_runs: int
    tag: str


def run_experiments(cfg):
    logger = get_logger(cfg.project, log_level="INFO")
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    schedule = build_experiment_schedule(cfg)
    evaluator = None
    if cfg.mode in {"run", "eval"} and not cfg.debug.disable_eval:
        sampler = hydra.utils.instantiate(cfg.sampler, cfg=cfg, _recursive_=False)
        evaluator = hydra.utils.instantiate(cfg.evaluator, cfg=cfg, sampler=sampler, logger=logger, _recursive_=False)

    sd_components = load_stable_diffusion_components(cfg) if cfg.type == "sd" else None

    index = 0
    for random_seed in schedule.random_seeds:
        for remove_img_name in schedule.remove_img_names:
            index += 1
            run_single_experiment(
                cfg=cfg,
                remove_img_name=remove_img_name,
                random_seed=random_seed,
                logger=logger,
                evaluator=evaluator,
                run_index=index,
                num_runs=schedule.num_runs,
                tag=schedule.tag,
                sd_components=sd_components,
            )

    if cfg.mode in {"run", "eval"} and not cfg.debug.disable_eval:
        save_summary(cfg, schedule.random_seeds, schedule.remove_img_names)
    logger.info(f"Finish: {schedule.tag}")


def build_experiment_schedule(cfg) -> ExperimentSchedule:
    tag = f"{cfg.name}[{datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d_%H:%M:%S')}]"
    random_seeds = [cfg.random_seeds] if isinstance(cfg.random_seeds, int) else list(cfg.random_seeds)
    remove_img_names = [cfg.remove_img_names] if isinstance(cfg.remove_img_names, str) else list(cfg.remove_img_names)
    return ExperimentSchedule(
        random_seeds=random_seeds,
        remove_img_names=remove_img_names,
        num_runs=len(random_seeds) * len(remove_img_names),
        tag=tag,
    )


def run_single_experiment(cfg, remove_img_name, random_seed, logger, evaluator, run_index, num_runs, tag, sd_components):
    artifact_paths = build_artifact_paths(cfg, random_seed, remove_img_name)
    ensure_run_directories(artifact_paths)
    save_run_metadata(artifact_paths, cfg, random_seed, remove_img_name)

    if cfg.mode == "train":
        train_model(cfg, remove_img_name, random_seed, logger, evaluator, run_index, num_runs, tag, sd_components, artifact_paths)
        return
    if cfg.mode == "eval":
        evaluate_saved_model(cfg, remove_img_name, random_seed, logger, evaluator, sd_components, artifact_paths)
        return
    if cfg.mode == "sample":
        sample_saved_model(cfg, remove_img_name, random_seed, logger, sd_components, artifact_paths)
        return
    if cfg.mode != "run":
        raise ValueError(f"Unsupported mode: {cfg.mode}")
    train_model(cfg, remove_img_name, random_seed, logger, evaluator, run_index, num_runs, tag, sd_components, artifact_paths)
    if not cfg.debug.disable_eval:
        sample_saved_model(cfg, remove_img_name, random_seed, logger, sd_components, artifact_paths)
        evaluate_saved_model(cfg, remove_img_name, random_seed, logger, evaluator, sd_components, artifact_paths)


def train_model(cfg, remove_img_name, random_seed, logger, evaluator, run_index, num_runs, tag, sd_components, artifact_paths):
    set_seed(random_seed)
    accelerator = create_accelerator(cfg)
    register_accelerator_hooks(accelerator)
    os.makedirs(cfg.artifacts.model_root, exist_ok=True)
    os.makedirs(cfg.artifacts.results_root, exist_ok=True)

    configure_library_logging(accelerator)
    model_runtime = prepare_model_runtime(cfg, accelerator, remove_img_name, sd_components)
    loss_fn = create_loss(cfg.name, dict(getattr(cfg.algorithm, cfg.name, {})), model_runtime.noise_scheduler, cfg.device)
    train_state = create_train_state(cfg, accelerator, remove_img_name, loss_fn, model_runtime.dataset_params)

    log_run_header(cfg, logger, remove_img_name, random_seed, run_index, num_runs, train_state.dataset_lengths)

    if cfg.name == "pretrain" or cfg.train.steps == 0:
        maybe_save_finetuned_model(accelerator, model_runtime, artifact_paths)
        cleanup_run(accelerator, model_runtime, loss_fn, train_state)
        return

    progress_bar = tqdm(total=cfg.train.steps, disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    while train_state.global_step < cfg.train.steps:
        train_step(
            cfg=cfg,
            accelerator=accelerator,
            model_runtime=model_runtime,
            dataset_iterators=train_state.dataset_iterators,
            loss_fn=loss_fn,
            gradient_state=train_state.gradient_state,
        )

        if accelerator.sync_gradients:
            progress_bar.update(1)
            train_state.global_step += 1

        accelerator.wait_for_everyone()

    progress_bar.close()
    maybe_save_finetuned_model(accelerator, model_runtime, artifact_paths)
    cleanup_run(accelerator, model_runtime, loss_fn, train_state)


def evaluate_saved_model(cfg, remove_img_name, random_seed, logger, evaluator, sd_components, artifact_paths):
    if evaluator is None:
        return
    set_seed(random_seed)
    accelerator = create_accelerator(cfg)
    configure_library_logging(accelerator)
    require_saved_model(artifact_paths)
    saved_samples = load_saved_samples(cfg, artifact_paths)
    model_runtime = prepare_model_runtime(
        cfg,
        accelerator,
        remove_img_name,
        sd_components,
        model_source_dir=artifact_paths.model_dir,
        with_optimizer=False,
    )
    metrics_batch = evaluator(
        remove_img_name,
        random_seed,
        cfg.train.steps,
        model_runtime.model,
        model_runtime.noise_scheduler,
        result_dir=artifact_paths.result_dir,
        saved_samples=saved_samples,
        **model_runtime.evaluator_params,
    )
    save_json(artifact_paths.metrics_path, extract_scalar_metrics(metrics_batch, artifact_paths))
    cleanup_run(accelerator, model_runtime, None, None)


def sample_saved_model(cfg, remove_img_name, random_seed, logger, sd_components, artifact_paths):
    set_seed(random_seed)
    accelerator = create_accelerator(cfg)
    configure_library_logging(accelerator)
    require_saved_model(artifact_paths)
    model_runtime = prepare_model_runtime(
        cfg,
        accelerator,
        remove_img_name,
        sd_components,
        model_source_dir=artifact_paths.model_dir,
        with_optimizer=False,
    )
    sampler = hydra.utils.instantiate(cfg.sampler, cfg=cfg, _recursive_=False)
    sampler.load_model(model_runtime.model, model_runtime.noise_scheduler, **model_runtime.evaluator_params)
    samples = generate_samples(cfg, sampler)
    sampler.reset_model()
    save_samples(cfg, artifact_paths, samples)
    cleanup_run(accelerator, model_runtime, None, None)


def create_accelerator(cfg):
    logging_dir = os.path.join(cfg.artifacts.results_root, cfg.logging_dir)
    project_config = ProjectConfiguration(project_dir=cfg.artifacts.results_root, logging_dir=logging_dir)
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    return Accelerator(
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        mixed_precision=cfg.train.mixed_precision,
        project_config=project_config,
        kwargs_handlers=[kwargs],
    )


def register_accelerator_hooks(accelerator):
    if version.parse(accelerate.__version__) < version.parse("0.16.0"):
        return

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                model.save_pretrained(os.path.join(output_dir, "unet"))
                weights.pop()

    def load_model_hook(models, input_dir):
        for _ in range(len(models)):
            model = models.pop()
            loaded = UNet2DModel.from_pretrained(input_dir, subfolder="unet")
            model.register_to_config(**loaded.config)
            model.load_state_dict(loaded.state_dict())
            del loaded

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)


def configure_library_logging(accelerator):
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


def require_saved_model(artifact_paths):
    unet_dir = os.path.join(artifact_paths.model_dir, "unet")
    if not os.path.isdir(unet_dir):
        raise FileNotFoundError(
            f"Saved model not found at `{unet_dir}`. Run `mode=train` or `mode=run` first."
        )


def require_saved_samples(cfg, artifact_paths):
    if cfg.type == "sd":
        required_paths = [
            os.path.join(artifact_paths.sample_dir, "original.npy"),
            os.path.join(artifact_paths.sample_dir, "modified.npy"),
        ]
    else:
        required_paths = [os.path.join(artifact_paths.sample_dir, "samples.npy")]
    missing = [path for path in required_paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            f"Saved samples not found under `{artifact_paths.sample_dir}`. Run `mode=sample` or `mode=run` first."
        )


def save_samples(cfg, artifact_paths, samples):
    if cfg.type == "sd":
        np.save(os.path.join(artifact_paths.sample_dir, "original.npy"), samples[0])
        np.save(os.path.join(artifact_paths.sample_dir, "modified.npy"), samples[1])
        return
    np.save(os.path.join(artifact_paths.sample_dir, "samples.npy"), samples)


def load_saved_samples(cfg, artifact_paths):
    require_saved_samples(cfg, artifact_paths)
    if cfg.type == "sd":
        return (
            np.load(os.path.join(artifact_paths.sample_dir, "original.npy")),
            np.load(os.path.join(artifact_paths.sample_dir, "modified.npy")),
        )
    return np.load(os.path.join(artifact_paths.sample_dir, "samples.npy"))


def generate_samples(cfg, sampler):
    eval_times, last_batch = divmod(cfg.eval.num_images, cfg.eval.batch_size)
    if last_batch > 0:
        total_rounds = eval_times + 1
    else:
        total_rounds = eval_times
        last_batch = cfg.eval.batch_size

    ddpm_batches = []
    sd_original_batches = []
    sd_modified_batches = []
    for index in tqdm(range(total_rounds), desc="sampling"):
        batch_size = cfg.eval.batch_size if index < total_rounds - 1 else last_batch
        batch_samples = sampler.sample_images(num_samples=batch_size, generator_seed=index, disable_tqdm=True)
        if cfg.type == "sd":
            sd_original_batches.append(batch_samples[0])
            sd_modified_batches.append(batch_samples[1])
        else:
            ddpm_batches.append(batch_samples)

    if cfg.type == "sd":
        return np.concatenate(sd_original_batches, axis=0), np.concatenate(sd_modified_batches, axis=0)
    return np.concatenate(ddpm_batches, axis=0)


@dataclass
class ModelRuntime:
    model: object
    optimizer: object
    lr_scheduler: object
    noise_scheduler: object
    model_params: dict
    dataset_params: dict
    evaluator_params: dict
    weight_dtype: torch.dtype


@dataclass
class TrainState:
    dataset_iterators: dict
    dataset_lengths: dict
    gradient_state: dict | None
    global_step: int = 0


@dataclass(frozen=True)
class RuntimeContext:
    dataset_params: dict
    model_params: dict
    evaluator_params: dict


def prepare_model_runtime(cfg, accelerator, remove_img_name, sd_components, model_source_dir=None, with_optimizer=True):
    weight_dtype = resolve_weight_dtype(cfg, accelerator)
    runtime_context = build_runtime_context(cfg, remove_img_name, sd_components, weight_dtype)

    model = instantiate_model(cfg, model_source_dir)
    noise_scheduler = hydra.utils.instantiate(cfg.scheduler)
    if with_optimizer:
        optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
        lr_scheduler = get_scheduler(
            cfg.train.lr.scheduler,
            optimizer=optimizer,
            num_warmup_steps=cfg.train.lr.warmup_steps * accelerator.num_processes,
            num_training_steps=cfg.train.steps,
        )
        model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    else:
        model = accelerator.prepare(model)
        optimizer = None
        lr_scheduler = None
    return ModelRuntime(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        noise_scheduler=noise_scheduler,
        model_params=runtime_context.model_params,
        dataset_params=runtime_context.dataset_params,
        evaluator_params=runtime_context.evaluator_params,
        weight_dtype=weight_dtype,
    )


def instantiate_model(cfg, model_source_dir=None):
    if model_source_dir is None:
        return hydra.utils.instantiate(cfg.model)
    return hydra.utils.instantiate(
        cfg.model,
        pretrained_model_name_or_path=model_source_dir,
        subfolder="unet",
    )


def resolve_weight_dtype(cfg, accelerator):
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        cfg.train.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        cfg.train.mixed_precision = accelerator.mixed_precision
    return weight_dtype


def build_runtime_context(cfg, remove_img_name, sd_components, weight_dtype) -> RuntimeContext:
    if cfg.type != "sd":
        return RuntimeContext(dataset_params={}, model_params={}, evaluator_params={})
    return build_stable_diffusion_runtime_context(cfg, remove_img_name, sd_components, weight_dtype)


def build_stable_diffusion_runtime_context(cfg, remove_img_name, sd_components, weight_dtype) -> RuntimeContext:
    original_prompt = sd_components.original_prompts[remove_img_name]
    modified_prompt = sd_components.modified_prompts[remove_img_name]
    sd_components.vae.requires_grad_(False)
    sd_components.text_encoder.requires_grad_(False)
    sd_components.vae.to(device=cfg.device, dtype=weight_dtype)
    sd_components.text_encoder.to(device=cfg.device, dtype=weight_dtype)

    input_ids = sd_components.tokenizer(
        modified_prompt,
        max_length=sd_components.tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids.repeat(cfg.train.batch_size, 1).to(device=cfg.device)

    dataset_params = {"vae": sd_components.vae}
    model_params = {
        "encoder_hidden_states": sd_components.text_encoder(input_ids, return_dict=False)[0],
    }
    evaluator_params = {
        "vae": sd_components.vae,
        "text_encoder": sd_components.text_encoder,
        "tokenizer": sd_components.tokenizer,
        "prompts": [original_prompt, modified_prompt],
    }
    return RuntimeContext(
        dataset_params=dataset_params,
        model_params=model_params,
        evaluator_params=evaluator_params,
    )


def create_dataset_iterators(cfg, accelerator, remove_img_name, loss_fn, dataset_params):
    transform = hydra.utils.instantiate(cfg.transform)
    dataset_iterators = {}
    dataset_lengths = {}
    for dataset_type, is_required in loss_fn.need_dataset.items():
        if not is_required:
            continue
        dataset = hydra.utils.instantiate(
            cfg.dataset[dataset_type],
            remove_img_name=remove_img_name,
            transform=transform,
            **dataset_params,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.train.batch_size,
            sampler=InfiniteSampler(dataset),
            num_workers=cfg.train.dataloader_num_workers,
        )
        dataset_iterators[dataset_type] = accelerator.prepare(iter(dataloader))
        dataset_lengths[dataset_type] = len(dataset)
    return dataset_iterators, dataset_lengths


def create_train_state(cfg, accelerator, remove_img_name, loss_fn, dataset_params) -> TrainState:
    dataset_iterators, dataset_lengths = create_dataset_iterators(cfg, accelerator, remove_img_name, loss_fn, dataset_params)
    gradient_state = {"main": {}, "correction": {}} if loss_fn.need_rescale else None
    return TrainState(
        dataset_iterators=dataset_iterators,
        dataset_lengths=dataset_lengths,
        gradient_state=gradient_state,
    )


def log_run_header(cfg, logger, remove_img_name, random_seed, run_index, num_runs, dataset_lengths):
    logger.info("***** Running training *****")
    logger.info(f"  Running: [{run_index}/{num_runs}]")
    logger.info(f"  Seed: {random_seed}")
    logger.info(f"  Remove image name: {remove_img_name}")
    for dataset_name, dataset_length in dataset_lengths.items():
        logger.info(f"  Num of {dataset_name} examples = {dataset_length}")
    logger.info(f"  Num training steps = {cfg.train.steps}")
    logger.info(f"  Instantaneous batch size per device = {cfg.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.train.gradient_accumulation_steps}")
    logger.info(f"  Unlearning algorithm = {cfg.name}")
    logger.info("****************************")


def train_step(cfg, accelerator, model_runtime, dataset_iterators, loss_fn, gradient_state):
    model_runtime.model.train()
    images, noises, noisy_images, timesteps, model_outputs = {}, {}, {}, {}, {}
    knn_neighbors = None

    with accelerator.accumulate(model_runtime.model):
        for dataset_type, iterator in dataset_iterators.items():
            batch = next(iterator)
            if dataset_type == "knn":
                images[dataset_type], knn_neighbors = [item.to(device=cfg.device, dtype=model_runtime.weight_dtype) for item in batch]
            else:
                images[dataset_type] = batch.to(device=cfg.device, dtype=model_runtime.weight_dtype)
            noises[dataset_type] = torch.randn_like(images[dataset_type])
            batch_size = images[dataset_type].shape[0]
            timesteps[dataset_type] = torch.randint(
                0,
                model_runtime.noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=images[dataset_type].device,
            ).long()
            noisy_images[dataset_type] = model_runtime.noise_scheduler.add_noise(images[dataset_type], noises[dataset_type], timesteps[dataset_type])
            model_outputs[dataset_type] = model_runtime.model(
                noisy_images[dataset_type],
                timesteps[dataset_type],
                **model_runtime.model_params,
            ).sample

        loss = loss_fn(model_outputs, images, noises, noisy_images, timesteps, knn_neighbors)
        if loss_fn.need_rescale:
            backward_with_rescaling(cfg, accelerator, model_runtime, loss_fn, loss, gradient_state)
        else:
            accelerator.backward(loss)

        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model_runtime.model.parameters(), 1.0)

        model_runtime.optimizer.step()
        model_runtime.lr_scheduler.step()
        model_runtime.optimizer.zero_grad()


def maybe_save_finetuned_model(accelerator, model_runtime, artifact_paths):
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    unwrapped_model = accelerator.unwrap_model(model_runtime.model)
    unwrapped_model.save_pretrained(os.path.join(artifact_paths.model_dir, "unet"))
    if hasattr(model_runtime.noise_scheduler, "save_pretrained"):
        model_runtime.noise_scheduler.save_pretrained(os.path.join(artifact_paths.model_dir, "scheduler"))


def backward_with_rescaling(cfg, accelerator, model_runtime, loss_fn, loss, gradient_state):
    loss_main, loss_correction = loss
    accelerator.backward(loss_main, retain_graph=True)
    saved_grads = {name: param.grad.clone() for name, param in model_runtime.model.named_parameters()}

    accelerator.backward(loss_correction)
    for name, param in model_runtime.model.named_parameters():
        grad = param.grad.clone() - saved_grads[name]
        gradient_state["correction"][name] = gradient_state["correction"].get(name, 0) + grad
    del saved_grads

    if not accelerator.sync_gradients:
        return

    for name, param in model_runtime.model.named_parameters():
        gradient_state["main"][name] = param.grad.clone() - gradient_state["correction"][name]

    correction_norm = _gradient_norm(gradient_state["correction"])
    if loss_fn.name == "siss":
        scaling_factor = cfg.algorithm.siss.rescale_factor / correction_norm
    elif loss_fn.name == "erasediff":
        inner_product = sum(
            torch.sum(gradient_state["main"][name] * gradient_state["correction"][name])
            for name, _ in model_runtime.model.named_parameters()
        )
        scaling_factor = cfg.algorithm.erasediff.eta - inner_product / (correction_norm ** 2)
        scaling_factor = -max(scaling_factor, 0.0)
    else:
        raise ValueError(f"Unknown loss name {loss_fn.name} for gradient rescaling")

    for name, param in model_runtime.model.named_parameters():
        param.grad = gradient_state["main"][name] - scaling_factor * gradient_state["correction"][name]

    gradient_state["main"].clear()
    gradient_state["correction"].clear()
    torch.cuda.empty_cache()


def _gradient_norm(gradients):
    total = 0.0
    for grad in gradients.values():
        total += torch.norm(grad, p=2).item() ** 2
    return total ** 0.5


def cleanup_run(accelerator, model_runtime, loss_fn, train_state):
    accelerator.end_training()
    del model_runtime, loss_fn, train_state
    torch.cuda.empty_cache()
    gc.collect()


def load_stable_diffusion_components(cfg):
    with open(cfg.original_prompts_path, "r", encoding="utf-8") as file:
        original_prompts = json.load(file)
    with open(cfg.modified_prompts_path, "r", encoding="utf-8") as file:
        modified_prompts = json.load(file)

    tokenizer = hydra.utils.instantiate(cfg.tokenizer)

    def zero_init_disabled_context():
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []
        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    with ContextManagers(zero_init_disabled_context()):
        text_encoder = hydra.utils.instantiate(cfg.text_encoder)
        vae = hydra.utils.instantiate(cfg.vae)

    return SDComponents(
        original_prompts=original_prompts,
        modified_prompts=modified_prompts,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
    )


def extract_scalar_metrics(metric_batch, artifact_paths):
    metrics = {}
    for metric_name, value in metric_batch.items():
        if isinstance(value, torch.Tensor):
            value = value.item()
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (int, float)):
            metrics[metric_name] = value
    metrics["run_name"] = artifact_paths.run_name
    metrics["method_name"] = artifact_paths.method_name
    return metrics


def save_summary(cfg, random_seeds, remove_img_names):
    method_name = f"{cfg.name}{f'_{cfg.subname}' if cfg.subname else ''}"
    method_root = Path(cfg.artifacts.results_root) / method_name
    run_metrics = []
    for metrics_path in sorted(method_root.glob("*/metrics.json")):
        with open(metrics_path, "r", encoding="utf-8") as file:
            run_metrics.append(json.load(file))

    metric_names = sorted({key for run in run_metrics for key in run if key not in {"run_name", "method_name"}})
    metric_summary = {}
    for metric_name in metric_names:
        values = [run[metric_name] for run in run_metrics if metric_name in run]
        if values:
            metric_summary[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": values,
            }

    results = {
        "method_name": method_name,
        "random_seeds": random_seeds,
        "remove_img_names": remove_img_names,
        "num_runs": len(run_metrics),
        "metrics": metric_summary,
        "runs": run_metrics,
    }
    save_json(str(method_root / "summary.json"), results)
