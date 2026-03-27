from __future__ import annotations

SUPPORTED_METHODS = {"pretrain", "vanilla", "neggrad", "erasediff", "siss", "retrack"}
SUPPORTED_TYPES = {"ddpm", "sd"}
SUPPORTED_MODES = {"run", "train", "eval", "sample"}


def validate_config(cfg) -> None:
    if "mode" not in cfg:
        raise ValueError(f"`mode` must be set to one of: {sorted(SUPPORTED_MODES)}.")
    if cfg.mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported mode `{cfg.mode}`. Supported modes: {sorted(SUPPORTED_MODES)}")
    if cfg.name is None:
        raise ValueError("`name` must be set to one of: pretrain, vanilla, neggrad, erasediff, siss, retrack.")
    if cfg.name not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported method `{cfg.name}`. Supported methods: {sorted(SUPPORTED_METHODS)}")
    if cfg.type not in SUPPORTED_TYPES:
        raise ValueError(f"Unsupported model type `{cfg.type}`. Supported types: {sorted(SUPPORTED_TYPES)}")
    if cfg.train.steps < 0 or cfg.eval.steps <= 0:
        raise ValueError("`train.steps` must be >= 0 and `eval.steps` must be > 0.")
    if cfg.train.batch_size <= 0 or cfg.eval.batch_size <= 0:
        raise ValueError("Batch sizes must be positive.")
    if cfg.eval.num_visualize > cfg.eval.num_images:
        raise ValueError("`eval.num_visualize` cannot exceed `eval.num_images`.")
    if cfg.type == "sd":
        for field_name in ("tokenizer", "text_encoder", "vae"):
            if field_name not in cfg:
                raise ValueError(f"Stable Diffusion config requires `{field_name}`.")
    if cfg.name == "retrack" and "knn" not in cfg.dataset:
        raise ValueError("ReTrack requires a `dataset.knn` configuration.")
    if "artifacts" not in cfg:
        raise ValueError("`artifacts.model_root` and `artifacts.results_root` must be configured.")
    for field_name in ("model_root", "results_root"):
        if field_name not in cfg.artifacts:
            raise ValueError(f"`artifacts.{field_name}` must be configured.")
