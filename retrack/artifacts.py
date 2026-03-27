from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from omegaconf import OmegaConf


@dataclass(frozen=True)
class ArtifactPaths:
    run_name: str
    method_name: str
    model_dir: str
    result_dir: str
    sample_dir: str
    metrics_path: str
    aggregate_summary_path: str
    metadata_path: str


def build_artifact_paths(cfg, random_seed, remove_img_name) -> ArtifactPaths:
    method_name = f"{cfg.name}{f'_{cfg.subname}' if cfg.subname else ''}"
    run_name = f"seed{random_seed}_{sanitize_name(remove_img_name)}"
    model_dir = os.path.join(cfg.artifacts.model_root, method_name, run_name)
    result_dir = os.path.join(cfg.artifacts.results_root, method_name, run_name)
    return ArtifactPaths(
        run_name=run_name,
        method_name=method_name,
        model_dir=model_dir,
        result_dir=result_dir,
        sample_dir=os.path.join(result_dir, "samples"),
        metrics_path=os.path.join(result_dir, "metrics.json"),
        aggregate_summary_path=os.path.join(cfg.artifacts.results_root, method_name, "summary.json"),
        metadata_path=os.path.join(model_dir, "metadata.json"),
    )


def ensure_run_directories(paths: ArtifactPaths) -> None:
    os.makedirs(paths.model_dir, exist_ok=True)
    os.makedirs(paths.result_dir, exist_ok=True)
    os.makedirs(paths.sample_dir, exist_ok=True)


def save_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def save_run_metadata(paths: ArtifactPaths, cfg, random_seed, remove_img_name) -> None:
    payload = {
        "run_name": paths.run_name,
        "method_name": paths.method_name,
        "random_seed": random_seed,
        "remove_img_name": remove_img_name,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    save_json(paths.metadata_path, payload)


def sanitize_name(value) -> str:
    text = str(value)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
