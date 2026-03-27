from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Check dataset/model asset completeness without running training.")
    parser.add_argument(
        "--storage-root",
        default=".",
        help="Root directory containing checkpoints/, datasets/, and results/.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.storage_root).resolve()
    checks = [
        check_cifar10(root),
        check_celeba_hq(root),
        check_mnist_with_tshirt(root),
        check_stable_diffusion(root),
    ]

    failures = []
    for name, errors in checks:
        if errors:
            failures.extend(f"[{name}] {error}" for error in errors)
            print(f"[FAIL] {name}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"[OK]   {name}")

    if failures:
        print(f"\nAsset check failed with {len(failures)} issue(s).", file=sys.stderr)
        sys.exit(1)

    print("\nAll required assets are present.")


def check_cifar10(root: Path):
    errors = []
    model_root = root / "checkpoints" / "cifar10" / "pretrained" / "ddpm_ema_cifar10_32"
    data_root = root / "datasets" / "cifar10"
    require_file(model_root / "model_index.json", errors)
    require_file(model_root / "unet" / "config.json", errors)
    require_model_weights(model_root / "unet", errors)
    require_file(model_root / "scheduler" / "scheduler_config.json", errors)
    require_nonempty_dir(data_root, errors)
    require_file(data_root / "train-00000-of-00001.parquet", errors)
    require_file(root / "checkpoints" / "sscd" / "sscd_disc_mixup.torchscript.pt", errors)
    return "cifar10", errors


def check_celeba_hq(root: Path):
    errors = []
    model_root = root / "checkpoints" / "celeba_hq" / "pretrained" / "ddpm_ema_celebahq_256"
    data_root = root / "datasets" / "celeba_hq_256"
    require_file(model_root / "model_index.json", errors)
    require_file(model_root / "config.json", errors)
    require_model_weights(model_root, errors, base_name="diffusion_pytorch_model")
    require_file(model_root / "scheduler_config.json", errors)
    require_nonempty_dir(data_root, errors)
    jpgs = sorted(data_root.glob("*.jpg"))
    if len(jpgs) < 10:
        errors.append(f"Expected many JPG images under `{data_root}`, found {len(jpgs)}.")
    for sample_name in [f"{index:05d}.jpg" for index in range(10001, 10011)]:
        require_file(data_root / sample_name, errors)
    require_file(root / "checkpoints" / "sscd" / "sscd_disc_mixup.torchscript.pt", errors)
    return "celeba_hq", errors


def check_mnist_with_tshirt(root: Path):
    errors = []
    model_root = root / "checkpoints" / "mnist_with_tshirt" / "pretrained"
    checkpoint_root = model_root / "checkpoint-117500"
    data_root = root / "datasets" / "mnist_with_tshirt"
    require_file(checkpoint_root / "scheduler.bin", errors)
    require_file(checkpoint_root / "unet_ema" / "config.json", errors)
    require_model_weights(checkpoint_root / "unet_ema", errors)
    require_nonempty_dir(data_root, errors)
    require_file(data_root / "train-00000-of-00001.parquet", errors)
    require_file(data_root / "tshirt.png", errors)
    require_file(root / "checkpoints" / "mnist_with_tshirt" / "classifier" / "mnist.pt", errors)
    return "mnist_with_tshirt", errors


def check_stable_diffusion(root: Path):
    errors = []
    model_root = root / "checkpoints" / "stable_diffusion" / "pretrained" / "stable_diffusion_v1_4"
    data_root = root / "datasets" / "sd"
    require_file(model_root / "model_index.json", errors)
    require_file(model_root / "unet" / "config.json", errors)
    require_model_weights(model_root / "unet", errors)
    require_file(model_root / "vae" / "config.json", errors)
    require_model_weights(model_root / "vae", errors)
    require_file(model_root / "text_encoder" / "config.json", errors)
    require_model_weights(model_root / "text_encoder", errors)
    for filename in ["merges.txt", "vocab.json", "tokenizer_config.json", "special_tokens_map.json"]:
        require_file(model_root / "tokenizer" / filename, errors)
    require_file(model_root / "scheduler" / "scheduler_config.json", errors)
    require_file(data_root / "original_prompts.json", errors)
    require_file(data_root / "modified_prompts.json", errors)
    require_file(root / "checkpoints" / "sscd" / "sscd_disc_mixup.torchscript.pt", errors)

    prompt_names = load_prompt_names(data_root / "original_prompts.json", errors)
    if prompt_names:
        for prompt_name in prompt_names:
            concept_root = data_root / prompt_name
            if prompt_name in ("emma_watson_beauty_beast", "j_dilla_equipment_smithsonian"):
                continue
            require_nonempty_dir(concept_root, errors)
            require_file(concept_root / "clustering_info.json", errors)
            require_file(concept_root / "kmeans_classifier.joblib", errors)
            require_file(concept_root / "kmeans_labels.json", errors)
            images_dir = concept_root / "images"
            require_nonempty_dir(images_dir, errors)
    return "stable_diffusion", errors


def require_file(path: Path, errors: list[str]):
    if not path.is_file():
        errors.append(f"Missing file: `{path}`")


def require_nonempty_dir(path: Path, errors: list[str]):
    if not path.is_dir():
        errors.append(f"Missing directory: `{path}`")
        return
    if not any(path.iterdir()):
        errors.append(f"Directory is empty: `{path}`")


def require_model_weights(path: Path, errors: list[str], base_name: str = "diffusion_pytorch_model"):
    candidates = [
        path / f"{base_name}.bin",
        path / f"{base_name}.safetensors",
        path / "pytorch_model.bin",
    ]
    if not any(candidate.is_file() for candidate in candidates):
        errors.append(f"Missing model weights under `{path}`")


def load_prompt_names(path: Path, errors: list[str]):
    if not path.is_file():
        return []
    try:
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
    except Exception as exc:
        errors.append(f"Failed to parse `{path}`: {exc}")
        return []
    if isinstance(payload, dict):
        return sorted(payload.keys())
    errors.append(f"`{path}` is not a JSON object.")
    return []


if __name__ == "__main__":
    main()
