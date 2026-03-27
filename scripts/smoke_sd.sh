#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/tmp/retrack_hf
export HF_DATASETS_CACHE=/tmp/retrack_hf/datasets
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

SMOKE_DEVICE="${SMOKE_DEVICE:-cuda}"

python main.py experiment=stable_diffusion name=vanilla device="$SMOKE_DEVICE" random_seeds=0 train.batch_size=1 eval.batch_size=1 train.steps=1 eval.steps=1 eval.num_images=4 eval.num_visualize=4 remove_img_names=sylvester_stallone debug.disable_eval=true
