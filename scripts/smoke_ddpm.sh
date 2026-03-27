#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/tmp/retrack_hf
export HF_DATASETS_CACHE=/tmp/retrack_hf/datasets
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

SMOKE_DEVICE="${SMOKE_DEVICE:-cuda}"

python main.py experiment=cifar10 name=vanilla device="$SMOKE_DEVICE" random_seeds=0 remove_img_names=[10000] train.batch_size=2 eval.batch_size=2 train.steps=1 eval.steps=1 eval.num_images=8 eval.num_visualize=8 debug.disable_eval=true
