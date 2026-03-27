from __future__ import annotations

import hydra
from diffusers.utils import check_min_version

from retrack.config import validate_config
from retrack.train import run_experiments


def _validate_runtime():
    check_min_version("0.27.0.dev0")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    _validate_runtime()
    validate_config(cfg)
    run_experiments(cfg)


if __name__ == "__main__":
    main()
