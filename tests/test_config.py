import unittest

from omegaconf import OmegaConf

from retrack.config import validate_config


class MinimalConfigValidationTest(unittest.TestCase):
    def test_rejects_invalid_configs(self):
        cfg = OmegaConf.create(
            {
                "mode": "run",
                "name": "unknown",
                "type": "ddpm",
                "train": {"steps": 1, "batch_size": 1},
                "eval": {"steps": 1, "batch_size": 1, "num_images": 1, "num_visualize": 1},
                "dataset": {"remain": {}},
                "artifacts": {"model_root": "checkpoints/tmp", "results_root": "results/tmp"},
            }
        )
        with self.assertRaises(ValueError):
            validate_config(cfg)

        cfg = OmegaConf.create(
            {
                "mode": "run",
                "name": "retrack",
                "type": "ddpm",
                "train": {"steps": 1, "batch_size": 1},
                "eval": {"steps": 1, "batch_size": 1, "num_images": 1, "num_visualize": 1},
                "dataset": {"remain": {}},
                "artifacts": {"model_root": "checkpoints/tmp", "results_root": "results/tmp"},
            }
        )
        with self.assertRaises(ValueError):
            validate_config(cfg)


if __name__ == "__main__":
    unittest.main()
