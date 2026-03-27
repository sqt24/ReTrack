# ReTrack

[AISTATS 2026] Official implementation of "ReTrack: Data Unlearning in Diffusion Models through Redirecting the Denoising Trajectory".

## Setup

1. Clone this repo.

```bash
git clone https://github.com/sqt24/ReTrack.git
cd ReTrack
```

2. Create the environment.

```bash
conda env create -f environment.yml
conda activate retrack
```

3. Download datasets and checkpoints.

```bash
bash download.sh
bash scripts/check_assets.sh
```

## Running Experiments

### Example Run

```bash
python main.py experiment=mnist_with_tshirt name=retrack mode=run
```

### Key Arguments

- `experiment`: one of `mnist_with_tshirt`, `cifar10`, `celeba_hq`, `stable_diffusion`
- `name`: one of `pretrain`, `vanilla`, `neggrad`, `erasediff`, `siss`, `retrack`
- `mode=train`: fine-tune only
- `mode=sample`: load a saved checkpoint and generate samples
- `mode=eval`: evaluate saved samples
- `mode=run`: train, sample, and evaluate in sequence

## Outputs

By default, outputs are written under the repository root:

- `checkpoints/<dataset>/unlearn/<method>/<run>/`: fine-tuned checkpoints and `metadata.json`
- `results/<dataset>/<method>/<run>/samples/`: generated samples
- `results/<dataset>/<method>/<run>/metrics.json`: per-run evaluation results
- `results/<dataset>/<method>/summary.json`: aggregated statistics across runs

## Citation

```bibtex
@inproceedings{
    shi2026retrack,
    title={ReTrack: Data Unlearning in Diffusion Models through Redirecting the Denoising Trajectory},
    author={Qitan Shi and Cheng Jin and Jiawei Zhang and Yuantao Gu},
    booktitle={The 29th International Conference on Artificial Intelligence and Statistics},
    year={2026},
    url={https://openreview.net/forum?id=oTUZfa0iPv}
}
```

