### Download checkpoints

# celeba_hq
mkdir -p checkpoints/celeba_hq/pretrained
hf download google/ddpm-ema-celebahq-256 --local-dir checkpoints/celeba_hq/pretrained/ddpm_ema_celebahq_256 

# cifar10
mkdir -p checkpoints/cifar10/pretrained/ddpm_ema_cifar10_32/
wget https://github.com/VainF/Diff-Pruning/releases/download/v0.0.1/ddpm_ema_cifar10.zip
unzip ddpm_ema_cifar10.zip -d checkpoints/cifar10/pretrained/ddpm_ema_cifar10_32/
mv checkpoints/cifar10/pretrained/ddpm_ema_cifar10_32/ddpm_ema_cifar10/* checkpoints/cifar10/pretrained/ddpm_ema_cifar10_32/
rmdir checkpoints/cifar10/pretrained/ddpm_ema_cifar10_32/ddpm_ema_cifar10/
rm ddpm_ema_cifar10.zip

# stable diffusion
mkdir -p checkpoints/stable_diffusion/pretrained
hf download CompVis/stable-diffusion-v1-4 --local-dir checkpoints/stable_diffusion/pretrained/stable_diffusion_v1_4

# mnist_with_tshirt and sscd
mkdir -p checkpoints/mnist_with_tshirt/pretrained
mkdir -p checkpoints/mnist_with_tshirt/classifier
mkdir -p checkpoints/sscd
curl -L -o checkpoints.zip https://www.kaggle.com/api/v1/datasets/download/kenhas/data-unlearning-in-diffusion-models-checkpoints
unzip checkpoints.zip
mv checkpoints/classifiers/sscd_disc_mixup.torchscript.pt checkpoints/sscd/sscd_disc_mixup.torchscript.pt
mv checkpoints/classifiers/mnist.pt checkpoints/mnist_with_tshirt/classifier/mnist.pt
mv checkpoints/mnist-tshirt/base/checkpoint-117500 checkpoints/mnist_with_tshirt/pretrained/
rmdir checkpoints/classifiers
rmdir checkpoints/mnist-tshirt/base
rmdir checkpoints/mnist-tshirt
rm checkpoints.zip



### Download datasets

# cifar10
mkdir -p datasets/cifar10
hf download uoft-cs/cifar10 --repo-type=dataset --local-dir datasets/cifar10
mv datasets/cifar10/plain_text/train-00000-of-00001.parquet datasets/cifar10/train-00000-of-00001.parquet
rm -r datasets/cifar10/plain_text
rm datasets/cifar10/README.md

# mnist_with_tshirt
mkdir -p datasets/mnist_with_tshirt
hf download claserken/mnist-with-tshirt --repo-type=dataset --local-dir datasets/mnist_with_tshirt
mv datasets/mnist_with_tshirt/data/train-00000-of-00001.parquet datasets/mnist_with_tshirt/train-00000-of-00001.parquet
rm -r datasets/mnist_with_tshirt/data
rm datasets/mnist_with_tshirt/README.md

# celeba_hq, stable diffusion and t-shirt
mkdir -p datasets/celeba_hq_256
mkdir -p datasets/sd
curl -L -o datasets.zip https://www.kaggle.com/api/v1/datasets/download/kenhas/data-unlearning-in-diffusion-models-datasets
unzip datasets.zip
mv data/datasets/tshirt.png datasets/mnist_with_tshirt/tshirt.png
mv data/datasets/celeba_hq_256/* datasets/celeba_hq_256/
mv data/datasets/sd/* datasets/sd/
mv data/datasets/modified_prompts.json datasets/sd/modified_prompts.json
mv data/datasets/original_prompts.json datasets/sd/original_prompts.json
rm -r data
rm datasets.zip
