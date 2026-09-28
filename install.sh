#!/bin/bash
# Create the conda env "kernel_attn" and build Weighted_Abs_Sum and Weighted_Laplace_Sum into it: bash install.sh
set -eo pipefail  # no -u: conda's cuda-nvcc activate script reads unset variables
cd "$(dirname "$(readlink -f "$0")")"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -n kernel_attn --override-channels -c nvidia -c conda-forge \
    python=3.12.12 cuda-toolkit=12.9.2 cuda-version=12.9 gcc_linux-64=13.4.0 gxx_linux-64=13.4.0
conda activate kernel_attn
export CUDA_HOME=$CONDA_PREFIX CUDA_PATH=$CONDA_PREFIX

pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu129
pip install pykeops==2.3 numpy==2.3.5 matplotlib==3.10.8 datasets==4.8.4 \
    transformers==5.5.4 sentence-transformers==5.5.0 mteb==2.12.30

# builds for all GPU archs; for a faster build set only yours, e.g. TORCH_CUDA_ARCH_LIST="8.6" pip install ...
pip install --no-build-isolation ./Weighted_Abs_Sum
pip install --no-build-isolation ./Weighted_Laplace_Sum
