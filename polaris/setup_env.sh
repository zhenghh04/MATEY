#!/bin/bash -l
# Setup MATEY conda environment on Polaris
# Run this once on a login node before submitting jobs

# Polaris compute nodes need proxy for external network access
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export ftp_proxy=http://proxy.alcf.anl.gov:3128

module load conda/2024-04-29
conda activate base

# Create a new environment
conda create -n matey python=3.10 -y
conda activate matey

# Install PyTorch with CUDA support (Polaris has A100 GPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install MATEY dependencies
pip install adan-pytorch==0.1.0 dadaptation==3.1 einops h5py numpy timm torchinfo \
    wandb tqdm zarr "ruamel.yaml==0.17.32" "ruamel.yaml.clib==0.2.7" \
    scipy matplotlib netCDF4 scikit-learn

# Install mpi4py (using system MPI)
module load cray-mpich
MPICC="cc -shared" pip install --no-cache-dir mpi4py

# flash-attn for A100
pip install flash-attn --no-build-isolation

# exodusii from Sandia
pip install git+https://github.com/sandialabs/exodusii.git

# Install MATEY itself in development mode
cd /eagle/datascience/hzheng/matey/MATEY
pip install -e .

echo "MATEY environment setup complete!"
conda list
