#!/bin/bash -l
#PBS -A AmSC_Demos
#PBS -q debug
#PBS -l select=1:system=polaris
#PBS -l walltime=00:30:00
#PBS -l filesystems=eagle:home
#PBS -l place=scatter
#PBS -N matey_mw_svit
#PBS -j oe

set -e

# ---- Paths ----
WORK_DIR=/eagle/datascience/hzheng/matey
MATEY_DIR=${WORK_DIR}/MATEY
JOB_DIR=${WORK_DIR}/job

cd ${JOB_DIR}

# ---- Environment (use Polaris base conda + pip install missing deps) ----
# Proxy needed for pip to reach external repos on compute nodes
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

module load conda/2024-04-29
conda activate base

# Install MATEY-specific deps not in base conda (only runs once thanks to pip cache)
pip install --user adan-pytorch==0.1.0 dadaptation==3.1 einops timm torchinfo \
    wandb tqdm "zarr<3" "numpy<2" "ruamel.yaml==0.17.32" "ruamel.yaml.clib==0.2.7" \
    netCDF4 scikit-learn
pip install --user git+https://github.com/sandialabs/exodusii.git

# Add MATEY to Python path (skip pip install -e which fails on compute nodes)
export PYTHONPATH="${MATEY_DIR}:${PYTHONPATH}"
export OMP_NUM_THREADS=1

# ---- Distributed setup (translate PBS → SLURM env vars for MATEY) ----
NNODES=$(wc -l < $PBS_NODEFILE)
NGPUS_PER_NODE=4
NTOTAL=$((NNODES * NGPUS_PER_NODE))

export MASTER_ADDR=$(head -1 $PBS_NODEFILE)
export MASTER_PORT=3442

# MATEY's setup_dist() reads SLURM env vars; set them from PBS
export SLURM_NTASKS=${NTOTAL}

# ---- Launch ----
echo "Launching MATEY MW SVIT: ${NNODES} nodes, ${NTOTAL} GPUs"
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

mpiexec -n ${NTOTAL} --ppn ${NGPUS_PER_NODE} \
    --cpu-bind depth --depth 8 \
    bash -c '
        export SLURM_PROCID=${PMI_RANK}
        export SLURM_LOCALID=${PMI_LOCAL_RANK}
        export NCCL_SOCKET_IFNAME=lo
        export TMPDIR=/tmp/matey_${PMI_RANK}
        mkdir -p ${TMPDIR}
        cd '"${MATEY_DIR}"'
        python '"${JOB_DIR}"'/basic_usage.py \
            --run_name demo_mw_svit \
            --config basic_config \
            --yaml_config '"${JOB_DIR}"'/config/Demo_MW_svit_polaris.yaml \
            --use_ddp
    '

echo "MATEY MW SVIT job completed."
