# Porting MATEY to ALCF Polaris: An Agentic Workflow Report

**Author:** Huihuo Zheng (ALCF)\
**Date:** April 8, 2026\
**System:** Polaris (ALCF) — NVIDIA A100 GPUs

---

## 1. Overview

MATEY (Multi-scale Adaptive Transformer for Evolving sYstems) is a scalable, open-source framework for developing transformer-based spatiotemporal foundation models for physical systems. Originally developed for ORNL's Frontier supercomputer (AMD MI250X GPUs, ROCM, SLURM), this report documents the automated porting of MATEY to ALCF's Polaris supercomputer (NVIDIA A100 GPUs, CUDA, PBS) using an agentic AI workflow powered by Claude Code with ALCF IRI and Globus MCP integrations.

The agentic workflow autonomously:
- Analyzed the MATEY codebase and identified porting requirements
- Created PBS job scripts and adapted configuration files
- Transferred files to Eagle storage via Globus
- Submitted jobs to Polaris via the ALCF IRI API
- Monitored job status and retrieved output
- Diagnosed and resolved 6 distinct runtime errors iteratively
- Achieved a successful SVIT model training run

## 2. System Differences

| Aspect | Frontier (ORNL) | Polaris (ALCF) |
|--------|-----------------|----------------|
| GPU | AMD MI250X | NVIDIA A100 |
| GPUs per node | 8 | 4 |
| Software stack | ROCM 6.3.1 | CUDA 12.1 |
| Job scheduler | SLURM | PBS Pro |
| Job launcher | `srun` | `mpiexec` |
| GPU cache | MIOPEN | cuDNN (automatic) |
| Network interface | `hsn0` (Slingshot) | `lo` / auto |
| Filesystem | Lustre (`/lustre/orion/`) | Eagle (`/eagle/`) |
| Internet access | Direct | Proxy (`proxy.alcf.anl.gov:3128`) |

## 3. Porting Adaptations

### 3.1 Job Scheduler Translation (SLURM to PBS)

The Frontier batch test script uses SLURM directives and `srun` for parallel job launch within a single allocation. On Polaris, this required:

- **PBS directives:** `#SBATCH` replaced with `#PBS` equivalents
- **Parallel launch:** `srun -N$NN -n$((NN*8)) ... &` replaced with `mpiexec --hostfile <per-test-hostfile> ... &`
- **Node assignment:** Created per-test hostfiles by slicing `$PBS_NODEFILE`
- **Environment variables:** PBS does not set SLURM variables; a translation layer maps `PMI_RANK` and `PMI_LOCAL_RANK` to `SLURM_PROCID` and `SLURM_LOCALID`

**Example PBS-to-SLURM translation in the mpiexec wrapper:**
```bash
mpiexec -n ${NTOTAL} --ppn ${NGPUS_PER_NODE} \
    --cpu-bind depth --depth 8 \
    bash -c '
        export SLURM_PROCID=${PMI_RANK}
        export SLURM_LOCALID=${PMI_LOCAL_RANK}
        export NCCL_SOCKET_IFNAME=lo
        export TMPDIR=/tmp/matey_${PMI_RANK}
        mkdir -p ${TMPDIR}
        cd '"${MATEY_DIR}"'
        python basic_usage.py --run_name demo_mw_svit \
            --config basic_config \
            --yaml_config config/Demo_MW_svit_polaris.yaml \
            --use_ddp
    '
```

### 3.2 NCCL Network Interface

MATEY's `distributed_utils.py` hardcodes `NCCL_SOCKET_IFNAME='hsn0'` (Frontier's Slingshot interface). This was patched to respect pre-set environment variables:

```python
# Before (Frontier-only)
os.environ['NCCL_SOCKET_IFNAME'] = 'hsn0'

# After (portable)
os.environ['NCCL_SOCKET_IFNAME'] = os.environ.get('NCCL_SOCKET_IFNAME', 'hsn0')
```

For single-node Polaris jobs, `NCCL_SOCKET_IFNAME=lo` (loopback) is sufficient. Multi-node jobs should use the appropriate Slingshot interface.

### 3.3 GPU Device Selection

On Frontier, each rank is bound to a specific GPU via `srun --gpu-bind=closest`. On Polaris, the initial approach of setting `CUDA_VISIBLE_DEVICES=${PMI_LOCAL_RANK}` caused failures because MATEY's `setup_dist()` calls `torch.cuda.set_device(local_rank)` — when only one GPU is visible, device indices > 0 are invalid.

**Solution:** Remove `CUDA_VISIBLE_DEVICES` and let `torch.cuda.set_device(local_rank)` handle GPU selection directly. All 4 GPUs remain visible to each rank, and the correct one is selected programmatically.

### 3.4 Python Environment

Polaris uses a system conda environment (`conda/2024-04-29`) with pre-installed PyTorch, timm, and torchvision. MATEY-specific dependencies are installed via `pip install --user`. Key considerations:

- **Proxy required:** Compute nodes need `http_proxy=http://proxy.alcf.anl.gov:3128` for pip/git access
- **NumPy version pinning:** System packages are compiled against NumPy 1.x; must pin `numpy<2` and `zarr<3` to prevent ABI incompatibility
- **Editable install failure:** `pip install -e` fails on compute nodes due to PBS temp directory issues; use `PYTHONPATH` instead
- **exodusii:** Must be installed from GitHub (`pip install --user git+https://github.com/sandialabs/exodusii.git`)

### 3.5 Temporary Directory Path Length

PBS creates job-specific temp directories with very long paths (e.g., `/var/tmp/pbs.7059055.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/`), exceeding the 108-character Unix domain socket path limit. Python's `multiprocessing` module uses these paths for inter-process communication sockets.

**Solution:** Override `TMPDIR` with a short path per rank:
```bash
export TMPDIR=/tmp/matey_${PMI_RANK}
mkdir -p ${TMPDIR}
```

## 4. Iterative Debugging Process

The agentic workflow used a rapid submit-diagnose-fix cycle, submitting 10 jobs to resolve 6 distinct issues:

| # | Error | Root Cause | Fix |
|---|-------|-----------|-----|
| 1 | `No .egg-info directory found` | Editable pip install fails on compute nodes | Use PYTHONPATH instead |
| 2 | `_ARRAY_API not found` | NumPy 2.x breaks system packages | Pin `numpy<2`, `zarr<3` |
| 3 | `No module named 'exodusii'` | Missing Sandia library | Install from GitHub with proxy |
| 4 | `CUDA error: invalid device ordinal` | CUDA_VISIBLE_DEVICES conflicts with set_device() | Remove CUDA_VISIBLE_DEVICES |
| 5 | `ZeroDivisionError` in batch sampler | Insufficient data for batch_size=16 | Reduce to batch_size=4 |
| 6 | `AF_UNIX path too long` | PBS temp dir exceeds 108-char socket limit | Set `TMPDIR=/tmp/matey_$RANK` |

Each iteration involved: edit script locally, transfer via Globus, submit via IRI API, monitor output, diagnose error, repeat.

## 5. Successful Training Results

**Job ID:** 7059126.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov

| Parameter | Value |
|-----------|-------|
| Model | sViT_all2all (Spatial Vision Transformer) |
| Parameters | 7,675,146 |
| Dataset | MiniWeather thermalcollision2d |
| Train samples | 8,991 |
| Valid samples | 999 |
| Batch size | 4 (per GPU) |
| GPUs | 4 x A100 (1 node) |
| Epochs | 5 |
| Batches/epoch | 10 |
| Time/epoch | ~1.5 seconds |
| Final train loss | 0.1572 |
| Final valid loss | 0.1576 |
| GPU memory | 0.16 GB |
| Checkpoint | `Demo_MW/basic_config/demo_mw_svit/training_checkpoints/ckpt.tar` |

Training completed with a clean exit message: `MATEY MW SVIT job completed.`

## 6. Files Produced

### Job Scripts
- `jobs/polaris/matey/run.sh` — Single-job PBS script (MW SVIT)
- `jobs/polaris/matey/submit_batch_tests.sh` — Multi-test PBS batch script (5 models, 6 nodes)
- `jobs/polaris/matey/setup_env.sh` — Conda environment setup with proxy

### Configuration Files
- `config/Demo_MW_avit_polaris.yaml` — Adaptive ViT
- `config/Demo_MW_svit_polaris.yaml` — Spatial ViT (batch_size: 4)
- `config/Demo_MW_vit_polaris.yaml` — Standard ViT
- `config/Demo_SOLPS_vit_polaris.yaml` — SOLPS autoregressive
- `config/Demo_JHUTDB_TT_polaris.yaml` — JHTDB turbulence (pre-existing)

### Code Modifications
- `MATEY/matey/utils/distributed_utils.py` — Portable NCCL interface selection

## 7. Agentic Workflow Tools Used

The porting was performed using Claude Code with the following MCP integrations:

- **ALCF IRI API:** Job submission (`submit_job`), status checking (`list_jobs`), job cancellation (`cancel_job`)
- **Globus Transfer:** File transfer between local machine and Eagle storage (`globus_transfer`, `globus_ls`, `globus_transfer_status`)
- **Globus Authentication:** Token-based auth for both IRI and Globus Transfer APIs
- **Cron scheduling:** Automated job monitoring at 1-5 minute intervals
- **Local file editing:** Iterative script modifications between submissions

The entire workflow — from initial codebase analysis to successful training — was completed in approximately 55 minutes with 10 job submissions.

## 8. Recommendations for Future Runs

1. **Multi-node runs:** Test with `submit_batch_tests.sh` on 6 nodes. The `NCCL_SOCKET_IFNAME` may need adjustment from `lo` to the actual Slingshot interface for inter-node communication.

2. **Data staging:** Ensure sufficient data in both `train/` and `valid/` directories. The `train_val_test` split in the config further reduces the effective sample count.

3. **Batch size scaling:** With 4 GPUs and batch_size=4, the effective batch per step is 16 (with `accum_grad=2`, effective is 32). Adjust based on available data and memory.

4. **Performance:** The A100 GPUs are significantly underutilized at 0.16 GB memory usage. Consider larger batch sizes or higher resolution data for production runs.

5. **Environment caching:** After the first successful run, pip installs are cached in `~/.local/`. Subsequent jobs start much faster.
