#!/usr/bin/env python
"""Generate toy JHTDB isotropic1024fine HDF5 data for MATEY testing.

Creates small 256x256x256 files with 4 fields (Vx, Vy, Vz, Pressure)
across multiple timesteps, matching the naming convention:
  isotropic1024fine_t_<t>_<t>_x_1_256_y_1_256_z_1_256.h5

Each file contains:
  - Velocity_XXXX: shape [nz, ny, nx, 3]
  - Pressure_XXXX: shape [nz, ny, nx, 1]
"""
import numpy as np
import h5py
import os
import argparse

def generate_toy_data(output_dir, n_timesteps=8, nx=256, ny=256, nz=256):
    os.makedirs(output_dir, exist_ok=True)

    for t in range(n_timesteps):
        fname = f"isotropic1024fine_t_{t}_{t}_x_1_{nx}_y_1_{ny}_z_1_{nz}.h5"
        fpath = os.path.join(output_dir, fname)
        print(f"Generating {fname} ...")

        with h5py.File(fpath, 'w') as f:
            ts_key = f"{t:04d}"
            # Random velocity field [nz, ny, nx, 3]
            vel = np.random.randn(nz, ny, nx, 3).astype(np.float32) * 0.1
            f.create_dataset(f"Velocity_{ts_key}", data=vel)
            # Random pressure field [nz, ny, nx, 1]
            pres = np.random.randn(nz, ny, nx, 1).astype(np.float32) * 0.01
            f.create_dataset(f"Pressure_{ts_key}", data=pres)

    print(f"Generated {n_timesteps} files in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./toy_data/isotropic1024fine",
                        help="Output directory for toy HDF5 files")
    parser.add_argument("--n_timesteps", type=int, default=8,
                        help="Number of timesteps to generate")
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--ny", type=int, default=256)
    parser.add_argument("--nz", type=int, default=256)
    args = parser.parse_args()
    generate_toy_data(args.output_dir, args.n_timesteps, args.nx, args.ny, args.nz)
