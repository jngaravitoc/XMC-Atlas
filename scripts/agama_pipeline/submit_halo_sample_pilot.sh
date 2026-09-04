#!/usr/bin/env bash
#SBATCH --job-name=halo_sample_pilot
#SBATCH --partition=batch
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
# %1 (serial, not %5): all 5 tasks write into the same shared
# halo_sample_orbits.h5/halo_sample_params.h5 (one HDF5 group per sim_id),
# and h5py.File isn't safe for multiple processes writing to one file
# concurrently -- keep this serialized to avoid corrupting that output.
#SBATCH --array=1-5%1
#SBATCH --output=/home/nicolas.garavito/projects/XMC-Atlas/scripts/agama_pipeline/logs/halo_sample_pilot_%A_%a.log

set -e

source /home/nicolas.garavito/activate_atlas_env.sh
export XMC_ATLAS=/home/nicolas.garavito/projects/XMC-Atlas
export XMC_ATLAS_SIMS=/data8/ngaravito/XMC-Atlas-sims/

cd /home/nicolas.garavito/projects/XMC-Atlas/scripts/agama_pipeline

# Same 5 sim IDs (lines 1-5 of sim_ids_sheng_iso.txt: 13, 28, 49, 52, 56) as
# submit_agama_exp_pilot.sh's --array=1-5%5.
SIM_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" sim_ids_sheng_iso.txt)

python halo_sample.py sheng sheng_iso "$SIM_ID"
