#!/usr/bin/env bash
#SBATCH --job-name=halo_sample
#SBATCH --partition=batch
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
# %1 (serial, not higher concurrency): all tasks write into the same shared
# halo_sample_orbits.h5/halo_sample_params.h5 (one HDF5 group per sim_id),
# and h5py.File isn't safe for multiple processes writing to one file
# concurrently -- keep this serialized to avoid corrupting that output.
# Cheap per sim_id (~20-30s from the 5-sim pilot, job 9826), so the full
# 1848-sim run is expected to take on the order of 10-13 hours serial.
#
# Batched 10 sims per array task (185 tasks for 1848 sims), not one task per
# sim: this cluster's MaxArraySize is 1001 (scontrol show config), so a
# 1-1848 array is rejected outright ("Invalid job array specification").
# Batching keeps the array size well under that limit; each task loops over
# its slice of sim_ids_sheng_iso.txt serially.
#SBATCH --array=1-185%1
#SBATCH --output=/home/nicolas.garavito/projects/XMC-Atlas/scripts/agama_pipeline/logs/halo_sample_%A_%a.log

set -e

source /home/nicolas.garavito/activate_atlas_env.sh
export XMC_ATLAS=/home/nicolas.garavito/projects/XMC-Atlas
export XMC_ATLAS_SIMS=/data8/ngaravito/XMC-Atlas-sims/

cd /home/nicolas.garavito/projects/XMC-Atlas/scripts/agama_pipeline

# Same sim_ids_sheng_iso.txt (1848 sims) used by compute_agama_exp.sh via
# submit_agama_exp_array.sh. BATCH_SIZE=10 sims per array task; sed's end
# line is allowed to run past EOF (it just stops at the last line), so the
# final, partial batch (task 185: lines 1841-1848, 8 sims) needs no special
# case.
BATCH_SIZE=10
START=$(( (SLURM_ARRAY_TASK_ID - 1) * BATCH_SIZE + 1 ))
END=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE ))

for SIM_ID in $(sed -n "${START},${END}p" sim_ids_sheng_iso.txt); do
    python halo_sample.py sheng sheng_iso "$SIM_ID"
done
