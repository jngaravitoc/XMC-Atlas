#!/usr/bin/env bash
#SBATCH --job-name=agama_density_fields_100
#SBATCH --partition=batch
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --array=1-100%20
#SBATCH --output=/home/nicolas.garavito/projects/XMC-Atlas/scripts/agama_pipeline/logs/agama_density_fields_100_%A_%a.log

set -e

source /home/nicolas.garavito/activate_atlas_env.sh
export XMC_ATLAS=/home/nicolas.garavito/projects/XMC-Atlas
export XMC_ATLAS_SIMS=/data8/ngaravito/XMC-Atlas-sims/

cd /home/nicolas.garavito/projects/XMC-Atlas/scripts/agama_pipeline

# Same 100-sim sample as particle_fields/submit_density_fields_100.sh (first
# 100 lines of sim_ids_sheng_iso.txt) -- these are exactly the sims job 9725
# already fit AGAMA coefficients for, so scripts/agama_expansions/suite_<id>/
# is populated for all 100 (confirmed on disk before writing this script).
# Each task writes to its own agama_density_<sim_id>.h5 (see
# agama_density_field.py), so tasks are safe to run concurrently -- no %1
# serialization needed. Loading coefficients + evaluating the fitted
# expansion is much lighter than the particle-KDE path (no raw particle
# loads), hence the modest time/mem here vs. the KDE batch's 00:40:00/32G.
SIM_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" sim_ids_sheng_iso.txt)

python agama_density_field.py sheng sheng_iso "$SIM_ID"
