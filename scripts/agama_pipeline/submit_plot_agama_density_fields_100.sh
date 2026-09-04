#!/usr/bin/env bash
#SBATCH --job-name=plot_agama_density_fields_100
#SBATCH --partition=batch
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/home/nicolas.garavito/projects/XMC-Atlas/scripts/agama_pipeline/logs/plot_agama_density_fields_100_%j.log

set -e

source /home/nicolas.garavito/activate_atlas_env.sh
export XMC_ATLAS=/home/nicolas.garavito/projects/XMC-Atlas
export XMC_ATLAS_SIMS=/data8/ngaravito/XMC-Atlas-sims/

cd /home/nicolas.garavito/projects/XMC-Atlas/scripts/agama_pipeline

# Single job, no array: plot_agama_density_fields_100.py loops over all 100
# sims itself (one HDF5 read + one figure per sim, snapshots 0/50/90) -- same
# reasoning as particle_fields/submit_plot_density_fields_100.sh: reading
# already-computed AGAMA density fields is cheap I/O, no BFE evaluation, so
# the full 100-sim pass is a few minutes at most, well within one task.
python plot_agama_density_fields_100.py
