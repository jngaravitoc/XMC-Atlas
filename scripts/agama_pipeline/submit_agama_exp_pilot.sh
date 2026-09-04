#!/usr/bin/env bash
#SBATCH --job-name=agama_exp_pilot
#SBATCH --partition=batch
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=1-5%5
#SBATCH --output=/home/nicolas.garavito/projects/XMC-Atlas/scripts/agama_pipeline/logs/pilot_%A_%a.log

set -e

source /home/nicolas.garavito/activate_atlas_env.sh
export XMC_ATLAS=/home/nicolas.garavito/projects/XMC-Atlas
export XMC_ATLAS_SIMS=/data8/ngaravito/XMC-Atlas-sims/

cd /home/nicolas.garavito/projects/XMC-Atlas/scripts/agama_pipeline

SIM_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" sim_ids_sheng_iso.txt)

bash compute_agama_exp.sh "$SIM_ID"
