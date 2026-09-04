#!/usr/bin/env bash
#SBATCH --job-name=agama_exp_arr
#SBATCH --partition=batch
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --array=1-1848%50
#SBATCH --output=/home/nicolas.garavito/projects/XMC-Atlas/scripts/agama_pipeline/logs/agama_exp_%A_%a.log

set -e

source /home/nicolas.garavito/activate_atlas_env.sh
export XMC_ATLAS=/home/nicolas.garavito/projects/XMC-Atlas
export XMC_ATLAS_SIMS=/data8/ngaravito/XMC-Atlas-sims/

cd /home/nicolas.garavito/projects/XMC-Atlas/scripts/agama_pipeline

SIM_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" sim_ids_sheng_iso.txt)

bash compute_agama_exp.sh "$SIM_ID"
