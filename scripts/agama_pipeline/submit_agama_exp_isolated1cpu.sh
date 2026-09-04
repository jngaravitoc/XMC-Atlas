#!/usr/bin/env bash
#SBATCH --job-name=agama_exp_iso1cpu
#SBATCH --partition=batch
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=/home/nicolas.garavito/projects/XMC-Atlas/scripts/agama_pipeline/logs/iso1cpu_%j.log

set -e

source /home/nicolas.garavito/activate_atlas_env.sh
export XMC_ATLAS=/home/nicolas.garavito/projects/XMC-Atlas
export XMC_ATLAS_SIMS=/data8/ngaravito/XMC-Atlas-sims/

cd /home/nicolas.garavito/projects/XMC-Atlas/scripts/agama_pipeline

# Solo run, no concurrent siblings: isolates pure cpus-per-task=1 scaling
# from the I/O contention seen in the 5-way concurrent pilot (job 9719).
# sim_id=88: untouched this session (not in the dedup group, not Model_108),
# so it's not benefiting from OS page-cache warmth like a repeat sim would.
bash compute_agama_exp.sh 88
