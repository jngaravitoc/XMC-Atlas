#!/usr/bin/env bash
#SBATCH --job-name=agama_density_field
#SBATCH --partition=batch
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/home/nicolas.garavito/projects/XMC-Atlas/scripts/agama_pipeline/logs/agama_density_field_%j.log

set -e

source /home/nicolas.garavito/activate_atlas_env.sh
export XMC_ATLAS=/home/nicolas.garavito/projects/XMC-Atlas
export XMC_ATLAS_SIMS=/data8/ngaravito/XMC-Atlas-sims/

cd /home/nicolas.garavito/projects/XMC-Atlas/scripts/agama_pipeline

# Single-sim job (Model_108, all snapshots found, all 3 AGAMA components --
# see agama_density_field.py's CLI). Loading coefficient files + evaluating
# the fitted expansion is much lighter than the particle-KDE path (no raw
# snapshot particle loads), so this is sized as a quick correctness check,
# not a batch run -- see submit_agama_density_fields_100.sh for the 100-sim
# array version.
python agama_density_field.py sheng sheng_iso 108
