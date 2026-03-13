#!/usr/bin/env bash
# Compute BFE + KDE density fields for multiple halos in parallel.
#
# Available arguments for compute_fields.py:
#   --halo_id HALO_ID        Halo model ID (required)
#   --output_dir DIR         Directory for output HDF5 files (required)
#   --suite SUITE            Simulation suite name (default: Sheng24)
#   --grid_range MIN MAX     Grid extent in kpc (default: -100 100)
#   --grid_bins N            Number of grid bins per axis (default: 20)
#   --skip_bfe               Skip BFE field computation
#   --skip_kde               Skip KDE field computation
#   --Ndens N                KDE neighbour count (default: 64)
#set -e

VALUES=(100 160 223 290 348 419 480 481 88 884 \
        1585 1603 2240 2242 2259 2903 3468 3499 4159 4163 \
        4181 4797 4807 4822 1500 3372 4110 770 1000 355)

# ---------- Configuration ----------
MAX_JOBS=30
OUTPUT_DIR="/n/nyx3/garavito/projects/XMC-Atlas/scripts/exp_expansions/density_fields/"
SUITE="Sheng24"
GRID_RANGE="-300 300"
GRID_BINS=20
NDENS=64
# Set to "--skip_bfe" or "--skip_kde" to skip; leave empty to compute both
SKIP_BFE="--skip_bfe"
SKIP_KDE=""
# ------------------------------------

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to run field computation for a single halo
run_fields() {
    local halo_id=$1

    echo "Starting field computation for halo_id: $halo_id"
    python compute_fields.py \
        --halo_id "$halo_id" \
        --output_dir "$OUTPUT_DIR" \
        --suite "$SUITE" \
        --grid_range $GRID_RANGE \
        --grid_bins "$GRID_BINS" \
        --Ndens "$NDENS" \
        $SKIP_BFE $SKIP_KDE
    echo "Finished field computation for halo_id: $halo_id"
}

export -f run_fields
export OUTPUT_DIR SUITE GRID_RANGE GRID_BINS NDENS SKIP_BFE SKIP_KDE

# Run all values in parallel with job limit
for halo_id in "${VALUES[@]}"; do
    while [ $(jobs -r -p | wc -l) -ge $MAX_JOBS ]; do
        sleep 5
    done
    run_fields "$halo_id" &
done

wait
echo "All field computations completed!"
