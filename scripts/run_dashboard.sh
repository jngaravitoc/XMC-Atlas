#!/usr/bin/env bash
#set -e

VALUES=(100 160 223 290 348 419 480 481 88 884 \
        1585 1603 2240 2242 2259 2903 3468 3499 4159 4163 \
        4181 4797 4807 4822 1500 3372 4110 770 1000 355)

VALUES=(100 160 223 290 348 419 480 481 88 884 1585)


# Number of parallel jobs to run simultaneously
MAX_JOBS=11
OUTPUT_DIR="exp_expansions/figures/density_fields/"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to run dashboard for a single halo
run_dashboard() {
    local halo_id=$1
    echo "Starting halo_id: $halo_id"
    python spherical_basis_density_dashboards.py \
        --halo_id "$halo_id" \
        --output_dir "$OUTPUT_DIR" \
        --make_video
    echo "Finished halo_id: $halo_id"
}

export -f run_dashboard
export OUTPUT_DIR

# Run all values in parallel with job limit
for halo_id in "${VALUES[@]}"; do
    # Wait if we've reached max concurrent jobs
    while [ $(jobs -r -p | wc -l) -ge $MAX_JOBS ]; do
        sleep 5
    done
    
    # Launch job in background
    run_dashboard "$halo_id" &
done

# Wait for all background jobs to complete
wait

echo "All dashboards completed!"
