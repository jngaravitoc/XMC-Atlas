#!/usr/bin/env bash
# Compute BFE + KDE density fields for multiple halos in parallel.
# Analogous to run_dashboard.sh but calls compute_fields.py only.
#set -e

VALUES=(100 160 223 290 348 419 480 481 88 884 \
        1585 1603 2240 2242 2259 2903 3468 3499 4159 4163 \
        4181 4797 4807 4822 1500 3372 4110 770 1000 355)

# Number of parallel jobs to run simultaneously
MAX_JOBS=11
OUTPUT_DIR="exp_expansions/figures/density_fields/"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to run field computation for a single halo
run_fields() {
    local halo_id=$1
    echo "Starting field computation for halo_id: $halo_id"
    python compute_fields.py \
        --halo_id "$halo_id" \
        --output_dir "$OUTPUT_DIR"
    echo "Finished field computation for halo_id: $halo_id"
}

export -f run_fields
export OUTPUT_DIR

# Run all values in parallel with job limit
for halo_id in "${VALUES[@]}"; do
    while [ $(jobs -r -p | wc -l) -ge $MAX_JOBS ]; do
        sleep 5
    done
    run_fields "$halo_id" &
done

wait
echo "All field computations completed!"

# ---- Now generate dashboards ----
echo ""
echo "Generating dashboards..."

run_dashboard() {
    local halo_id=$1
    echo "Starting dashboard for halo_id: $halo_id"
    python make_dashboard.py \
        --halo_id "$halo_id" \
        --output_dir "$OUTPUT_DIR" \
        --make_video
    echo "Finished dashboard for halo_id: $halo_id"
}

export -f run_dashboard

for halo_id in "${VALUES[@]}"; do
    while [ $(jobs -r -p | wc -l) -ge $MAX_JOBS ]; do
        sleep 5
    done
    run_dashboard "$halo_id" &
done

wait
echo "All dashboards completed!"
