#!/usr/bin/env bash

VALUES=(100 160 223 290 348 419 480 481 88 884 \
       1585 1603 2240 2242 2259 2903 3468 3499 4159 4163 \
       4181 4797 4807 4822 1500 3372 4110 770 1000 355)
# ---------- Configuration ----------
# ------------------------------------
# Create output directory

# Function to run field computation for a single halo
run_fields() {
    local halo_id=$1

    python compute_fields.py \
        --halo_id "$halo_id" \
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
