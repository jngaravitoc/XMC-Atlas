#!/usr/bin/env bash

VALUES=(100 160 223 290 348 419 480 481 88 884 \
      1585 1603 2240 2242 2259 2903 3468 3499 4159 4163 \
      4181 4797 4807 4822 1500 3372 4110 770 1000 355)


VALUES=(108)
#VALUES=(160 290 355 480 481 770 1000 3372 4163 4181 4822)

# ---------- Configuration ----------
# ------------------------------------
# Create output directory

# Function to run field computation for a single halo
run_fields() {
    local halo_id=$1

    python evaluate_fields_quality.py \
        "$halo_id"
}

export -f run_fields

# Run all values sequentially
for halo_id in "${VALUES[@]}"; do
    echo "Running halo ${halo_id}..."
    run_fields "$halo_id"
done

