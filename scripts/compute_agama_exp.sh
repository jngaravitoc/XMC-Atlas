#!/usr/bin/env bash
set -e

VALUES=(100 160 223 290 348 419 480 481 88 884 \
        1585 1603 2240 2242 2259 2903 3468 3499 4159 4163 \
        4181 4797 4807 4822 1500 3372 4110 770 1000 355)


#loop over values and create directories
for val in "${VALUES[@]}"; do
    dir_name="suite_${val}"
    mkdir -p "$dir_name"
    echo "Created directory: $dir_name"
    python build_agama_expansion.py "$val" \
        --coefs_freq=1 \
        --output_dir="agama_expansions/${dir_name}"
    echo "Done computing agama expansion: $dir_name"
done

