#!/usr/bin/env bash
# Create GIF animations from density field dashboard PNGs.
#
# For each halo ID, checks that all 101 frames (000.png – 100.png) exist
# before running convert.  Skips halos with missing frames.
#
# Usage:
#   cd scripts/exp_expansions/figures/density_fields/
#   bash make_gifs.sh

set -e

VALUES=(100 160 223 290 348 419 480 481 88 884 \
        1585 1603 2240 2242 2259 2903 3468 3499 4159 4163 \
        4181 4797 4807 4822 1500 3372 4110 770 1000 355)

EXPECTED_FRAMES=101   # 000 to 100 inclusive

for halo_id in "${VALUES[@]}"; do
    hid=$(printf "%04d" "$halo_id")
    prefix="halo_${hid}_density_field_center"
    gif_name="halo_${hid}_density_field.gif"

    # Count how many frames exist
    n_frames=$(ls "${prefix}_"*.png 2>/dev/null | wc -l)

    if [[ "$n_frames" -ne "$EXPECTED_FRAMES" ]]; then
        echo "SKIP  halo ${hid}: found ${n_frames}/${EXPECTED_FRAMES} frames"
        continue
    fi

    # Verify the sequence is contiguous (000 .. 100)
    missing=0
    for i in $(seq 0 100); do
        frame=$(printf "%s_%03d.png" "$prefix" "$i")
        if [[ ! -f "$frame" ]]; then
            echo "SKIP  halo ${hid}: missing frame ${frame}"
            missing=1
            break
        fi
    done

    if [[ "$missing" -eq 1 ]]; then
        continue
    fi

    echo "CREATE ${gif_name} (${n_frames} frames)..."
    convert -delay 5 -loop 0 "${prefix}_"*.png "${gif_name}"
    echo "  done"
done

echo "All GIFs processed."
