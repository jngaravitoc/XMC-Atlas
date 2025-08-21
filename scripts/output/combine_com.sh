#!/bin/bash
# Generalized concatenation script

# Base prefix of your run (adjust this to match the common start of your files)
prefix="MWLMC5_100M_b0_vir_OM3_G4"

# List of suffixes we want to merge
suffixes=(
    "_nba_halo_pot.txt"
    "_nba_disk_pot.txt"
    "_nba_bulge_pot.txt"
    "_pb_halo_pot.txt"
    "_pb_bulge_pot.txt"
    "_pb_disk_pot.txt"
    "_pb_halo_ssc.txt"
)

# Loop over each suffix
for suf in "${suffixes[@]}"; do
    # Find matching files, sort numerically by the snapshot index
    files=$(ls ${prefix}_*[0-9][0-9][0-9]${suf} 2>/dev/null | sort -t_ -k7,7n)

    if [[ -z "$files" ]]; then
        echo "⚠️ No files found for suffix $suf"
        continue
    fi

    # Build output filename by removing snapshot number
    outfile="${prefix}${suf}"

    # Concatenate into one file
    cat $files > "$outfile"

    echo "✅ Combined $(echo $files | wc -w) files into $outfile"
done

