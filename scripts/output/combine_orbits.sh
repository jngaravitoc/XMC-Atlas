#!/bin/bash

PREFIX="MWLMC3_100M_b0_vir_OM3_G4"

# Extract unique suffixes after the second underscore after PREFIX
for suffix in $(ls ${PREFIX}_*_*\.txt 2>/dev/null \
                  | sed -E "s/^${PREFIX}_[^_]+_(.*)\.txt$/\1/" \
                  | sort -u); do
    echo "Combining all files ending with: $suffix"
    cat ${PREFIX}_*_${suffix}.txt > ${PREFIX}_${suffix}.txt
    rm ${PREFIX}_*_${suffix}.txt 
done

