#!/usr/bin/env bash
set -e

# Usage: bash run_30_basis.sh <component>
#   component: halo, bulge, or lmc

COMPONENT="${1:?Usage: bash run_30_basis.sh <halo|bulge|lmc>}"

if [[ "$COMPONENT" != "halo" && "$COMPONENT" != "bulge" && "$COMPONENT" != "lmc" ]]; then
    echo "Error: component must be one of: halo, bulge, lmc"
    exit 1
fi

VALUES=(100 160 223 290 348 419 480 481 88 884 \
        1585 1603 2240 2242 2259 2903 3468 3499 4159 4163 \
        4181 4797 4807 4822 1500 3372 4110 770 1000 355)

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

OUTPUT_ROOT="/n/nyx3/garavito/projects/XMC-Atlas/exp_expansions"
mkdir -p "${OUTPUT_ROOT}/basis" "${OUTPUT_ROOT}/coefficients"

parallel -j 30 "python build_spherical_basis.py {} ${COMPONENT} --ncoefs=1 --fit-type=initial \
--output_dir=${OUTPUT_ROOT} > ${OUTPUT_ROOT}/basis/run_{}_${COMPONENT}.log 2>&1" ::: "${VALUES[@]}"
