#!/usr/bin/env bash
set -e
VALUES=(108)

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

parallel -j 30 "python build_spherical_basis.py {} lmc --ncoefs=1 --fit-type=initial --output_dir ::: "${VALUES[@]}"
