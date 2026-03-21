#!/usr/bin/env bash
set -e
VALUES=(108)

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

parallel -j 1 "python build_spherical_basis.py {} lmc --ncoefs=10 --fit-type=mean --output_dir" ::: "${VALUES[@]}"
