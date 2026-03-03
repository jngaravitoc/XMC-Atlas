#!/usr/bin/env bash
set -e

VALUES=(100 160 223 290 348 419 480 481 88 884 \
        1585 1603 2240 2242 2259 2903 3468 3499 4159 4163 \
        4181 4797 4807 4822 1500 3372 4110 770 1000 355)

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

#parallel -j 30 "python build_spherical_basis.py {} halo --coefs_freq=1 --fit-type=initial
#--output_dir=sheng24_30 > sheng24_30/run_{}.log 2>&1" ::: "${VALUES[@]}"



python build_spherical_basis.py 100 bulge --coefs_freq=1 --fit-type=initial --output_dir=sheng24_30
