#!/usr/bin/env bash
set -e

#VALUES=(108)

#VALUES=(100 160 223 290 348 419 480 481 88 884 \
#585 1603 2240 2242 2259 2903 3468 3499 4159 4163\
#181 4797 4807 4822 1500 3372 4110 770 1000 355)


VALUES=(4163 181 4797 4807 4822 1500 3372 4110 770 1000 355)



export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

parallel -j 30 "python build_spherical_basis.py {} lmc --ncoefs=10 --fit-type=mean --output_dir" ::: "${VALUES[@]}"
