#!/usr/bin/env bash
set -e


#VALUES=(160 223 290 348 419 480 481 88 884 \
#585 1603 2240 2242 2259 2903 3468 3499 4159 4163\
#181 4797 4807 4822 1500 3372 4110 770 1000 355)


#VALUES=(160 223 290 348 419 480 481 88 884 \
VALUES=(4797 4807 4822 1500 3372 4110 770 1000 355)


#mpirun -np 30 python halos_bfe_from_same_basis.py 108 lmc
#loop over values and create directories
for val in "${VALUES[@]}"; do
    mpirun -np 30 python halos_bfe_from_same_basis.py "$val" lmc
done
