#!/usr/bin/env bash
set -e

python build_spherical_basis.py 108 lmc --ncoefs=1 --fit-type=initial --output_dir=./
