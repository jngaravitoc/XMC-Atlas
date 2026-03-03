"""
Pipeline to compute optimal basis for the Sheng+24 simulation suite.

Author: github.com/jngaravitoc

Usage: python build_basis.py sim_id halo_component
"""

import os
import sys
import numpy as np
from pathlib import Path
import argparse
# third-party
from mpi4py import MPI
import pyEXP

# BFE local libraries
THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR / "exp_pipeline"))




def main(outpath):
    """
    Build the disk basis for the specified simulation/component.
    Parameters
	----------
	SIM_ID : int
		Simulation ID.
	component : str
		Galaxy component (e.g., 'halo').
	ncoefs : int
		Number of coefficient samples.
	fit_type : str, optional
		Method for fitting coefficients (default: 'median').
	compute_mise : bool, optional
		Whether to compute the MISE (default: False).
    """
    # PATHS:


    def print_hello(rank, size, name):
        msg = "Hello World! I am process {0} of {1} on {2}.\n"
        sys.stdout.write(msg.format(rank, size, name))

    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    name = MPI.Get_processor_name()
    print_hello(rank, size, name)


    disk_config="""
---
id: cylinder
parameters:
  acyl: 3.8
  hcyl: 0.28
  mmax: 6
  nmax: 24
  nmaxfid: 24
  lmaxfid: 24
  ncylodd: 12
  ncylnx: 16
  ncylny: 16
  rnum: 200
  pnum: 0
  tnum: 40
  vflag: 0
  logr: false
  cachename: cache_disk.sheng24
...
"""

    # move to outopath folder
    os.chdir(outpath)
    basis = pyEXP.basis.Basis.factory(disk_config)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build spherical basis functions for extrnal simulations."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        nargs="?",
        default="test_halo_108",
        help="Directory output name for datasets"
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    main(
        outpath=args.output_dir,
    )

    
