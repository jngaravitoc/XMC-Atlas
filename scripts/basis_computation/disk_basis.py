#Generate the disk basis for the MW

import pyEXP
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import sys

def print_hello(rank, size, name):
	msg = "Hello World! I am process {0} of {1} on {2}.\n"
	sys.stdout.write(msg.format(rank, size, name))

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()
print_hello(rank, size, name)


# Make the disk basis config

disk_config = """
---
id: cylinder
parameters:
  acyl: 3.5                        # exponential disk scale length, Martin's suggestion
  hcyl: 0.9                       # exponential disk scale height
  nmaxfid: 64                      # maximum radial order for spherical basis
  lmaxfid: 64                      # maximum harmonic order for spherical basis
  mmax: 0                          # maximum azimuthal order of cylindrical basis
  nmax: 12                          # maximum radial order of cylindrical basis, Mike's suggestion
  ncylodd: 3                       # vertically anti-symmetric basis functions, Martin's suggestion - nmax/2
  ncylnx: 64                    # grid points in radial direction
  ncylny: 32                      # grid points in vertical direction
  rnum: 200                        # radial quadrature knots for Gram matrix
  pnum: 0                          # azimuthal quadrature knots for Gram matrix
  tnum: 40                         # latitudinal quadrature knots for Gram matrix
  vflag: 0                         # verbose output flag
  logr: false                      # logarithmically spaced radial grid
  cachename: lmc01.cache.run0    # name of the basis cache file
...
"""

disk_basis = pyEXP.basis.Basis.factory(disk_config)


