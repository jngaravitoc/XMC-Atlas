#!/usr/bin/env python3
"""
Build a basis expansion models
"""

import os
import sys
import numpy as np
from mpi4py import MPI
import gala.potential as gp
import pyEXP
#import EXPtools


# -------------------
# User parameters
# -------------------
MODEL_NAME = "GC21_MW_DM_halo.txt"
POTENTIAL_NAME = "../../GC21/GC21LMC1.potential"
NMAX = 10
LMAX = 8
N_RBINS = 400
BASIS_NAME = f"GC21_MW_DM_halo_{NMAX}_{LMAX}.yaml"
CACHE_NAME = f".cache_GC21_MW_DM_halo_{NMAX}_{LMAX}"
CONFIG_NAME = f"config_GC21_MW_DM_halo_{NMAX}_{LMAX}.yaml"
OUTPATH = "../../GC21/basis/"


def compute_halo_basis():
    """Main driver for building the halo basis expansion."""
    
    if not os.path.exists(POTENTIAL_NAME):
        raise FileNotFoundError(f"{POTENTIAL_NAME} not found")

    # Load potential
    GC21_pot = gp.potential.load(POTENTIAL_NAME)
    GC21_MW_DMhalo = GC21_pot["halo"]

    # Halo parameters
    Mtot = GC21_MW_DMhalo.parameters["m"].value
    rbins = np.linspace(1e-3, 300, N_RBINS)

    # Halo profile
    halo_profile = GC21_MW_DMhalo.density(rbins)

    # TODO: define rcen properly
    rcen = 0.0  

    # Build model
    model = EXPtools.make_model(
        rcen,
        halo_profile,
        Mtotal=Mtot,
        output_filename=OUTPATH+MODEL_NAME,
        physical_units=True,
    )
    R = model["radius"]

    # Build basis
    config = EXPtools.make_config(
        basis_id="sphereSL",
        lmax=LMAX,
        nmax=NMAX,
        rmapping=R[-1],
        modelname=OUTPATH+MODEL_NAME,
        cachename=OUTPATH+CACHE_NAME,
    )

    basis = pyEXP.basis.Basis.factory(config)
    EXPtools.write_basis(config, OUTPATH+BASIS_NAME)

    print(f"Basis written to {OUTPATH+BASIS_NAME}")



def compute_GC21_disk_basis():
	size = MPI.COMM_WORLD.Get_size()
	rank = MPI.COMM_WORLD.Get_rank()
	name = MPI.Get_processor_name()
	#sys.stdout.write(msg.format(rank, size, name))

	# Make the disk basis config

	disk_config = """
	---
	id: cylinder
	parameters:                         
	  acyl: 4.5                         # exponential disk scale length, Martin's suggestion
	  hcyl: 0.9                         # exponential disk scale height
	  lmaxfid: 64                       # maximum harmonic order for spherical basis
	  nmaxfid: 64                       # maximum radial order for spherical basis
	  mmax: 6                           # maximum azimuthal order of cylindrical basis
	  nmax: 12                          # maximum radial order of cylindrical basis, Mk's suggestion
	  ncylnx: 256                       # grid points in radial direction
	  ncylny: 128                       # grid points in vertical direction
	  ncylodd: 3                        # vertically anti-symmetric basis functions, Ma suggestion - nmax/2
	  rnum: 200                         # radial quadrature knots for Gram matrix
	  pnum: 0                           # azimuthal quadrature knots for Gram matrix
	  tnum: 80                          # latitudinal quadrature knots for Gram matrix
	  vflag: 0                          # verbose output flag
	  logr: false                       # logarithmically spaced radial grid
      ashift: 0                         # 
	  cachename: test_GC21_mwdisk.cache.basis   # name of the basis cache file
	...
	"""
	disk_basis = pyEXP.basis.Basis.factory(disk_config)



if __name__ == "__main__":
    #compute_halo_basis()
	compute_GC21_disk_basis()
