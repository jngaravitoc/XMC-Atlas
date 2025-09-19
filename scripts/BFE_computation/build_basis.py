#!/usr/bin/env python3
"""
Build a basis expansion models
"""

import os
import numpy as np
import gala.potential as gp
import pyEXP
import EXPtools


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


def main():
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


if __name__ == "__main__":
    main()
