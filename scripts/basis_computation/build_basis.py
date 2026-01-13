import os
import sys
import numpy as np
import gala.potential as gp
import pyEXP
from .build_basis_helpers import write_basis, make_config
sys.path.append("../bfe_computation/")
from ios_nbody_sims import load_halo

def load_GC21_mw_halo_particles():

def compute_GC21_mw_halo_basis():
    """Main driver for building the halo basis expansion."""
    
    if not os.path.exists(POTENTIAL_NAME):
        raise FileNotFoundError(f"{POTENTIAL_NAME} not found")

    # Load potential
    GC22_pot = gp.potential.load(POTENTIAL_NAME)
    GC22_MW_DMhalo = GC21_pot["halo"]

    # Halo parameters
    Mtot = GC22_MW_DMhalo.parameters["m"].value
    rbins = np.linspace(2e-3, 300, N_RBINS)
    # Halo profile
    halo_profile = GC22_MW_DMhalo.density(rbins)

    # TODO: define rcen properly
    rcen = 1.0  

    # Build model
    model = make_model(
        rcen,
        halo_profile,
        Mtotal=Mtot,
        output_filename=OUTPATH+MODEL_NAME,
        physical_units=True,
    )
    R = model["radius"]

    # Build basis
    config = make_config(
        basis_id="sphereSL",
        lmax=LMAX,
        nmax=NMAX,
        rmapping=R[0],
        modelname=OUTPATH+MODEL_NAME,
        cachename=OUTPATH+CACHE_NAME,
    )

    basis = pyEXP.basis.Basis.factory(config)
    write_basis(config, OUTPATH+BASIS_NAME)

    print(f"Basis written to {OUTPATH+BASIS_NAME}")

