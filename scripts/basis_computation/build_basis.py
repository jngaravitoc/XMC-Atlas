import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pyEXP
import nba
import gala.potential as gp
#import EXPtools

from plot_helpers import plot_profiles
from fit_density import fit_profile, fit_density_profile
from basis_utils import make_basis

sys.path.append("../")
sys.path.append("../BFE_computation")

from sanity_checks import check_monotonic_profiles, check_monotonic_contiguous_snapshots
from ios_nbody_sims import load_particle_data
from compute_bfe_helpers import (
    load_sheng24_exp_center, 
    get_snapshot_suffixes
    )

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

def make_density_profile(pos, mass, rmin=0.1, rmax=400, nbins=500):
    """
    TODO: this profile is constructed by building the data.
    try this also  by using KDTREE
    """
    rbins_linear = np.linspace(rmin, rmax, nbins) 
    profile = nba.structure.Profiles(pos, rbins_linear)
    r_profile, dens_profile = profile.density(mass=mass)
    return r_profile, dens_profile

if __name__ == "__main__":
    #--------------------------
    # Define parameters  
    # -------------------------
    # Simulation files, paths
    SIM_ID = 108
    SNAPSHOT_PATH = "../../../../XMC-Atlas-sims/Sheng/Model_{}".format(SIM_ID)
    SNAPNAME = "snapshot"
    nsnap = 0 
    suite = "Sheng24"
    SIM_PARAMS_PATH = '../../suites/Sheng24/orbits'
    SIM_PARAMS_FILE = 'MW_LMC_orbits_iso.txt'
    # profile fitting
    
    rmin = 0.1 # should be 4*softening
    rmax = 500
    nbins = 101

    # pipeline params
    paranoid = True
    figure_name = 'sim_{:04d}_density_profile_evolution.png'.format(SIM_ID)
    

    #--------------------------
    # 1. Load halo centers
    # -------------------------


    sim_params = load_sheng24_exp_center(
        SIM_PARAMS_PATH, 
        SIM_PARAMS_FILE,
        SIM_ID, return_vel=False)

    tsim = sim_params['time']
    mw_center = sim_params['mw_center']
    lmc_center = sim_params['lmc_center']
    
    print("-> Done loading simulation centers")

    snap_suffixes = get_snapshot_suffixes(SNAPSHOT_PATH, prefix=SNAPNAME+"_")
    NSNAPS = len(snap_suffixes)
    
    if paranoid == True:
        snap_check, missing_snaps = check_monotonic_contiguous_snapshots(snap_suffixes)
        print(f"[paranoid] {NSNAPS} snapshots found in {SNAPSHOT_PATH}")
        print(f"[paranoid] Snapshots are monotonic:", snap_check)
        if snap_check == False:
            print(f"[error] snapshots {missing_snaps} are missing")
        assert NSNAPS == len(tsim), "number of snapshots found differ from centers"
    #----------------------------------------
    # 2. Load particle data and recenter halo
    # ----------------------------------------

    rho_part_all = np.zeros((NSNAPS, nbins-1))

    for i in range(NSNAPS):
        snapshot = SNAPNAME + "{:03d}.hdf5".format(i)
        data = load_particle_data(SNAPSHOT_PATH, SNAPNAME, ['MWhalo'], nsnap=i, suite=suite)
        pos = data['MWhalo']['pos']
        mass = data['MWhalo']['mass']
        pos_center = pos - mw_center[i]
        
        #--------------------------
        # 3. Compute density profile 
        # -------------------------

        r_bins_part, rho_part_all[i]  = make_density_profile(pos_center, mass, rmin=rmin, rmax=rmax, nbins=nbins)
    print("-> Done computing density profiles")

    if paranoid == True: 
        check_center, fail_ids = check_monotonic_profiles(rho_part_all[:,:10])
        print("[paranoid] centering check passing:", check_center)
        print("[paranoid] Fail indices:", fail_ids)
        if len(fail_ids)>0:
            _ = plot_profiles(
                r_bins_part, 
                rho_part_all[fail_ids], 
                time=tsim[fail_ids], 
                title=f"Non monotonic density profiles in Halo {SIM_ID}", 
                filename="non_monotonic_density_profiles.png")
        
        assert check_center == True

    #--------------------------
    # 4. Fit density profile 
    # -------------------------
    
    rho_fit, fit_params = fit_profile(r_bins_part, rho_part_all[0])
    

    if paranoid == True:
        _ = plot_profiles(
            r_bins_part, 
            rho_part_all, 
            time=tsim,
            r_fit = r_bins_part,
            rho_fit = rho_fit,
            title=f"Halo {SIM_ID} density profile", 
            filename=figure_name)
   
    
    #--------------------------
    # 5. Compute Halo basis 
    # -------------------------
    
    r_basis = np.linspace(0.01, 500, 500)
    rho_fit = fit_density_profile(r_basis, *fit_params)

    basis_config = {
    "basis_id": "sphereSL",
    "numr" : 500,
    "rmin" : 0.01,
    "rmax" : 500,
    "Lmax" : 1,
    "nmax" : 5,
    "rmapping" : 1.0,
    "modelname" : "test_model.txt",
    "cachename" : "test_cache.txt",
    }


    bconfig = make_basis(
        r_basis, 
        rho_fit, 
        Mtotal=1, 
        basis_params= basis_config)
    pyEXP.basis.Basis.factory(bconfig)

    #--------------------------
    # 6. Compute coefficients
    # -------------------------

    #--------------------------
    # 7. Compute MISE
    # -------------------------

