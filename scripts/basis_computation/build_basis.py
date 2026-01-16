import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pyEXP
import nba
import gala.potential as gp

from plot_helpers import plot_profiles
from fit_density import fit_profile, fit_density_profile
from basis_utils import make_basis
from basis_fidelity import bfe_density_profiles 

sys.path.append("../")
sys.path.append("../BFE_computation")

from atlas_data_products import write_density_profiles
from sanity_checks import check_monotonic_profiles, check_monotonic_contiguous_snapshots
from ios_nbody_sims import load_particle_data
from compute_bfe_helpers import (
    load_sheng24_exp_center, 
    get_snapshot_suffixes
    )

#from exp_coefficients import compute_exp_coefs_parallel

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

def make_density_profile(pos, mass, r_edges):
    """
    TODO: this profile is constructed by building the data.
    try this also  by using KDTREE
    """
    profile = nba.structure.Profiles(pos, r_edges)
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
    softening = 0.6 #
    # profile fitting params
    rmin = 0.1 # should be 4*softening
    rmax = 500
    nbins = 101
    component = 'halo'

    # basis params:
    basis_filenames = "basis_{}_{:04d}.yaml".format(component, SIM_ID)
    nbins_basis = 1000
    lmax = int(1)
    nmax = int(10)
    rmapping = 1.0
    modelname = 'modelname_{}_{:04d}.txt'.format(component, SIM_ID)
    cachename = 'cache_{}_{:04d}.txt'.format(component, SIM_ID)

    # Coefficients:
    coefs_filename = '{}_{:04d}_coefficients.h5'.format(component, SIM_ID)
    
    # pipeline params:
    paranoid = True
    figure_name = '{}_{:04d}_density_profile_evolution.png'.format(component, SIM_ID)
    outpath = "./test_sheng24/"
    particle_profiles_filename = "density_profiles_sheng24.h5"
    bfe_profiles_filename = "bfe_density_profiles_sheng24.h5"
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
    r_bins_model = np.linspace(rmin, rmax, nbins)

    for i in range(NSNAPS):
        snapshot = SNAPNAME + "{:03d}.hdf5".format(i)
        data = load_particle_data(SNAPSHOT_PATH, SNAPNAME, ['MWhalo'], nsnap=i, suite=suite)
        pos = data['MWhalo']['pos']
        mass = data['MWhalo']['mass']
        print(data["MWhalo"].keys())
        pos_center = pos - mw_center[i]
        
        #--------------------------
        # 3. Compute density profile 
        # -------------------------

        r_bins_part, rho_part_all[i]  = make_density_profile(pos_center, mass, r_bins_model)
    print("-> Done computing density profiles")
    
    write_density_profiles(
        suite_id=SIM_ID, 
        snaps=np.arange(0, NSNAPS, 1), 
        rbins=r_bins_part,
        profiles=rho_part_all,
        filename=outpath+particle_profiles_filename)

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
    
    # TODO: do the fit for the mean
    rho_fit, fit_params = fit_profile(r_bins_part, rho_part_all[0])
    

    if paranoid == True:
        _ = plot_profiles(
            r_bins_part, 
            rho_part_all, 
            time=tsim,
            r_fit = r_bins_part,
            rho_fit = rho_fit,
            title=f"Halo {SIM_ID} density profile", 
            filename=outpath+figure_name)
   
    
    #--------------------------
    # 5. Compute Halo basis 
    # -------------------------
    
    r_basis = np.linspace(rmin, rmax, nbins_basis)
    rho_fit = fit_density_profile(r_basis, *fit_params)

    basis_config = {
    "basis_id": "sphereSL",
    "numr" : nbins_basis,
    "rmin" : rmin,
    "rmax" : rmax,
    "Lmax" : lmax,
    "nmax" : nmax,
    "rmapping" : rmapping,
    "modelname" : modelname,
    "cachename" : cachename,
    }

    #TODO define Mtotal
    bconfig = make_basis(
        r_basis, 
        rho_fit, 
        Mtotal=1, 
        basis_params= basis_config,
        basis_filename = basis_filenames)

    basis = pyEXP.basis.Basis.factory(bconfig)

    #--------------------------
    # 6. Compute coefficients
    # -------------------------
    # TODO: why if this import is earlier in the script
    # the cache is not build in step 5?

    from exp_coefficients import compute_exp_coefs_parallel

    compname = 'halo'
    runtag   = 'run1'
    time     = 0.0
    basis.enableCoefCovariance(pcavar=True, nsamples=100, covar=True)
    basis.writeCoefCovariance(compname, runtag, time)

    gadget_particle_mass = 1e10
    units = [('mass', 'Msun', gadget_particle_mass),
             ('length', 'kpc', 1.0),
             ('velocity', 'km/s', 1.0),
             ('G', 'mixed', 43007.1)]

    
     
    for i in range(0, NSNAPS, 10):
        snapshot = SNAPNAME + "{:03d}.hdf5".format(i)
        data = load_particle_data(SNAPSHOT_PATH, SNAPNAME, ['MWhalo'], nsnap=i, suite=suite)
        pos = data['MWhalo']['pos']
        mass = data['MWhalo']['mass']
        pos_center = pos - mw_center[i]
        

        compute_exp_coefs_parallel(
            data["MWhalo"],
            basis,
            component,
            coefs_filename,
            unit_system=units)
    
    #Read coefficients
    coefs = pyEXP.coefs.Coefs.factory(coefs_filename)
    coefs_times = coefs.Times()
    #--------------------------------
    # 7. Compute BFE density profile
    # ------------------------------
    rho_bfe_t = np.zeros((NSNAPS, len(r_bins_part)))
    print(len(coefs_times))
    for t in range(len(coefs_times)):
        rho_bfe_t[i] = bfe_density_profiles(
            basis, 
            coefs, 
            r_bins=r_bins_part, 
            time=coefs_times[t])
    
    write_density_profiles(
        suite_id=SIM_ID, 
        snaps=np.arange(0, NSNAPS, 1), 
        rbins=r_bins_part, 
        profiles=rho_bfe_t,
        filename=outpath+bfe_profiles_filename)

    #--------------------------
    # 8. Compute MISE
    # -------------------------

    
