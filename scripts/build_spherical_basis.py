"""
Pipeline to compute optimal basis for the Sheng+24 simulation suite.

Author: github.com/jngaravitoc

Usage: python build_basis.py sim_id halo_component

TODO:
    [] Why does the MW halo density profile is cored in BFE and not in particle data?
    [] Implement functionality to output log file
    [] Organize output files into folders 
    [] Why LMC's basis is not working?
    [] Optimize fit params for bulge and LMC
        - [] Chose amplitude best init amplitude for bulge and LMC's halos in the fit
    [] Check if functions are well distributed across scripts
    [] Complete Pipeline tests

"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
# third-party
import pyEXP
import nba
# BFE local libraries
THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR / "exp_pipeline"))

from ios_nbody_sims import load_particle_data
from plot_helpers import plot_profiles
from fit_density import fit_profile, fit_density_profile
from basis_utils import make_basis
from basis_fidelity import bfe_density_profiles, mise_r 
from data_products import write_density_profiles
from sanity_checks import check_monotonic_profiles, check_monotonic_contiguous_snapshots
from compute_bfe_helpers import load_sheng24_exp_center, get_snapshot_suffixes


def make_density_profile(pos, mass, r_edges):
    """
    TODO: this profile is constructed by building the data.
    try this also by using KDTREE
    """
    profile = nba.structure.Profiles(pos, r_edges)
    r_profile, dens_profile = profile.density(mass=mass)
    return r_profile, dens_profile

def main(
    SIM_ID, 
    component, 
    ncoefs, 
    fit_type='median', 
    compute_mise=False, 
    output_dir='test_108'):
    """
    Build the spherical basis for the specified simulation/component.
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
    suite = "Sheng24"
    if suite == "Sheng24":
        SNAPSHOT_PATH = "/n/nyx3/garavito/XMC-Atlas-sims/Sheng/Model_{}".format(SIM_ID)
        softening = 0.6 # TODO:implement this in plots!!
        SNAPNAME = "snapshot"
        SIM_PARAMS_PATH = '/n/nyx3/garavito/projects/XMC-Atlas/suites/Sheng24/orbits'
        SIM_PARAMS_FILE = 'MW_LMC_orbits_iso.txt'
    
    else:
        softening = None
        raise ValueError(f"Suite {suite} not implemented")

    nsnap = 0  # ??

    outpath = "/n/nyx3/garavito/projects/XMC-Atlas/scripts/{}/".format(output_dir)
    
    figure_name = '{}_{:04d}_density_profile_evolution.png'.format(component, SIM_ID)
    particle_profiles_filename = "{}_{:04d}_density_profiles_sheng24.h5".format(component, SIM_ID)
    bfe_profiles_filename = "bfe_{}_{:04d}_density_profiles_sheng24.h5".format(component, SIM_ID)
    modelname = 'modelname_{}_{:04d}.txt'.format(component, SIM_ID)
    cachename = 'cache_{}_{:04d}.txt'.format(component, SIM_ID)
    basis_filenames = "basis_{}_{:04d}.yaml".format(component, SIM_ID)
    coefs_filename = '{}_{:04d}_coefficients.h5'.format(component, SIM_ID)

    
    #--------------------------
    # profile fitting params
    #--------------------------
    
    rmin = 0.1 # should be 4*softening
    nbins = 101 # <- Should this be fixed? 

    # Are these the best params vvv
    if component == 'bulge':
        rmax = 40
    elif component == 'lmc':
        rmax = 300
    elif component == 'halo':
        rmax = 500
    
    #--------------------------
    # basis params:
    #--------------------------
    
    if component == 'bulge':
        nbins_basis = 400
    elif component == 'lmc':
        nbins_basis = 600
    elif component == 'halo':
        nbins_basis = 1000
    
    lmax = int(1)
    nmax = int(10)
    rmapping = 1.0
    
    #--------------------------
    # Pipeline params:
    #--------------------------
    paranoid = True
    cwd_path = os.getcwd()
    #--------------------------
    # log ouput.
    # 

    #------------------------------------------

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
    
    if component == "halo":
        comp = "MWhalo"
        exp_center = mw_center
    elif component == "lmc":
        comp = "LMChalo"
        exp_center = lmc_center
    elif component == "bulge":
        comp = "MWbulge"
        exp_center = mw_center
    
    print("-> Done loading simulation centers")

    snap_suffixes = get_snapshot_suffixes(SNAPSHOT_PATH, prefix=SNAPNAME+"_")
    NSNAPS = len(snap_suffixes)
    
    if paranoid == True:
        snap_check, missing_snaps = check_monotonic_contiguous_snapshots(snap_suffixes)
        print(f"[paranoid] {NSNAPS} snapshots found in {SNAPSHOT_PATH}")
        print(f"[paranoid] Snapshots are monotonic:", snap_check)
        if snap_check == False:
            print(f"[error] snapshots {missing_snaps} are missing")
        assert NSNAPS == len(tsim), "number of snapshots found differ from size of centers"
    
    #------------------------------------------
    
    #----------------------------------------
    # 2. Load particle data and recenter halo
    # ----------------------------------------

    rho_part_all = np.zeros((NSNAPS, nbins-1))
    r_bins_model = np.linspace(rmin, rmax, nbins)
    
    for i in range(NSNAPS):
        snapshot = SNAPNAME + "{:03d}.hdf5".format(i)
        data = load_particle_data(SNAPSHOT_PATH, SNAPNAME, [comp], nsnap=i, suite=suite)
        pos = data[comp]['pos']
        mass = data[comp]['mass']
        pos_center = pos - exp_center[i]
        
        #--------------------------
        # 3. Compute density profile 
        # -------------------------
        r_bins_part, rho_part_all[i] = make_density_profile(pos_center, mass, r_bins_model)

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

    
    #------------------------------------------
    #--------------------------
    # 4. Fit density profile 
    # -------------------------
    
    # TODO: do the fit for the mean
    if fit_type  == 'mean':
        rho_to_fit = np.mean(rho_part_all)
    if fit_type == 'median':
        rho_to_fit = np.median(rho_part_all)
    elif fit_type == 'initial':
        rho_to_fit = rho_part_all[0]
    elif fit_type == 'final':
        rho_to_fit = rho_part_all[-1]
    else:
        raise ValueError

    assert len(rho_to_fit) == len(r_bins_part)

    rho_fit, fit_params = fit_profile(r_bins_part, rho_to_fit)
    

    if paranoid == True:
        _ = plot_profiles(
            r_bins_part, 
            rho_part_all, 
            time=tsim,
            r_fit = r_bins_part,
            rho_fit = rho_fit,
            title=f"{component} {SIM_ID} density profile", 
            filename=outpath+figure_name)
   
    
    #------------------------------------------
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
        model_output = outpath,
        basis_params= basis_config,
        basis_filename = outpath+basis_filenames)

    # move to outopath folder
    os.chdir(outpath)
    basis = pyEXP.basis.Basis.factory(bconfig)
    # move back to original path 
    os.chdir(cwd_path)
    #------------------------------------------
    #--------------------------
    # 6. Compute coefficients
    # -------------------------
    # TODO: why if this import is earlier in the script
    # the cache is not build in step 5?

    from exp_coefficients import compute_exp_coefs_parallel

    # TODO check if we need this?
    compname = comp
    runtag   = 'run1'
    time     = 0.0

    basis.enableCoefCovariance(pcavar=True, nsamples=100, covar=True)
    basis.writeCoefCovariance(compname, runtag, time)

    gadget_particle_mass = 1e10
    units = [('mass', 'Msun', gadget_particle_mass),
             ('length', 'kpc', 1.0),
             ('velocity', 'km/s', 1.0),
             ('G', 'mixed', 43007.1)]


    for i in range(0, NSNAPS, ncoefs):
        snapshot = SNAPNAME + "{:03d}.hdf5".format(i)
        data = load_particle_data(SNAPSHOT_PATH, SNAPNAME, [comp], nsnap=i, suite=suite)
        pos = data[comp]['pos']
        mass = data[comp]['mass']
        pos_center = pos - exp_center[i]
        

        compute_exp_coefs_parallel(
            data[comp],
            basis,
            component,
            outpath+coefs_filename,
            unit_system=units)
    
    #Read coefficients
    coefs = pyEXP.coefs.Coefs.factory(outpath+coefs_filename)
    coefs_times = coefs.Times()
   

    #------------------------------------------
    #--------------------------------
    # 7. Compute BFE density profile
    # ------------------------------
    
    # TODO: this can be sumarized in a function

    rho_bfe_t = np.zeros((NSNAPS, len(r_bins_part)))
    j=0
    for i in range(0, NSNAPS, ncoefs):
        rho_bfe_t[i] = bfe_density_profiles(
            basis, 
            coefs, 
            r_bins=r_bins_part, 
            time=coefs_times[j])
        j+=1
    
    write_density_profiles(
        suite_id=SIM_ID, 
        snaps=np.arange(0, NSNAPS, 1), 
        rbins=r_bins_part, 
        profiles=rho_bfe_t,
        filename=outpath+bfe_profiles_filename)

    if compute_mise == True:
        all_mise_r = np.zeros_like(rho_bfe_t)
        #mise_r(rho_bfe_t[i], rho_part_all[i])
        #mise_r(np.log10(rho_bfe_t[i]), np.log10(rho_part_all[i]))
        # write(mise!)
        # plot mise
    
    # plot here


    #------------------------------------------
    #--------------------------
    # 8. Compute MISE
    # -------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build spherical basis functions for extrnal simulations."
    )

    # Positional arguments
    parser.add_argument(
        "sim_id",
        type=int,
        nargs="?",
        default=108,
        help="Simulation ID (default: 108)",
    )
    parser.add_argument(
        "component",
        type=str,
        nargs="?",
        default="halo",
        choices=["halo", "lmc", "bulge"],
        help="Galaxy component to process (default: halo)",
    )
    
	# Optional arguments
    parser.add_argument(
        "--ncoefs",
        type=int,
        nargs="?",
        default=10,
        help="Number of coefficient samples (default: 10)",
    )

    # Optional arguments
    parser.add_argument(
        "--fit-type",
        type=str,
        default="median",
        choices=["median", "mean", "initial", "final"],
        help="Method for fitting coefficients (default: median)",
    )
    parser.add_argument(
        "--compute-mise",
        action="store_true",
        help="Compute the MISE (default: False)",
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
        SIM_ID=args.sim_id,
        component=args.component,
        ncoefs=args.ncoefs,
        fit_type=args.fit_type,
        compute_mise=args.compute_mise,
        output_dir=args.output_dir,
    )

    
