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
from mpi4py import MPI
# BFE local libraries
THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR / "exp_pipeline"))

from ios_nbody_sims import load_particle_data
from plot_helpers import plot_profiles
from fit_density import fit_profile, fit_density_profile
from basis_utils import make_basis, load_basis
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
    output_dir='exp_expansions/coefficients',
    paranoid=True):
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
    
    basis_path = "/n/nyx3/garavito/projects/XMC-Atlas/scripts/exp_expansions/basis/"
    coefs_filename = '{}_{:04d}_coefficients.h5'.format(component, SIM_ID)
    
    
    if world_rank == 0:


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
    #--------------------------
    # 1. Load basis  
    # -------------------------

    # Load basis
    os.chdir(basis_path)
    config_name = f"basis_halo_{SIM_ID:04d}.yaml"
    print(f"Loading basis from {config_name}...")
    basis = load_basis(config_name)
    print(f"  Basis loaded")
	 

    from exp_coefficients import compute_exp_coefs_parallel

    # TODO check if we need this?
    #compname = "MWhalo"
    #runtag   = 'run1'
    #time     = 0.0

    #basis.enableCoefCovariance(pcavar=True, nsamples=100, covar=True)
    #basis.writeCoefCovariance(compname, runtag, time)

    gadget_particle_mass = 1e10
    units = [('mass', 'Msun', gadget_particle_mass),
             ('length', 'kpc', 1.0),
             ('velocity', 'km/s', 1.0),
             ('G', 'mixed', 43007.1)]


    for i in range(0, NSNAPS, ncoefs):
        print("computing coefficients in snap {}".format(i))
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
        print("Done computing coefficients in snap {}".format(i))
    
    #Read coefficients
    coefs = pyEXP.coefs.Coefs.factory(outpath+coefs_filename)
    coefs_times = coefs.Times()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build spherical basis functions for extrnal simulations."
    )

    # Positional arguments
    parser.add_argument(
        "sim_id",
        type=int,
        nargs="?",
        default=100,
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
        "--coefs_freq",
        type=int,
        nargs="?",
        default=10,
        help="Number of coefficient samples (default: 10)",
    )
	
    # Optional arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        nargs="?",
        default="exp_expansions/coefficients/",
        
    )


    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    main(
        SIM_ID=args.sim_id,
        component=args.component,
        ncoefs=args.coefs_freq,
        output_dir=args.output_dir,
    )

    
