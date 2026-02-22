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
sys.path.append(str(THIS_DIR / "agama_pipeline"))

from ios_nbody_sims import load_particle_data
from plot_helpers import plot_profiles
from fit_density import fit_profile, fit_density_profile
from basis_utils import make_basis
from basis_fidelity import bfe_density_profiles, mise_r 
from data_products import write_density_profiles
from sanity_checks import check_monotonic_profiles, check_monotonic_contiguous_snapshots
from compute_bfe_helpers import load_sheng24_exp_center, get_snapshot_suffixes
from agama_BFEs import fitAgamaBFE, write_snapshot_coefs_to_h5


def main(
    SIM_ID, 
    ncoefs, 
    output_dir):
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
        SNAPNAME = "snapshot"
        SIM_PARAMS_PATH = '/n/nyx3/garavito/projects/XMC-Atlas/suites/Sheng24/orbits'
        SIM_PARAMS_FILE = 'MW_LMC_orbits_iso.txt'
    
    else:
        softening = None
        raise ValueError(f"Suite {suite} not implemented")

    nsnap = 0  # ??

    outpath = "/n/nyx3/garavito/projects/XMC-Atlas/scripts/{}/".format(output_dir)
    
    #figure_name = '{}_{:04d}_density_profile_evolution.png'.format(component, SIM_ID)
    

    
    #--------------------------
    # profile fitting params
    #--------------------------
    
    rmin = 0.1 # should be 4*softening
    nbins = 101 # <- Should this be fixed? 

    
    #  
    lmax = int(12)
    nmax = int(20)
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
   
    centers = {
        "mw_center": mw_center,
        "lmc_center": lmc_center
        }

    
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

    
    gadget_particle_mass = 1e10
    units = [('mass', 'Msun', gadget_particle_mass),
             ('length', 'kpc', 1.0),
             ('velocity', 'km/s', 1.0),
             ('G', 'mixed', 43007.1)]
   
    components = ['MWhalo', 'LMChalo', 'MWdisk', 'MWbulge']


    for i in range(NSNAPS):
        snapshot = SNAPNAME + "{:03d}.hdf5".format(i)
        data = load_particle_data(SNAPSHOT_PATH, SNAPNAME, components, nsnap=i, suite=suite)
        fitAgamaBFE(
            part=data,
            center_coords= centers,
            nsnap = i,
            OUTPUT_PATH=outpath)
    

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
    
	# Optional arguments
    parser.add_argument(
        "--coefs_freq",
        type=int,
        nargs="?",
        default=10,
        help="Number of coefficient samples (default: 10)",
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
        ncoefs=args.coefs_freq,
        output_dir=args.output_dir,
    )

    


    write_snapshot_coefs_to_h5(
    snapshot_ids = range(0, 101),
    coef_file_patterns=[
    args.output_dir+"/{snap:03d}.MW.none_6.coef_mult",
    args.output_dir+"/{snap:03d}.MW.none_6.coef_cylsp",
    args.output_dir+"/{snap:03d}.LMC.none_6.coef_mult",],
    h5_output_paths=[
        args.output_dir+"/MW.none_6.coef_mult.h5",
        args.output_dir+"/MW.none_6.coef_mult.h5",
        args.output_dir+"/LMC.none_6.coef_mult.h5",]
    )
