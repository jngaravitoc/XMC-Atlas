import os
import sys
import datetime
import time
import logging
import yaml
import numpy as np
import pyEXP

# Local libraries
from ios_nbody_sims import load_particle_data
from exp_coefficients import compute_exp_coefs, compute_exp_coefs_parallel
from agama_coefficients import compute_agama_coefs
from compute_bfe_helpers import (
    check_coefficients_path,
    sample_snapshots,
    check_snaps_in_folder,
    load_exp_basis,
    load_GC21_exp_center,
)

def setup_logger(logfile="bfe_computation.log"):
    logging.basicConfig(
        filename=logfile,
        filemode="w",                     # overwrite each run; use "a" to append
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO                # or DEBUG for more detail
    ) 

def load_config_file(config_path):
    """
    Load and parse the configuration YAML file.

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    dict
        A dictionary containing all YAML entries, plus an additional key
        `expansion_type` which can be either "EXP" or "AGAMA" depending on
        which block (`exp` or `agama`) is non-null in the YAML file.

        If both `exp` and `agama` are provided, "EXP" takes precedence.
        If both are null, `expansion_type` will be None.

    Notes
    -----
    The YAML file is expected to contain the sections:
    - "paths"
    - "simulations"
    - "exp"
    - "agama"

    The function determines `expansion_type` as follows:
    - If the `exp` section is not null → "EXP"
    - Else if the `agama` section is not null → "AGAMA"
    - Else → None
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Determine expansion type
    exp_block = config.get("exp")
    agama_block = config.get("agama")

    if exp_block is not None:
        expansion_type = "EXP"
    elif agama_block is not None:
        expansion_type = "AGAMA"
    else:
        expansion_type = None

    config["expansion_type"] = expansion_type

    return config

def main(config_file, suite): 
    """
    Docstring for main
    
    :param config_file: Description
    :param suite: Description
    """
    global snapname
    global snapshot_dir
    global simulation_files
    global output_dir
    
    # TODO describe all the config parameters in a markdown file
    # Load parameters
    cfg = load_config_file(config_file)
    snapshot_dir = cfg["paths"]["snapshot_dir"]
    output_dir = cfg["paths"]["output_dir"]
    coefs_file = cfg["paths"]["coefficients_filename"]
    origin_dir = cfg["paths"]["origins_dir"]
    # Simulations parameters
    snapname = cfg["simulations"]["snapname"]
    component = cfg["simulations"]["component"]
    initial_snap = cfg["simulations"]["initial_snap"]    
    final_snap = cfg["simulations"]["final_snap"]        
    nsnaps_to_compute_exp = cfg["simulations"]["nsnaps_to_compute_exp"]
    npart = cfg["simulations"]["npart_per_snapshot"]
    simulation_files = cfg["simulations"]["simulation_files"] 

    expansion_type = cfg["expansion_type"]

    # Expansion type
    if cfg["expansion_type"] == "EXP":
        basis_paths = cfg["exp"]["basis_paths"]
        unit_system= cfg["exp"]["units_system"]
        compute_variance = cfg["exp"]["compute_bfe_variance"]
        os.path.isdir(basis_paths)
    
    elif cfg["expansion_type"] == "AGAMA":
        rmax_exp = cfg["agama"]["rmax_exp"]
        rmax_sel = cfg["agama"]["rmax_sel"]
        pole_l = cfg["agama"]["pole_l"]
        sym = cfg["agama"]["sym"]

    # Check that coefficients file exist
    os.path.isdir(snapshot_dir)
    check_coefficients_path(output_dir)
    
    # log outputs
    log_name = f"{output_dir}exp_expansion_{snapname}.log"
    print(f"> Log file created in: {log_name}")
    setup_logger(log_name)
    logging.info("Expansion created on:")
    logging.info(str(datetime.datetime.now()))
    logging.info("Expansion run with the following parameters:")
    logging.info(cfg)
    
    # Sample the snapshots in which the coefficients are going to be computed
    # snaps_to_compute_exp is an array with the snapshots numbers to use. 
    # e.g., it would be use as: snap_{snaps_to_compute_exp[0]}.hdf5
    snaps_to_compute_exp = sample_snapshots(initial_snap, final_snap, nsnaps_to_compute_exp)
    nsnaps = len(snaps_to_compute_exp)

    # Check that all the requested snaps_to_compute_exp snapshots are in the snapshot_dir folder
    all_snapnames = []
    for n in range(nsnaps):
        all_snapnames.append(snapname+ "_{:03d}.hdf5".format(snaps_to_compute_exp[n]))
    check_snaps_in_folder(snapshot_dir, all_snapnames)

    logging.info(f"> All snapshots found in {snapshot_dir}")
    
    # Load centers here
    centers = load_GC21_exp_center(origin_dir, simulation_files, suite, component, return_vel=False)
    assert centers.shape[1] == 3, "centers array dimension has to be (nsnaps, 3)"
    
    # Load basis only ones across snapshots
    if expansion_type=='EXP':
        # Move to directory conteining basis files
        os.chdir(basis_paths)
        # Load basis
        basis = load_exp_basis(simulation_files, basis_paths, component, suite, compute_variance) 
        logging.info("-> Done loading basis")


    for snap in snaps_to_compute_exp:
        particle_data, snap_time = load_particle_data( 
            snapshot_dir, 
            snapname, 
            component, 
            nsnap=snap,
            npart=npart,
            )
        logging.info("Done loading particle data")

        # Recentering
        particle_data['pos'] -= centers[snap]
        logging.info("Done re-centering data")
           
        if expansion_type=='EXP':
            # Define unit system
            compute_exp_coefs(
                particle_data, 
                snap_time,
                basis, 
                component, 
                coefs_file,
                unit_system)

            logging.info("Done computing coefficients")
            sys.exit()
    # AGAMA

        elif expansion_type=='AGAMA':
            raise NotImplementedError("AGAMA expansions are work in progress")
            for n in range(init_snap, final_snap+1):
                compute_agama_coefs(
                    snapshot_dir, 
                    snapname+"_{:03d}.hdf5".format(n), 
                    orbit, 
                    npart, 
                    dt, 
                    runtime_log)

        else: 
            raise NotImplementedError("Only AGAMA and EXP expansions are implemented")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compute_exp_coefficients.py CONFIG_FILE.yaml")
        config_example = "https://github.com/jngaravitoc/XMC-Atlas/tree/main/scripts/BFE_computation/config_files"
        print("A CONFIG_FILE example can be found at: {config_example}")
        sys.exit(1)
        
    config = sys.argv[1]
    suites = ['GC21']
    
    for suite in suites:
        main(config, suite)
