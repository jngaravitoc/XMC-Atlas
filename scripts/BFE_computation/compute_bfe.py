# computes EXP coefficients for the XMC-Atlas

import os
import sys
import re
import datetime
import time
import logging
import numpy as np
import yaml
import pynbody

import nba
import pyEXP

# Local libraries
from ios_nbody_sims import LoadSim, check_snaps_in_folder
from compute_bfe_helpers import (
    read_simulations_files,
    load_exp_basis, 
    load_config_file,
    setup_logger,
)

def check_coefficients_path(outpath):
    if os.path.isdir(outpath):
        logging.info(f"> Coefficients {outpath} folder exists")
    else:
        logging.info(f"> Creating coefficients folder in: {outpath}")
        os.makedirs(outpath, exist_ok=True)

def sample_snapshots(initial_snap, final_snap, nsnaps_to_compute_exp):
    snaps_to_compute_exp = np.arange(initial_snap, final_snap+1, 1, dtype=int)
    nsnaps = len(snaps_to_compute_exp)

    assert snaps_to_compute_exp[0] == initial_snap
    assert snaps_to_compute_exp[-1] == final_snap
    if nsnaps_to_compute_exp:
        nsample = round(nsnaps / nsnaps_to_compute_exp)
        snaps_to_compute_exp = snaps_to_compute_exp[::nsample]
    
    nsnaps_sample = len(snaps_to_compute_exp)
    logging.info("Computing coefficients in {} snapshots".format(nsnaps_sample))
    return snaps_to_compute_exp

def load_GC21_exp_center(origin_dir, nsnap, component, suite, return_vel=False ):
    """
    Loads the center of the GC21 simulations

    Paramters:
    ----------
    centers_parh : str
        filename with the centers.

    Returns:
    --------
    
    halo_com_pos : np.ndarray, shape (3,N)
    halo_com_vel : np.ndarray, shape (3,N) (optional)

    TODO: This could be skipped by loading once the snapshots and caching the orbit to avoid
    reading at every snapshot.
    
    """
    
    #tag = re.sub(r"_\d+\.hdf5$", "", str(snapname))
    #print(tag, snapname)
    #nsnap = int(re.search(r'_(\d+)\.hdf5$', snapname).group(1))
    
    # Identify LMC model
    #match = re.match(r"^([A-Za-z0-9]+)_", snapname)
    #sim = match.group(1)

    center_file = read_simulations_files(simulation_files, suite, component, quantity='expansion_center')
    # TODO this should be in the params file
    origin_file = os.path.join(origin_dir, center_file)
    if not os.path.isfile(origin_file):
        raise FileNotFoundError(f"> Origins file not found in {origin_file}")

    density_center = np.loadtxt(origin_file)[nsnap,0:3]
    
    if return_vel == True:
        velocity_center = np.loadtxt(center_file)[nsnap,3:6]
        return density_center, velocity_center
    else:
        return density_center


def recentering(origin_dir, particle_data, nsnap, component, suite):
    """
    Recenters particle data

    Parameters:
    -----------
    particle data : np.ndarray, shape (N, 3)
    snapname : str
        string with the snapshot name
    density_center : np.ndarray, shape (3)

    TODO: adds velocity center if needed!
    """
    # Load expansion centers
    expansion_center = load_GC21_exp_center(origin_dir, nsnap, component, suite)
    particle_data['pos'] = particle_data['pos'] -  expansion_center 
    return particle_data

# TODO move this function to bfe_computation_helper.py
def load_particle_data(origin_dir, component, suite, nsnap, **kwargs):
    """
    Load particle data

    Returns:
    --------

    Particle data:

    snap_time:
    """
    full_snapname = snapname + "_{:03d}.hdf5".format(nsnap)
    load_data = LoadSim(snapshot_dir, full_snapname)
    # Load center
    print("--------------------------------")
    if component=='MWHaloiso':
        particle_data = load_data.load_halo('MWnoLMC', **kwargs)
    
    elif component=='MWHalo':
        particle_data = load_data.load_halo('MW', **kwargs)
    
    elif component=='LMChalo':
        particle_data = load_data.load_halo('LMC', **kwargs)
     
    elif component=='MWdisk':
        particle_data = load_data.load_mw_disk(**kwargs)
    
    elif component=='MWbulge':
        particle_data = load_data.load_mw_bulge(**kwargs)

    logging.info("Done loading snapshot")
    
    # TODO check: are we passing here nsnap too?
    particle_data = recentering(origin_dir, particle_data, nsnap, component, suite)
    logging.info("Done re-centering data")
    # TODO does this need to be done here?
    snap_times = load_data.load_snap_time() 
    logging.info("Done loading snap-time data")

    return particle_data, snap_times

def compute_exp_coefs(halo_data, snap_time, basis, component, coefs_file, unit_system, runtime_log, **kwargs):
    # Compute coefficients
    # TODO define units
    start_time = time.time()
    coef = basis.createFromArray(halo_data['mass'], halo_data['pos'], snap_time)
    coefs = pyEXP.coefs.Coefs.makecoefs(coef, name=component)
    coefs.add(coef)
    
    #TODO: move unit systems to another function or a file
    if unit_system == 'Gadget':
        coefs.setUnits([ 
        ('mass', 'Msun', 1e10), 
        ('length', 'kpc', 1.0),
        ('velocity', 'km/s', 1.0), 
        ('G', 'mixed', 43007.1) 
        ])

    if os.path.exists(coefs_file):
        coefs.ExtendH5Coefs(coefs_file)
    else:
        coefs.WriteH5Coefs(coefs_file)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("*Done computing coefficients")
    
    with open(runtime_log, "a") as f:
        nparticles = len(halo_data['mass'])
        f.write(f"Coefficients for snapshot t={snap_time}\
                  and nparticles={nparticles} computed \
                  in: {elapsed_time:.2f} s\n")


def compute_agama_coefs(snapshot_dir, snapname, expansion_center, npart, dt, runtime_log):
    import agama
    from agama_external_sims import create_GizmoLike_snapshot, fit_potential
    #pos, mass, nsnap  = load_mwhalo(snapname, snapshot_dir, npart)
    #pos, mass, nsnap  = load_snapshot(snapname, snapshot_dir, outpath, npart)
 
    halo_data = LoadSim(snapshot_dir, snapname, expansion_center, suite)
    mw_halo_particles = halo_data.load_halo(halo='MW', quantities=['pos', 'vel', 'mass'], npart=npart)
    # Load center
    pos_center, vel_center, nsnap = load_center(snapname, expansion_center)
    mwhalo_recenter = nba.com.CenterHalo(mw_halo_particles)
    mwhalo_recenter.recenter(pos_center, [0,0,0])    

    mw_disk_particles = halo_data.load_mw_disk(quantities=['pos', 'vel', 'mass'])
    mw_disk_recenter = nba.com.CenterHalo(mw_disk_particles)
    mw_disk_recenter.recenter(pos_center, [0,0,0])    


    sim_time = nsnap * dt # Gyrs # TODO get this from header! 
    print(sim_time)
    # re-center
    
    # Compute coefficients
    start_time = time.time()
    print(mw_halo_particles.keys())    

    # build snapshot WITHOUT gas to demonstrate robustness when gas absent
    
    snapshot = create_GizmoLike_snapshot(
        pos_dark=mw_halo_particles['pos'],
        mass_dark=mw_halo_particles['mass'],
        pos_star=mw_disk_particles['pos'],
        mass_star=mw_disk_particles['mass'],
        #pos_gas=np.ones((100, 3)),
        #mass_gas=np.ones(100),
        # temperature_gas=temperature_gas,
    )

    print("Running fit_potential on synthetic snapshot...")
    t0 = time.perf_counter()
    outputs = fit_potential(
        snapshot,
        nsnap=nsnap,
        sym=["n"],
        pole_l=[4],
        rmax_sel=600.0,
        rmax_exp=500.0,
        save_dir="./demo_output",
        file_ext="spline",
        verbose=True,
        halo='MW_iso_beta1',
    )
    dt = time.perf_counter() - t0

    print("\nBenchmark complete.")
    print(f"Elapsed time: {dt:.3f} s")
    print("Generated files summary:")
    for key, files in outputs.items():
        print(f"  {key}: {len(files)} files")
        for f in files[:3]:
            print(f"    - {f}")
        if len(files) > 3:
            print(f"    ... (+{len(files)-3} more)")
    print("Done.")



    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("*Done computing Agama coefficients")

def main(config_file, suite): 
    #TODO: check that there are all these snapshots in the folder before starting BFE
    #computation.
    global snapname
    global snapshot_dir
    global simulation_files
    global output_dir
    
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
    
    log_name = f"{output_dir}exp_expansion_{snapname}.log"
    print(f"> Log file created in: {log_name}")
    setup_logger(log_name)
    logging.info("Expansion created on:")
    logging.info(str(datetime.datetime.now()))
    logging.info("Expansion run with the following parameters:")
    logging.info(cfg)
    

    # Sample the snapshots in which the coefficients are going to be computed
    snaps_to_compute_exp = sample_snapshots(initial_snap, final_snap, nsnaps_to_compute_exp)
    nsnaps = len(snaps_to_compute_exp)

    # TODO check that all the snaps_to_compute_exp snapshots are in the folder
    all_snapnames = []
    for n in range(nsnaps):
        all_snapnames.append(snapname+ "_{:03d}.hdf5".format(snaps_to_compute_exp[n]))
    check_snaps_in_folder(snapshot_dir, all_snapnames)

    logging.info(f"> All snapshots found in {snapshot_dir}")
   
    # EXP 
    if expansion_type=='EXP':
        # Move to directory conteining basis files
        os.chdir(basis_paths)
        # Load basis
        # Basis_file_name_path
        basis = load_exp_basis(simulation_files, basis_paths, component, suite, compute_variance) 
        logging.info("-> Done loading basis")
        # Compute coefficients

        
        for snap in snaps_to_compute_exp:
            particle_data, snap_time = load_particle_data(   
                origin_dir,
                component, 
                suite,
                nsnap=snap,
                npart=npart,
                )
            logging.info("Done loading particle data")

            sys.exit()
            # Define unit system
            compute_exp_coefs(
                particle_data, 
                snap_time,
                basis, 
                component, 
                coefs_file,
                unit_system,
                runtime_log)

            logging.info("Done computing coefficients")


    # AGAMA
    elif expansion_type=='AGAMA':
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
