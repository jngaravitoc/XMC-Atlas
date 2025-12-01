# computes EXP coefficients
import os
import sys
import re
import time

import numpy as np
import yaml
import pynbody

import nba
import pyEXP

from ios_nbody_sims import LoadSim

def load_config(config_file):
    """Load YAML configuration parameters."""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def load_basis(conf_name):
    """
    Load a basis configuration from a YAML file and initialize a Basis object.

    Parameters
    ----------
    conf_name : str
        Path to the YAML configuration file. If the provided filename does not 
        end with `.yaml`, the extension is automatically appended.

    Returns
    -------
    basis : pyEXP.basis.Basis
        An initialized Basis object created from the configuration.

    Raises
    ------
    FileNotFoundError
        If the specified YAML file does not exist.
    """

    # Check file existence
    if not os.path.exists(conf_name):
        raise FileNotFoundError(f"Configuration file not found: {conf_name}")

    # Load YAML safely
    with open(conf_name, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Build basis from configuration
    basis = pyEXP.basis.Basis.factory(config)
    return basis
    

#load_halo(snapname, snapshot_dir, orbit_path, halo, npart=None):


def load_center(snapname, center_path):
    # Load COM 
    tag = re.sub(r"_\d+\.hdf5$", "", snapname)
    nsnap = int(re.search(r'_(\d+)\.hdf5$', snapname).group(1))
    
    # Identify LMC model
    match = re.match(r"^([A-Za-z0-9]+)_", snapname)
    sim = match.group(1)

    #bulge_com = np.loadtxt(f"{outpath}/{sim}/{tag}_nba_bulge_pot.txt")[nsnap,0:3]
    halo_com_pos = np.loadtxt(center_path)[nsnap,0:3]
    halo_com_vel = np.loadtxt(center_path)[nsnap,3:6]

    return halo_com_pos, halo_com_vel, nsnap

    """
    # Compute density profile
    mwhalo_rcom = nba.com.CenterHalo(halo_data)
    mwhalo_rcom.recenter(halo_com, np.array([0,0,0]))    

    #mwbulge_rcom = nba.com.CenterHalo(bulge_data)
    #mwbulge_rcom.recenter(bulge_com, np.array([0,0,0]))    
 
    return halo_com, halo_data['mass'], nsnap
    """

def recentering(particle_data, snapname, center):
    center = load_center(snapname, center)
    mwhalo_recenter = nba.com.CenterHalo(particle_data)
    mwhalo_recenter.recenter(halo_data, center)    
    return particle_data

def load_particles_data(snapshot_dir, snapname, expansion_center, dt, component): 

    load_data = LoadSim(snapshot_dir, snapname, expansion_center)
    # Load center
    center = load_center(snapname, orbit)

    if component=='MWHaloiso':
        particle_data = load_data.load_halo('MWnoLMC')
    
    elif component=='MWHalo':
        particle_data = load_data.load_halo('MW')
    
    elif component=='LMChalo':
        particle_data = load_data.load_halo('LMC')
     
    elif component=='MWdisk':
        particle_data = load_data.load_mw_disk(**kwargs)
    
    elif component=='MWbulge':
        particle_data = load_data.load_mw_bulge(**kwargs)
    
    
    particle_data = recentering(particle_data, snapname, centers)

    snap_times = nsnap * dt # Gyrs # TODO get this from header! 

    return particle_data, snap_times

def compute_exp_coefs(halo_data, basis, component, coefs_file, runtime_log):
    #pos, mass, nsnap  = load_mwhalo(snapname, snapshot_dir, npart)
    #pos, mass, nsnap  = load_snapshot(snapname, snapshot_dir, outpath, npart)
 
    
    # Compute coefficients
    start_time = time.time()
    coef = basis.createFromArray(halo_data['mass'], halo_data['pos'], sim_time)
    coefs = pyEXP.coefs.Coefs.makecoefs(coef, name=component)
    coefs.add(coef)

    if os.path.exists(coefs_file):
        coefs.ExtendH5Coefs(coefs_file)
    else:
        coefs.WriteH5Coefs(coefs_file)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("*Done computing coefficients")

    with open(runtime_log, "a") as f:
        f.write(f"Snapshot {nsnap}: {elapsed_time:.2f} s\n")



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


	

def main(config_file, expansion_type, suite):
    #TODO: check that there are all these snapshots in the folder before starting BFE
    #computation.

    cfg = load_config(config_file)
    snapname = cfg["snapname"]
    snapshot_dir = cfg["snapshot_dir"]
    orbit = cfg["orbit"]
    dt = cfg.get("dt", 0.02) # TODO: see if this can obtain from snap header?
    component = cfg.get("component", "dm")
    runtime_log = cfg.get("runtime_log", "runtime_log.txt")
    npart = cfg.get("npart", None)

    init_snap = cfg["init_snap"]    
    final_snap = cfg["final_snap"]    


    if expansion_type=='EXP':
        os.chdir('../../GC21/basis/')
        halo_basis_yaml = cfg["halo_basis_yaml"]
        coefs_file = cfg["coefs_file"]
        
        if not os.path.exists(halo_basis_yaml):
            raise FileNotFoundError(f"Basis file not found: {halo_basis_yaml}")
    
        
        # Load basis
        basis = load_basis(halo_basis_yaml) 
        
        # Compute coefficients
        for n in range(init_snap, final_snap+1):
            compute_exp_coefs(basis, component, coefs_file, snapshot_dir,
                              snapname+"_{:03d}.hdf5".format(n), 
                              orbit, npart, dt, runtime_log, suite)

    elif expansion_type=='AGAMA':
        for n in range(init_snap, final_snap+1):
            compute_agama_coefs(snapshot_dir, snapname+"_{:03d}.hdf5".format(n), 
                                orbit, npart, dt, runtime_log)

        

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compute_exp_coefficients.py CONFIG_FILE.yaml")
        sys.exit(1)
    config = sys.argv[1]
    expansion_type = sys.argv[2]
    
    suites = ['GC21']
    
    for suite in suites:
        main(config, expansion_type, suite)
