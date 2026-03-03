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
    

def load_snapshot(snapname, snapshot_dir, outpath, npart=None):
    #GC21_bulge = nba.ios.ReadGC21(snapshot_dir, snapname)
    #bulge_data = GC21_bulge.read_halo(['pos', 'vel', 'mass', 'pid', 'pot'], halo='MW', ptype='bulge')

    GC21_halo = nba.ios.ReadGC21(snapshot_dir, snapname)

    if npart:
        halo_data = GC21_halo.read_halo(['pos', 'vel', 'mass', 'pid', 'pot'], halo='MW', ptype='dm', randomsample=npart)
    else:   
        halo_data = GC21_halo.read_halo(['pos', 'vel', 'mass', 'pid', 'pot'], halo='MW', ptype='dm')

    print("*Done loading {snapname}")

    # Load COM 
    tag = re.sub(r"_\d+\.hdf5$", "", snapname)
    nsnap = int(re.search(r'_(\d+)\.hdf5$', snapname).group(1))
    
    # Identify LMC model
    match = re.match(r"^([A-Za-z0-9]+)_", snapname)
    sim = match.group(1)

    #bulge_com = np.loadtxt(f"{outpath}/{sim}/{tag}_nba_bulge_pot.txt")[nsnap,0:3]
    halo_com = np.loadtxt(f"{outpath}/{sim}/{tag}_nba_halo_pot.txt")[nsnap,0:3]

    # Compute density profile
    mwhalo_rcom = nba.com.CenterHalo(halo_data)
    mwhalo_rcom.recenter(halo_com, np.array([0,0,0]))    

    #mwbulge_rcom = nba.com.CenterHalo(bulge_data)
    #mwbulge_rcom.recenter(bulge_com, np.array([0,0,0]))    
 

    return halo_com, halo_data['mass'], nsnap

def compute_coefs(basis, component, coefs_file, snapshot_dir, snapname, outpath, npart, dt, runtime_log):
    pos, mass, nsnap  = load_snapshot(snapname, snapshot_dir, outpath, npart)
    sim_time = nsnap * dt # Gyrs
    
    
    # Compute coefficients
    start_time = time.time()
    coef = basis.createFromArray(mass, pos.T, sim_time)
    coefs = pyEXP.coefs.Coefs.makecoefs(coef, name=component)
    coefs.add(coef)

    if os.path.exists(coefs_file):
        coefs.ExtendH5Coefs(coefs_file)
    else:
        coefs.WriteH5Coefs(coefs_file)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("*Done computing coefficients}")

    with open(runtime_log, "a") as f:
        f.write(f"Snapshot {nsnap}: {elapsed_time:.2f} s\n")


def main(config_file):
    cfg = load_config(config_file)
    snapname = cfg["snapname"]
    snapshot_dir = cfg["snapshot_dir"]
    outpath = cfg["outpath"]
    halo_basis_yaml = cfg["halo_basis_yaml"]
    coefs_file = cfg["coefs_file"]
    dt = cfg.get("dt", 0.02)
    component = cfg.get("component", "dm")
    runtime_log = cfg.get("runtime_log", "runtime_log.txt")
    npart = cfg.get("npart", None)

    init_snap = cfg["init_snap"]    
    final_snap = cfg["final_snap"]    

    if not os.path.exists(halo_basis_yaml):
        raise FileNotFoundError(f"Basis file not found: {halo_basis_yaml}")

    # Load basis
    basis = load_basis(halo_basis_yaml) 
    
    # Compute coefficients
    for n in range(init_snap, final_snap+1, 50):
        compute_coefs(basis, component, coefs_file, snapshot_dir, snapname+"_{:03d}.hdf5".format(n), outpath, npart, dt, runtime_log)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compute_exp_coefficients.py CONFIG_FILE.yaml")
        sys.exit(1)
    config = sys.argv[1]
    main(config)
