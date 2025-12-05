#!/usr/bin/env python3
"""
Build a basis expansion models
"""

import os
import sys
import numpy as np
from mpi4py import MPI
import gala.potential as gp
import pyEXP
import yaml
#import EXPtools


# -------------------
# User parameters
# -------------------
MODEL_NAME = "GC21_MW_DM_halo.txt"
POTENTIAL_NAME = "../../GC21/GC21LMC1.potential"
NMAX = 10
LMAX = 8
N_RBINS = 400
BASIS_NAME = f"GC21_MW_DM_halo_{NMAX}_{LMAX}.yaml"
DISK_BASIS_NAME = f"GC21_MW_DM_halo_{NMAX}_{LMAX}.yaml"
CACHE_NAME = f".cache_GC21_MW_DM_halo_{NMAX}_{LMAX}"
CONFIG_NAME = f"config_GC21_MW_DM_halo_{NMAX}_{LMAX}.yaml"
OUTPATH = "../../GC21/basis/"

# copy-paste from EXPtools
def write_basis(basis, basis_name):
    """
    Write a basis configuration dictionary to a YAML file.

    Parameters
    ----------
    basis : dict
        Dictionary containing the basis configuration.
    conf_name : str
        Name of the YAML file to write. If the provided name does not
        end with `.yaml`, the extension is automatically appended.

    Returns
    -------
    str
        The final filename used to save the YAML configuration.
    """
    # Ensure file has .yaml extension
    if not basis_name.endswith(".yaml"):
        basis_name += ".yaml"

    # Write to file
    with open(basis_name, "w") as file:
        yaml.dump(basis, file, default_flow_style=False, sort_keys=False)

    return 0



# copy-paste from EXPtools
def make_config(basis_id, float_fmt_rmin="{:.7f}", float_fmt_rmax="{:.3f}",
                float_fmt_rmapping="{:.3f}", **kwargs):
    """
    Create a YAML configuration file string for building a basis model.

    Parameters
    ----------
    basis_id : str
        Identifier of the basis model. Must be either 'sphereSL' or 'cylinder'.
    float_fmt_rmin : str, optional
        Format string for rmin (default ``"{:.7f}"``).
    float_fmt_rmax : str, optional
        Format string for rmax (default ``"{:.3f}"``).
    float_fmt_rmapping : str, optional
        Format string for rmapping (default ``"{:.3f}"``).
    **kwargs : dict
        Additional keyword arguments required depending on the basis type:

        - For ``sphereSL``:
          ['lmax', 'nmax', 'rmapping', 'modelname', 'cachename']

        - For ``cylinder``:
          ['acyl', 'hcyl', 'nmaxfid', 'lmaxfid', 'mmax', 'nmax',
           'ncylodd', 'ncylnx', 'ncylny', 'rnum', 'pnum', 'tnum',
           'vflag', 'logr', 'cachename']

    Returns
    -------
    str
        YAML configuration file contents.

    Raises
    ------
    KeyError
        If mandatory parameters for the given basis are missing.
    FileNotFoundError
        If ``modelname`` is required but cannot be opened.
    ValueError
        If the model file does not contain valid radius data.
    """

    #check_basis_params(basis_id, **kwargs)

    if basis_id == "sphereSL":
        modelname = kwargs["modelname"]
        try:
            R = np.loadtxt(modelname, skiprows=3, usecols=0)
        except OSError as e:
            raise FileNotFoundError(f"Could not open model file '{modelname}'") from e
        if R.size == 0:
            raise ValueError(f"Model file '{modelname}' contains no radius data")

        rmin, rmax, numr = R[0], R[-1], len(R)

        config_dict = {
            "id": basis_id,
            "parameters": {
                "numr": int(numr),
                "rmin": rmin,
                "rmax": rmax,
                "Lmax": int(kwargs["lmax"]),
                "nmax": int(kwargs["nmax"]),
                "rmapping": float(kwargs["rmapping"]),
                "modelname": str(modelname),
                "cachename": str(kwargs["cachename"]),
            },
        }

    elif basis_id == "cylinder":
        config_dict = {
            "id": basis_id,
            "parameters": {
                "acyl": float(kwargs["acyl"]),
                "hcyl": float(kwargs["hcyl"]),
                "nmaxfid": int(kwargs["nmaxfid"]),
                "lmaxfid": int(kwargs["lmaxfid"]),
                "mmax": int(kwargs["mmax"]),
                "nmax": int(kwargs["nmax"]),
                "ncylodd": int(kwargs["ncylodd"]),
                "ncylnx": int(kwargs["ncylnx"]),
                "ncylny": int(kwargs["ncylny"]),
                "rnum": int(kwargs["rnum"]),
                "pnum": int(kwargs["pnum"]),
                "tnum": int(kwargs["tnum"]),
                "vflag": int(kwargs["vflag"]),
                "logr": bool(kwargs["logr"]),
                "cachename": str(kwargs["cachename"]),
            },
        }

    return yaml.dump(config_dict, sort_keys=False)

def compute_halo_basis():
    """Main driver for building the halo basis expansion."""
    
    if not os.path.exists(POTENTIAL_NAME):
        raise FileNotFoundError(f"{POTENTIAL_NAME} not found")

    # Load potential
    GC21_pot = gp.potential.load(POTENTIAL_NAME)
    GC21_MW_DMhalo = GC21_pot["halo"]

    # Halo parameters
    Mtot = GC21_MW_DMhalo.parameters["m"].value
    rbins = np.linspace(1e-3, 300, N_RBINS)

    # Halo profile
    halo_profile = GC21_MW_DMhalo.density(rbins)

    # TODO: define rcen properly
    rcen = 0.0  

    # Build model
    model = EXPtools.make_model(
        rcen,
        halo_profile,
        Mtotal=Mtot,
        output_filename=OUTPATH+MODEL_NAME,
        physical_units=True,
    )
    R = model["radius"]

    # Build basis
    config = EXPtools.make_config(
        basis_id="sphereSL",
        lmax=LMAX,
        nmax=NMAX,
        rmapping=R[-1],
        modelname=OUTPATH+MODEL_NAME,
        cachename=OUTPATH+CACHE_NAME,
    )

    basis = pyEXP.basis.Basis.factory(config)
    EXPtools.write_basis(config, OUTPATH+BASIS_NAME)

    print(f"Basis written to {OUTPATH+BASIS_NAME}")






if __name__ == "__main__":
    #compute_halo_basis()
	#compute_GC21_disk_basis()
	basis_cache="mwiso_disk.cache.basis"

	disk_conf = make_config('cylinder', acyl=3.5,hcyl=0.9,nmaxfid=64, lmaxfid=64, mmax=0,nmax=24,
		ncylodd=3,ncylnx=256,ncylny=128,rnum=200,pnum=0, tnum=0,vflag=0,logr="false",cachename=basis_cache)

	write_basis(disk_conf, 'mwdisk_GC21_basis.yaml')	
