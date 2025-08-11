"""
Functionality to compute BFE coefficients using EXP.
Adapted from :https://github.com/EXP-code/pyEXP-examples/blob/main/external-simulations/IntegrateInSgrSnapshot.py

"""

import numpy as np
import pyEXP
from EXP_utils import *


def compute_coefs(halo_config, m, pos, time, coef_file, compname, add_coef = False):
    # Build basis from config file
    print("-> Constructing basis \n")
    halo_basis = pyEXP.basis.Basis.factory(halo_config)

    print("-> Creating coefficients from particles masses and positions \n")
    # if you want to use the array creator, do this:
    halo_coef = halo_basis.createFromArray(m, pos, time=time)
    
    print("-> Computing coefficients \n")
    # make a makecoefs instance
    # only do this at the first step: afterwards just add (see below).
    halo_coefs = pyEXP.coefs.Coefs.makecoefs(halo_coef, compname)

    # add the coefficients to the makecoefs instance
    halo_coefs.add(halo_coef)

    # write the first step (afterwards use ExtendH5Coefs)
    if add_coef == False:
        halo_coefs.WriteH5Coefs(coef_file)
    elif add_coef == True: 
        halo_coefs.ExtendH5Coefs(coef_file)

    return 0
