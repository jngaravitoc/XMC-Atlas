import os
import time
import numpy as np
import logging
from mpi4py import MPI
import pyEXP

def compute_exp_coefs(halo_data, snap_time, basis, component, coefs_file, unit_system, **kwargs):
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
    print("> Done computing coefficients*")
    
    nparticles = len(halo_data['mass'])
    logging.info(f"Coefficients for snapshot t={snap_time}\
                  and nparticles={nparticles} computed \
                  in: {elapsed_time:.2f} s\n")


