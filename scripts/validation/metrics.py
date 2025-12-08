"""
Functions provided by Hayden Foote UofA
"""

import nunpy as np
from scipy.spatial import KDTree
import pyEXP

# helper functions for error estimation
# plot MIRSE with Gadget with respect to a particular choice of expansion truncation


def density_points(pos, m, points, k_max=1000):
    '''
    Reproduction of snapAnalysis' density_points for an arbitrary selection of particles
    
    '''
    # construct a KD Tree for nearest-neighbor lookup
    tree = KDTree(pos)

    # query the tree for the distances to the k_max-th nearest neighbors
    dist, idx = tree.query(points, k=[k_max])
    m = k_max*m
    v = (4./3.*np.pi*(dist.flatten())**3.)
    
    return m/v

def compute_MISE(exp_field, target_field):
	MISE = np.mean((exp_density/target_density - 1.)**2)
	return MISE


def compute_exp_field(coefficients, basis, time, grid):
	# calculate density field on the provided grid
	fields = pyEXP.field.FieldGenerator([coefficients.Times()[time]], grid)
	surfaces = fields.points(basis, coefficients)

	# get the density in proper units
	field_grid = surfaces[coefficients.Times()[t]]['dens']

	return field_grid




def MIRSE_grid(basis, coefs, grid, t, target_density, nmax=18, lmax=10, verbose=False):
    '''
    Calculates the mean-sqaure-relative-error between an EXP reconstruction and the gadget density field
    for a grid of nmax and lmax expansion truncations

    Parameters
    ----------
    basis : pyEXP basis
        basis set
    
    coefs : pyEXP coefficients
        coefficient set

    grid : np.ndarray
        Nx3 array of points at which the density field is sampled

    t : int
        time index for the coefficient set
    target_density : np.ndarray
        Targe density Nx3 
    nmax : int, optional
        max radial order

    lmax : int, optional
        max harmonic order

    verbose : bool, optional
        if True, prints progress

    Returns
    -------
    np.ndarray
        nmax x lmax array of MIRSE values between the simulation density field and the EXP reconstruction
    '''

    # construct arrays to hold MSE values
    MIRSE = np.zeros([nmax+1, lmax+1])

    for n in range(nmax+1):
        for l in range(lmax+1):
            if verbose:
                print(f"calculating nmax = {n}, lmax = {l}")
            
            # truncate coefficients to desired length
            #truncated_coefs = EXPtools.utils.coefficients.truncate_expansion(coefs, nmax=n, lmax=l)

            # calculate density field on the provided grid
            fields = pyEXP.field.FieldGenerator([coefs.Times()[t]], grid)
            surfaces = fields.points(basis, truncated_coefs)

            # get the density in proper units
            dens = surfaces[coefs.Times()[t]]['dens']

            # calculate MIRSE
            MIRSE[n, l] = np.mean((dens/gad_density - 1.)**2)

    return MIRSE

def MISE_grid(basis, coefs, grid, t, gad_density, nmax=18, lmax=10, verbose=False):
    '''
    Calculates the mean-square-error between an EXP reconstruction and the gadget density field
    for a grid of nmax and lmax expansion truncations

    Parameters
    ----------
    basis : pyEXP basis
        basis set
    
    coefs : pyEXP coefficients
        coefficient set

    grid : np.ndarray
        Nx3 array of points at which the density field is sampled

    t : int
        time index for the coefficient set


    nmax : int, optional
        max radial order

    lmax : int, optional
        max harmonic order

    verbose : bool, optional
        if True, prints progress

    Returns
    -------
    np.ndarray
        nmax x lmax array of MISE values between the simulation density field and the EXP reconstruction
    '''

    # construct arrays to hold MSE values
    MISE = np.zeros([nmax+1, lmax+1])

    for n in range(nmax+1):
        for l in range(lmax+1):
            if verbose:
                print(f"calculating nmax = {n}, lmax = {l}")
            
            # truncate coefficients to desired length
            truncated_coefs = EXPtools.utils.coefficients.truncate_expansion(coefs, nmax=n, lmax=l)

            # calculate density field on the provided grid
            fields = pyEXP.field.FieldGenerator([coefs.Times()[t]], grid)
            surfaces = fields.points(basis, truncated_coefs)

            # get the density in proper units
            dens = surfaces[coefs.Times()[t]]['dens']

            # calculate MIRSE
            MISE[n, l] = np.mean((dens - gad_density)**2)

    return MISE
