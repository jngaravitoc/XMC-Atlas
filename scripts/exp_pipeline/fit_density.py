import numpy as np
import EXPtools
from scipy.optimize import curve_fit


def fit_density_profile(radii, amplitud, scale_radius, core):
    profile_fit = EXPtools.Profiles(radii, scale_radius, amplitud, alpha=1, beta=3.0)
    rho_fit = profile_fit.power_halo(core)
    return rho_fit

def fit_profile(r_part, rho_part):
	fit_params, fit_cov = curve_fit(
		fit_density_profile, 
		r_part, 
		rho_part, 
		p0=np.array([1e-1, 20, 10]), 
		bounds=([5e-4, 5, 0.0], [1, 50, 20]))
	rho_fit = fit_density_profile(r_part, *fit_params)
	return rho_fit, fit_params
