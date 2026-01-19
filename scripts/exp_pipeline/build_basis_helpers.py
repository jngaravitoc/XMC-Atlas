#!/usr/bin/env python3
"""
Build a basis expansion models
"""

import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#import EXPtools

def compute_GC21_mw_halo_basis():
    """Main driver for building the halo basis expansion."""
    
    if not os.path.exists(POTENTIAL_NAME):
        raise FileNotFoundError(f"{POTENTIAL_NAME} not found")

    # Load potential
    GC22_pot = gp.potential.load(POTENTIAL_NAME)
    GC22_MW_DMhalo = GC21_pot["halo"]

    # Halo parameters
    Mtot = GC22_MW_DMhalo.parameters["m"].value
    rbins = np.linspace(2e-3, 300, N_RBINS)
    # Halo profile
    halo_profile = GC22_MW_DMhalo.density(rbins)

    # TODO: define rcen properly
    rcen = 1.0  

    # Build model
    model = make_model(
        rcen,
        halo_profile,
        Mtotal=Mtot,
        output_filename=OUTPATH+MODEL_NAME,
        physical_units=True,
    )
    R = model["radius"]

    # Build basis
    config = make_config(
        basis_id="sphereSL",
        lmax=LMAX,
        nmax=NMAX,
        rmapping=R[0],
        modelname=OUTPATH+MODEL_NAME,
        cachename=OUTPATH+CACHE_NAME,
    )

    basis = pyEXP.basis.Basis.factory(config)
    write_basis(config, OUTPATH+BASIS_NAME)

    print(f"Basis written to {OUTPATH+BASIS_NAME}")



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
                "pcavar": True,
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
                "pcavar": True,
            },
        }

    return yaml.dump(config_dict, sort_keys=False)


def plot_profiles(radius, density, time, title='Profile Evolution',
                  r_fit=None, rho_fit=None, fit_label='Fit', filename=None):
    """
    Plot density profiles as a function of radius, colored by time,
    optionally overlaying a single fit curve, and save as a PNG.

    Parameters
    ----------
    radius : array-like, shape (N,)
        Radial coordinates for the profiles.
    density : array-like, shape (M, N)
        Density profiles at different times. `density[i]` corresponds to `time[i]`.
    time : array-like, shape (M,)
        Times corresponding to each density profile.
    title : str, optional
        Title of the plot (default is 'Profile Evolution').
    r_fit : array-like, optional
        Radial coordinates of the fit curve.
    rho_fit : array-like, optional
        Density values of the fit curve.
    fit_label : str, optional
        Label for the fit curve (default 'Fit').
    filename : str, optional
        If provided, saves the figure to this file (e.g., 'profiles.png').

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    ax : matplotlib.axes.Axes
        The axes object with the plot.
    """
    radius = np.asarray(radius)
    density = np.asarray(density)
    time = np.asarray(time)

    if density.shape[0] != len(time):
        raise ValueError("density.shape[0] must match length of time")

    # Normalize time for colormap
    norm = mpl.colors.Normalize(vmin=np.min(time), vmax=np.max(time))
    cmap = mpl.cm.cividis  # colorblind-friendly sequential colormap

    fig, ax = plt.subplots(figsize=(6, 5))

    # Plot all profiles
    for i in range(len(time)):
        color = cmap(norm(time[i]))
        ax.loglog(radius, density[i], color=color, alpha=0.8, linewidth=1.0)

    # Overlay fit curve if provided
    if r_fit is not None and rho_fit is not None:
        r_fit = np.asarray(r_fit)
        rho_fit = np.asarray(rho_fit)
        ax.loglog(r_fit, rho_fit, color='k', linestyle='--', linewidth=2.0, label=fit_label)
        ax.legend()

    # Colorbar
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(time)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Time [Gyr]")

    ax.set_xlabel("Radius")
    ax.set_ylabel("Density")
    ax.set_title(title)

    ax.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()

    # Save to file if requested
    if filename is not None:
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {filename}")

    return fig, ax




