#!/usr/bin/env python3
"""
Build a basis expansion models
"""

import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Circle
from EXPtools.visuals import use_exptools_style


import pyEXP
from metrics import mise, mirse

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

class FieldProjections:
    def __init__(self, grid, basis, coefs, times):
        self.grid = grid
        self.basis = basis
        self.coefs = coefs
        self.times = times
        assert np.shape(self.grid)[0] == 3
        assert np.shape(self.grid)[1] == np.shape(self.grid)[2] == np.shape(self.grid)[3]
        self.nbins = np.shape(self.grid)[1]
        self.mesh = np.zeros((self.nbins**3, 3))
        self.mesh[:,0] = self.grid[0].flatten()
        self.mesh[:,1] = self.grid[1].flatten()
        self.mesh[:,2] = self.grid[2].flatten()
		
    def compute_fields_in_points(self):
        """
        returns a dictionary 
        times is an array with the time values
        """
        fields = pyEXP.field.FieldGenerator(self.times, self.mesh)
        points = fields.points(self.basis, self.coefs)
        return points

    def twod_field(self, points, time, field):
        """
		Parameters:
		time : float 
			has to be in the list of times
		field: list of strings conatining desired fields
		"""
        available_fields = ['azi force', 'dens', 'dens m=0', 
							'dens m>0', 'mer force', 'potl', 
							'potl m=0', 'potl m>0', 'rad force']
		
        if isinstance(field, str):
            field = [field]
		
        for f in field:
            if f not in available_fields:
                raise ValueError("field value {} not available".format(f))

        if time not in self.times:
            raise ValueError("Requested time not in times")
		
        fields = []

        for f in field:
            field_arr = np.array(points[time][f]).reshape(self.nbins, self.nbins, self.nbins)
            fields.append(field_arr)
        return fields
	
    def kde_density(self, pos, mass, Ndens=64):
        """
        Ndens: n neighboors
        """
        kddens = pyEXP.util.KDdensity(mass = mass, pos = pos, Ndens = Ndens)
        kd_dens = np.zeros(self.nbins**3)
        for i in range(self.nbins**3):
            kd_dens[i] = kddens.getDensityAtPoint(self.mesh[i,0], self.mesh[i,1], self.mesh[i,2])

        return kd_dens.reshape(self.nbins, self.nbins, self.nbins)


def density_dashboard(kd_dens, dens_bfe, mise_logdens, mirse_dens, rvir=300, mean_axis=0):
    """
    Plot a 2x2 density comparison dashboard between KDE and BFE reconstructions.

    The figure shows projected density maps and their error diagnostics:
        (0,0) KDE density
        (0,1) BFE density
        (1,0) Log10 MIRSE residuals
        (1,1) MISE in log-density space

    Parameters
    ----------
    kd_dens : ndarray
        Kernel density estimate evaluated on a 3D grid with shape (Nx, Ny, Nz)
        or equivalent cube compatible with projection along `mean_axis`.

    dens_bfe : ndarray
        Density reconstructed from the basis function expansion on the same grid
        as `kd_dens`.

    mise_logdens : ndarray
        Mean Integrated Squared Error computed in log-density space. Must be a
        2D projected map compatible with visualization.

    mirse_dens : ndarray
        Mean Integrated Root Squared Error in density space, provided as a 2D map.

    mean_axis : int, optional
        Axis along which the density cube is averaged to obtain the projected map.
        Default is 0.

    rvir : float (keyword-only)
        Virial radius of the halo in the same spatial units as the map (kpc). A
        circle of this radius is overplotted on each panel.

    Returns
    -------
    matplotlib.figure.Figure
        The generated dashboard figure.

    Notes
    -----
    All panels use fixed color limits to enable visual comparison between
    reconstruction methods across different halos and simulations.
    """
    
    rvir = rvir

    # ---------- circle factory ----------
    def _make_r200_circle(radius, edgecolor='white', label=None):
        return Circle(
            (0, 0), radius,
            facecolor='none',
            edgecolor=edgecolor,
            linestyle='--',
            linewidth=1.0,
            label=label
        )

    use_exptools_style(usetex=True)
    # ---------- figure ----------
    fig, ax = plt.subplots(2, 2, figsize=(6,6), sharey=True)

    # ====== KDE ======
    im1 = ax[0,0].imshow(
        np.log10(np.mean(kd_dens, axis=mean_axis).T),
        extent=[-300, 300, -300, 300],
        cmap='twilight', vmin=-8, vmax=-3
    )
    ax[0,0].set_xlim(-300, 300)
    ax[0,0].set_ylim(-300, 300)
    ax[0,0].add_patch(_make_r200_circle(rvir))

    divider = make_axes_locatable(ax[0,0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im1, cax=cax)

    ax[0,0].set_title('KDE')

    # ====== BFE ======
    im2 = ax[0,1].imshow(
        np.log10(np.mean(dens_bfe, axis=mean_axis).T),
        extent=[-300, 300, -300, 300],
        cmap='twilight', vmin=-8, vmax=-3
    )
    ax[0,1].set_xlim(-300, 300)
    ax[0,1].set_ylim(-300, 300)
    ax[0,1].add_patch(_make_r200_circle(rvir))

    divider = make_axes_locatable(ax[0,1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im2, cax=cax, label=r'$\rm{Log_{10}} \rho$')

    ax[0,1].set_title('BFE')

    # ====== MIRSE ======
    im3 = ax[1,0].imshow(
        np.log10(mirse_dens.T),
        extent=[-300, 300, -300, 300],
        cmap='RdBu_r', vmin=-1.6, vmax=1.6
    )
    ax[1,0].set_xlim(-300, 300)
    ax[1,0].set_ylim(-300, 300)
    ax[1,0].add_patch(_make_r200_circle(rvir))

    divider = make_axes_locatable(ax[1,0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im3, cax=cax)

    ax[1,0].set_title(r'$\rm{Log_{10}}$\ MIRSE ($\rho$)')

    # ====== MISE ======
    im4 = ax[1,1].imshow(
        mise_logdens.T,
        extent=[-300, 300, -300, 300],
        cmap='gist_heat_r', vmin=0, vmax=0.2
    )
    ax[1,1].set_xlim(-300, 300)
    ax[1,1].set_ylim(-300, 300)
    ax[1,1].add_patch(_make_r200_circle(rvir, edgecolor='black'))

    divider = make_axes_locatable(ax[1,1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im4, cax=cax)

    ax[1,1].set_title(r'MISE ($\rm{Log_{10}}\rho$)')

    # ---------- shared formatting ----------
    for a in ax.flat:
        a.set_aspect('equal', adjustable='box')
        a.set_xlabel('kpc')

    ax[0,0].set_ylabel('kpc')
    ax[1,0].set_ylabel('kpc')

    fig.tight_layout()

    return fig
