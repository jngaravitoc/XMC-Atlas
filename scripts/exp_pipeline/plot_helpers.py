#!/usr/bin/env python3
"""
Build a basis expansion models
"""

import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

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




