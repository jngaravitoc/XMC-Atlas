#!/usr/bin/env python3
"""
Grid-based field evaluation and KDE density computation using pyEXP.
"""
import os
import sys
import numpy as np
import pyEXP
from field_io import read_bfe_fields, read_merged_kde_density

sys.path.append("../")
from config import DENSITY_FIELDS_PATHS, EXP_EXPANSIONS_PATH
sys.path.append("../exp_pipeline")
import metrics

class FieldsQuality:
    """
    Class for evaluating the quality of BFE (Basis Function Expansion) field reconstructions
    by comparing them to KDE (Kernel Density Estimation) density fields using various error metrics.

    Parameters
    ----------
    halo_id : int or str
        Identifier for the halo whose fields are being analyzed.
    field_type : str, optional
        The type of field to analyze (default is 'dens').
    """

    def __init__(self, halo_id, field_type='dens', component='lmc'):
        """
        Initialize the FieldsQuality object with file paths and time keys for the given halo.

        Parameters
        ----------
        halo_id : int or str
            Identifier for the halo whose fields are being analyzed.
        field_type : str, optional
            The type of field to analyze (default is 'dens').
        """
        self.field_type = field_type

        if component == 'MWhalo':
            fields_BFE_filename = "halo_{:04d}_BFE_fields.h5"
            fields_KDE_filename = "halo_{:04d}_kde_density.h5"
            coefs_file = os.path.join(EXP_EXPANSIONS_PATH, 
                                      "coefficients", 
                                      "halo_{:04d}_coefficients_center.h5".format(halo_id))
 
        elif component == 'lmc_init':
            fields_BFE_filename = "lmc_init_{:04d}_BFE_fields.h5"
            fields_KDE_filename = "lmc_init_{:04d}_kde_density.h5"
            coefs_file = os.path.join(EXP_EXPANSIONS_PATH, 
                                      "coefficients", 
                                      "lmc_init_{:04d}_coefficients.h5".format(halo_id))

        elif component == 'lmc_mean':
            fields_BFE_filename = "lmc_mean_{:04d}_BFE_fields.h5"
            fields_KDE_filename = "lmc_mean_{:04d}_kde_density.h5"
            coefs_file = os.path.join(EXP_EXPANSIONS_PATH, 
                                      "coefficients", 
                                      "lmc_mean_{:04d}_coefficients.h5".format(halo_id))


        self.bfe_fields_file = os.path.join(DENSITY_FIELDS_PATHS, 
                                    fields_BFE_filename.format(halo_id))

        self.kde_fields_file = os.path.join(DENSITY_FIELDS_PATHS, 
                                    fields_KDE_filename.format(halo_id))

        self.time_keys = pyEXP.coefs.Coefs.factory(coefs_file).Times()

    def metrics_evolution(self):
        """
        Compute the evolution of error metrics (MISE, log-MISE, MIRSE and their variances)
        between BFE and KDE density fields over all available snapshots.

        Returns
        -------
        dict
            Dictionary containing arrays for each metric across snapshots:
            'mise_median', 'mise_log_median', 'mirse_median',
            'mise_var', 'mise_log_var', 'mirse_var'.
        """
        kde_data, _kde_attrs = read_merged_kde_density(self.kde_fields_file)
        kde_snapshots = sorted(kde_data.keys())

        n_snaps = min(len(self.time_keys), len(kde_snapshots))
    
        mise_dens_arr = np.zeros(n_snaps)
        mise_logdens_arr = np.zeros(n_snaps)
        mirse_dens_arr = np.zeros(n_snaps)
        mise_dens_var = np.zeros(n_snaps)
        mise_logdens_var = np.zeros(n_snaps)
        mirse_dens_var = np.zeros(n_snaps)

        for i in range(n_snaps):
            t_key = self.time_keys[i]
            t_val = float(t_key)
            snap_key = kde_snapshots[i]

            # Read BFE density for this snapshot
            dens_bfe = read_bfe_fields(self.bfe_fields_file, "dens", t_val)

            # KDE density
            kd_dens = kde_data[snap_key]

            # Error metrics
            mise_val = metrics.mise(dens_bfe, kd_dens, axis=0)
            mise_log = metrics.mise(np.log10(dens_bfe), np.log10(kd_dens), axis=0)
            mirse_val = metrics.mirse(dens_bfe, kd_dens, axis=0)

            mise_dens_arr[i] = np.median(mise_val)
            mise_logdens_arr[i] = np.median(mise_log)
            mirse_dens_arr[i] = np.median(mirse_val)
            mise_dens_var[i] = np.var(mise_val)
            mise_logdens_var[i] = np.var(mise_log)
            mirse_dens_var[i] = np.var(mirse_val)

        return {
            'mise_median': mise_dens_arr,
            'mise_log_median': mise_logdens_arr,
            'mirse_median': mirse_dens_arr,
            'mise_var': mise_dens_var,
            'mise_log_var': mise_logdens_var,
            'mirse_var': mirse_dens_var,
        }



class FieldProjections:
    """Evaluate BFE fields and KDE densities on a regular 3-D Cartesian grid.

    Parameters
    ----------
    grid : ndarray, shape (3, N, N, N)
        Stacked meshgrid arrays (x, y, z).
    basis : pyEXP basis object
        The spherical basis expansion.
    coefs : pyEXP coefs object
        Coefficient container.
    times : array-like
        Snapshot times.
    """

    def __init__(self, grid, basis, coefs, times):
        self.grid = grid
        self.basis = basis
        self.coefs = coefs
        self.times = times
        assert np.shape(self.grid)[0] == 3
        assert np.shape(self.grid)[1] == np.shape(self.grid)[2] == np.shape(self.grid)[3]
        self.nbins = np.shape(self.grid)[1]
        self.mesh = np.zeros((self.nbins**3, 3))
        self.mesh[:, 0] = self.grid[0].flatten()
        self.mesh[:, 1] = self.grid[1].flatten()
        self.mesh[:, 2] = self.grid[2].flatten()

    def compute_fields_in_points(self):
        """Evaluate all BFE fields at every grid point for every time.

        Returns
        -------
        points : dict
            ``{time: {field_name: ndarray, ...}, ...}``
        """
        fields = pyEXP.field.FieldGenerator(self.times, self.mesh)
        points = fields.points(self.basis, self.coefs)
        return points

    def twod_field(self, points, time, field):
        """Extract one or more 3-D field arrays for a single snapshot.

        Parameters
        ----------
        points : dict
            Output of :meth:`compute_fields_in_points`.
        time : float
            Must be present in ``self.times``.
        field : str or list of str
            Field name(s) to extract.

        Returns
        -------
        list of ndarray
            Each array has shape ``(nbins, nbins, nbins)``.
        """
        available_fields = [
            'azi force', 'dens', 'dens m=0',
            'dens m>0', 'mer force', 'potl',
            'potl m=0', 'potl m>0', 'rad force',
        ]

        if isinstance(field, str):
            field = [field]

        for f in field:
            if f not in available_fields:
                raise ValueError(f"field value {f!r} not available")

        if time not in self.times:
            raise ValueError("Requested time not in times")

        fields = []
        for f in field:
            field_arr = np.array(points[time][f]).reshape(
                self.nbins, self.nbins, self.nbins
            )
            fields.append(field_arr)
        return fields

    def kde_density(self, pos, mass, Ndens=64):
        """Compute KDE density at every grid point.

        Parameters
        ----------
        pos : ndarray, shape (Np, 3)
            Particle positions.
        mass : ndarray, shape (Np,)
            Particle masses.
        Ndens : int, optional
            Number of nearest neighbours (default 64).

        Returns
        -------
        kd_dens : ndarray, shape (nbins, nbins, nbins)
        """
        kddens = pyEXP.util.KDdensity(mass=mass, pos=pos, Ndens=Ndens)
        kd_dens = np.zeros(self.nbins**3)
        for i in range(self.nbins**3):
            kd_dens[i] = kddens.getDensityAtPoint(
                self.mesh[i, 0], self.mesh[i, 1], self.mesh[i, 2]
            )
        return kd_dens.reshape(self.nbins, self.nbins, self.nbins)


def fields_residuals_slider(
    data1: np.ndarray,
    data2: np.ndarray,
    data3: np.ndarray,
    axis: int = 2,
    cmaps: tuple = ("viridis", "plasma", "cividis"),
    levels: int = 20,
    filled: bool = True,
    titles: tuple = ("Field 1", "Field 2", "Field 3"),
    coord_range: tuple = (-1.0, 1.0),
):
    """
    Visualize three (T, N, N, N) arrays as side-by-side 2D contour plots
    over time, with a single shared slider to scrub the T dimension.

    The spatial mean is taken along `axis` (0-indexed into the three spatial
    axes 1-3) to collapse each array to shape (T, N, N).

    Subplots share the y-axis. Each subplot has its own colorbar and title.
    Axis labels are "x / r200c" and "y / r200c".

    Parameters
    ----------
    data1, data2, data3 : np.ndarray, shape (T, N, N, N)
        The three 4-D fields to display.
    axis        : int   – spatial axis to average over (0, 1, or 2). Default 2.
    cmaps       : 3-tuple of colormap names, one per subplot.
    levels      : number of contour levels. Default 20.
    filled      : True → contourf (filled), False → contour (lines only).
    titles      : 3-tuple of subplot titles.
    coord_range : (min, max) extent along both spatial axes, in units of r200c.
                  Default (-1, 1) so ticks run from -1 to 1 r200c.
    """
    datasets = (data1, data2, data3)
    for i, d in enumerate(datasets):
        if d.ndim != 4:
            raise ValueError(f"data{i+1} must be 4-D, got shape {d.shape}")
    shapes = [d.shape for d in datasets]
    if len(set(shapes)) != 1:
        raise ValueError(f"All three arrays must share the same shape. Got {shapes}")

    T = data1.shape[0]
    spatial_axis = axis + 1

    # Reduce each dataset: (T, N, N, N) → (T, N, N)
    reduced = [d.mean(axis=spatial_axis) for d in datasets]

    # Fixed colorbar limits for each subplot
    vlims = [(5, 14), (5, 14), (-1, 1)]
    # Pre-compute explicit level arrays so colorbars span exactly vmin→vmax
    level_arrays = [np.linspace(vmin, vmax, levels + 1) for (vmin, vmax) in vlims]

    Ny, Nx = reduced[0].shape[1], reduced[0].shape[2]
    x_lin = np.linspace(coord_range[0], coord_range[1], Nx)
    y_lin = np.linspace(coord_range[0], coord_range[1], Ny)
    X, Y = np.meshgrid(x_lin, y_lin)

    # ── layout ────────────────────────────────────────────────────────────────
    #   Row 0  : three contour subplots (tall)
    #   Row 1  : single time slider     (thin)
    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(
        2, 3,
        height_ratios=[18, 1],
        hspace=0.45,
        wspace=0.08,
    )

    axes = []
    for col in range(3):
        if col == 0:
            ax = fig.add_subplot(gs[0, col])
        else:
            ax = fig.add_subplot(gs[0, col], sharey=axes[0])
            ax.tick_params(labelleft=False)
        axes.append(ax)

    # Single slider spanning all three columns
    ax_slider = fig.add_subplot(gs[1, :])
    suptitle = fig.suptitle("Halo 21537, snapshot = 0", fontsize=14, fontweight="bold")
    plot_fn_name = "contourf" if filled else "contour"

    # ── draw initial frame (t=0) ──────────────────────────────────────────────
    cbars = []

    for col, (ax, red, lvls, (vmin, vmax), cmap, ttl) in enumerate(
        zip(axes, reduced, level_arrays, vlims, cmaps, titles)
    ):
        plot_fn = getattr(ax, plot_fn_name)
        cs = plot_fn(X, Y, red[0], levels=lvls, vmin=vmin, vmax=vmax, cmap=cmap,
                     extend="both")

        cbar = fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        cbars.append(cbar)

        ax.set_title(ttl, fontsize=12, fontweight="bold", pad=6)
        ax.set_xlabel(r"$x\,/\,r_\mathrm{200c}$", fontsize=14)
        if col == 0:
            ax.set_ylabel(r"$y\,/\,r_\mathrm{200c}$", fontsize=14)

        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect("equal")
        circle = plt.Circle((0, 0), 1.0, color="white", fill=False, linestyle="--", linewidth=1.5)
        ax.add_patch(circle)

    # ── slider ────────────────────────────────────────────────────────────────
    slider = Slider(ax_slider, r"$\rm{snapshot}$", 0, T - 1, valinit=0, valstep=1, color="#4a90d9")

    def _update(val: float) -> None:
        t = int(slider.val)
        for ax, red, lvls, (vmin, vmax), cmap in zip(axes, reduced, level_arrays, vlims, cmaps):
            for coll in ax.collections:
                coll.remove()
            plot_fn = getattr(ax, plot_fn_name)
            plot_fn(X, Y, red[t], levels=lvls, vmin=vmin, vmax=vmax, cmap=cmap,
                    extend="both")
            circle = plt.Circle((0, 0), 1.0, color="white", fill=False, linestyle="--", linewidth=1.5)
            ax.add_patch(circle)
            fig.canvas.draw_idle()

        suptitle.set_text(f"Halo 21537, snapshot = {t}")
       
    slider.on_changed(_update)

    plt.show()
    return fig, axes, slider
