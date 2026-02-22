import sys
import numpy as np

sys.path.append("./exp_pipeline/")

from plot_helpers import FieldProjections, density_dashboard
from metrics import mise, mirse


def compute_dashboard(grid, basis, coefs, times, pos, mass, rvir, time_index=0, return_mises=False):
    """
    Compute KDE/BFE density comparison dashboard.

    Parameters
    ----------
    grid : ndarray (3, N, N, N)
    basis : EXP basis object
    coefs : ndarray
    times : ndarray
    pos : ndarray (Np,3)
    mass : ndarray (Np,)
    rvir : float
    time_index : int

    Returns
    -------
    fig : matplotlib.figure.Figure
    """

    # --------------------------------------------------
    # 1. Compute fields from BFE
    # --------------------------------------------------
    FP = FieldProjections(grid, basis, coefs, times)

    print("Computing EXP fields...")
    points = FP.compute_fields_in_points()

    time = times[time_index]
    dens_bfe = FP.twod_field(points, time, 'dens')[0]

    # --------------------------------------------------
    # 2. KDE density
    # --------------------------------------------------
    print("Computing KDE density...")
    kd_dens = FP.kde_density(pos, mass)

    nbins = dens_bfe.shape[0]

    # --------------------------------------------------
    # 3. Error metrics
    # --------------------------------------------------
    print("Computing MISE & MIRSE...")

    mise_dens = mise(dens_bfe, kd_dens.reshape(nbins, nbins, nbins), axis=0)

    mise_logdens = mise(
        np.log10(dens_bfe),
        np.log10(kd_dens.reshape(nbins, nbins, nbins)),
        axis=0
    )

    mirse_dens = mirse(dens_bfe, kd_dens.reshape(nbins, nbins, nbins), axis=0)

    # --------------------------------------------------
    # 4. Plot dashboard
    # --------------------------------------------------
    print("Plotting dashboard...")
    fig = density_dashboard(
        kd_dens.reshape(nbins, nbins, nbins),
        dens_bfe,
        mise_logdens,
        mirse_dens,
        mean_axis=0,
        rvir=rvir
    )

    if return_mises == True:
        return fig, mise_dens, mise_logdens, mirse_dens
    else:
        return fig

