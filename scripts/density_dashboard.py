import sys
import numpy as np

sys.path.append("./exp_pipeline/")

from plot_helpers import FieldProjections, density_dashboard
from metrics import mise, mirse


def compute_bfe_fields(grid, basis, coefs, eval_times):
    """
    Compute BFE fields on a grid for given evaluation times.

    Parameters
    ----------
    grid : ndarray (3, N, N, N)
    basis : EXP basis object
    coefs : ndarray
    eval_times : list of floats

    Returns
    -------
    list of [time: field_dict]
    """
    times = coefs.Times()
    FP = FieldProjections(grid, basis, coefs, times)

    print("Computing EXP fields...")
    points = FP.compute_fields_in_points()

    fields = []
    for t in eval_times:
        fields.append(FP.twod_field(points, t, 'dens')[0])

    return fields, FP

def compute_dashboard(FP, dens_bfe, pos, mass, rvir, return_mises=False):
    """
    Compute KDE/BFE density comparison dashboard.

    Parameters
    ----------
    FP : FieldProjections object
    dens_bfe : ndarray (nbins, nbins, nbins)
        3D BFE density field
    pos : ndarray (Np,3)
        Particle positions
    mass : ndarray (Np,)
        Particle masses
    rvir : float
        Virial radius
    return_mises : bool
        If True, return error metrics

    Returns
    -------
    fig : matplotlib.figure.Figure
    mise_dens, mise_logdens, mirse_dens : ndarray (optional)
    """
    
    # --------------------------------------------------
    # 0. Validate inputs
    # --------------------------------------------------
    dens_bfe = np.asarray(dens_bfe)
    
    if dens_bfe.ndim != 3:
        raise ValueError(f"dens_bfe must be 3D, got shape {dens_bfe.shape}")
    
    nbins = dens_bfe.shape[0]
    
    if dens_bfe.shape != (nbins, nbins, nbins):
        raise ValueError(
            f"dens_bfe must be cubic, got shape {dens_bfe.shape}. "
            f"Expected ({nbins}, {nbins}, {nbins})"
        )
    
    if FP.nbins != nbins:
        raise ValueError(
            f"dens_bfe grid size ({nbins}) does not match FP grid size ({FP.nbins})"
        )


    # --------------------------------------------------
    # 1. KDE density
    # --------------------------------------------------
    print("Computing KDE density...")
    kd_dens = FP.kde_density(pos, mass)
    kd_dens_3d = kd_dens.reshape(nbins, nbins, nbins)


    # --------------------------------------------------
    # 3. Error metrics
    # --------------------------------------------------
    print("Computing MISE & MIRSE...")

    mise_dens = mise(dens_bfe, kd_dens_3d, axis=0)

    mise_logdens = mise(
        np.log10(dens_bfe),
        np.log10(kd_dens_3d),
        axis=0
    )

    mirse_dens = mirse(dens_bfe, kd_dens_3d, axis=0)

    # --------------------------------------------------
    # 4. Plot dashboard
    # --------------------------------------------------
    print("Plotting dashboard...")
    fig = density_dashboard(
        kd_dens_3d,
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

