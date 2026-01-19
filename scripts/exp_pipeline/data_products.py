import h5py
import numpy as np
from pathlib import Path


def write_density_profiles(
    suite_id: int,
    snaps: np.ndarray,
    rbins : np.ndarray,
    profiles: np.ndarray,
    filename: str = "simulation_outputs.h5",
):
    """
    Write or append simulation outputs to an HDF5 file.

    Parameters
    ----------
    suite_id : int
        Simulation identifier (0–2000), written as 'suite_XXXX'.
    snap : ndarray, shape (T,)
        Time array.
    profiles : ndarray, shape (T, R)
        Profile data corresponding to each time step.
    filename : str, optional
        Output HDF5 filename.

    Raises
    ------
    ValueError
        If array shapes are inconsistent.
    """
    snaps = np.asarray(snaps)
    rbins = np.asarray(rbins)
    profiles = np.asarray(profiles)

    if snaps.ndim != 1:
        raise ValueError("snaps must be a 1D array")

    if rbins.ndim != 1:
        raise ValueError("rbins must be a 1D array")

    if profiles.ndim != 2:
        raise ValueError("profiles must be a 2D array (T, R)")

    if profiles.shape[0] != snaps.shape[0]:
        raise ValueError("profiles.shape[0] must match len(time)")

    suite_name = f"suite_{suite_id:04d}"

    # Ensure directory exists if filename includes a path
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(filename, mode="a") as f:
        # Create or open the suite group
        grp = f.require_group(suite_name)

        # Overwrite datasets if they already exist
        if "snap" in grp:
            del grp["snap"]
    
        if "profile" in grp:
            del grp["profile"]

        if "rbins" in grp:
            del grp["rbins"]

        grp.create_dataset(
            "snap",
            data=snaps,
            dtype=snaps.dtype,
            compression="gzip",
            compression_opts=4,
        )
        
        grp.create_dataset(
            "rbins",
            data=rbins,
            dtype=rbins.dtype,
            compression="gzip",
            compression_opts=4,
        )

        grp.create_dataset(
            "profile",
            data=profiles,
            dtype=profiles.dtype,
            compression="gzip",
            compression_opts=4,
        )

