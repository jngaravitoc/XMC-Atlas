import numpy as np

def check_monotonic_profiles(density):
    """
    Check that each density profile (row) in a 2D array is monotonically decreasing.
    
    Parameters
    ----------
    density : array-like, shape (M, N)
        Array of density profiles. Each row corresponds to a profile at a given time.
    
    Returns
    -------
    all_monotonic : bool
        True if all profiles are monotonically decreasing, False otherwise.
    failed_indices : list of int
        List of row indices that are not monotonically decreasing.
    """
    density = np.asarray(density)
    failed_indices = []
    failed_snap = []
    for idx, profile in enumerate(density):
        if not np.all(profile[:-1] >= profile[1:]):
            failed_indices.append(idx)
    
    all_monotonic = len(failed_indices) == 0
    return all_monotonic, failed_indices


def check_monotonic_contiguous_snapshots(suffixes):
    """
    Check that suffixes are integer-valued, monotonic, and contiguous.

    Parameters
    ----------
    suffixes : sequence of str or int
        Snapshot suffixes (e.g. ["000", "001", "002"]).

    Returns
    -------
    ok : bool
        True if suffixes are valid, False otherwise.
    missing : list of int
        Missing integer values (empty if none).
    """
    try:
        values = sorted(int(s) for s in suffixes)
    except ValueError as exc:
        raise ValueError("All suffixes must be convertible to integers.") from exc

    if len(values) == 0:
        return True, []

    expected = list(range(values[0], values[-1] + 1))
    missing = sorted(set(expected) - set(values))

    ok = len(missing) == 0
    return ok, missing

