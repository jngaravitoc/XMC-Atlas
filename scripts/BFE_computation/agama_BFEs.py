"""
Agama Basis Function Expansion fitting for MW/LMC N-body simulations for XMC Atlas format. 

Fits Multipole (spherical harmonic) and CylSpline (azimuthal) expansions
to particle data from N-body snapshots using the Agama library.

Utilities for storing and loading Agama potential coefficient files via HDF5.

Each snapshot's coefficient text (normally a plain ``.coef_*`` file) is stored
as a UTF-8 string dataset inside a named group, e.g.::

    snap_000/coefs   (str)
    snap_001/coefs   (str)
    ...

Potentials are reconstructed at runtime by materialising the string to a
fast temporary file (RAM-backed ``/dev/shm`` when available, otherwise the
system temp directory), creating the Agama object, then removing the file.

Author: Arpit Arora 
Date: 2026-02-12
"""
from __future__ import annotations

from typing import Sequence
import os, uuid, h5py, re, tempfile
import numpy as np
from pathlib import Path
from typing import Optional, Sequence, Union, Iterable, List, Callable
import agama
agama.setUnits(mass=1, length=1, velocity=1) # Msol, kpc, km/s. Time is in kpc/(kms/s)

# ---------------------------------------------------------------------------
# Component-to-expansion routing
# ---------------------------------------------------------------------------
_MW_SPHERICAL_KEYS = {"MWhalo", "MWhaloiso", "MWbulge"}
_MW_DISK_KEYS = {"MWdisk"}
_LMC_KEYS = {"LMChalo"}

__all__ = [
    "fitAgamaBFE",
    "write_coef_to_h5",
    "write_snapshot_coefs_to_h5",
    "load_coef_from_h5",
    "load_agama_potential_from_h5",
    "load_agama_evolving_potential_from_h5",
]

def _resolve_center(
    center_array: np.ndarray,
    nsnap: int,
    label: str,
) -> np.ndarray:
    """
    Extract a 3-element position vector from a flexible center specification.

    Handles four layouts:
        (3,)   -> single position, used directly.
        (6,)   -> pos+vel packed; first 3 elements taken as position.
        (N, 3) -> time series of positions; row ``nsnap`` is selected.
        (N, 6) -> time series of pos+vel; row ``nsnap``, first 3 columns.

    Parameters
    ----------
    center_array : np.ndarray
        Center coordinate(s).  Accepted shapes: ``(3,)``, ``(6,)``,
        ``(N, 3)``, or ``(N, 6)``.
    nsnap : int
        Snapshot index used when *center_array* is 2-D.
    label : str
        Human-readable label (e.g. ``'mw_center'``) for error messages.

    Returns
    -------
    np.ndarray
        Shape ``(3,)`` position vector in the same units as the input.

    Raises
    ------
    IndexError
        If ``nsnap`` exceeds the number of rows in a 2-D array.
    ValueError
        If the array shape is not one of the recognised layouts.
    """
    arr = np.asarray(center_array, dtype=np.float64)

    # --- 1-D cases ---
    if arr.ndim == 1:
        if arr.shape[0] == 3:
            return arr
        if arr.shape[0] == 6:
            return arr[:3]
        raise ValueError(
            f"{label}: 1-D array must have length 3 or 6, got {arr.shape[0]}"
        )

    # --- 2-D cases ---
    if arr.ndim == 2:
        if arr.shape[1] not in (3, 6):
            raise ValueError(
                f"{label}: 2-D array must have 3 or 6 columns, got shape {arr.shape}"
            )
        if nsnap >= arr.shape[0]:
            raise IndexError(
                f"{label}: nsnap={nsnap} out of range for array with "
                f"{arr.shape[0]} rows"
            )
        return arr[nsnap, :3]

    raise ValueError(f"{label}: expected 1-D or 2-D array, got ndim={arr.ndim}")


def _extract_particles(
    part: dict,
    keys: set[str],
    center: np.ndarray,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Gather and recenter particles from one or more component keys.

    Parameters
    ----------
    part : dict
        Top-level particle dictionary keyed by component name.  Each value
        must itself be a dict with at least ``'pos'`` and ``'mass'`` entries.
    keys : set of str
        Component names to include (e.g. ``{'MWhalo', 'MWbulge'}``).
    center : np.ndarray
        Shape ``(3,)`` position used to recenter the particles.
    verbose : bool, optional
        Print per-component diagnostics.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray) or None
        ``(positions, masses)`` where *positions* has shape ``(N, 3)`` and
        *masses* has shape ``(N,)``.  Returns ``None`` if no matching
        component is found in *part*.
    """
    pos_list: list[np.ndarray] = []
    mass_list: list[np.ndarray] = []

    for key in sorted(keys):
        if key not in part:
            continue
        comp = part[key]

        pos = np.asarray(comp["pos"], dtype=np.float64)
        mass = np.asarray(comp["mass"], dtype=np.float64)

        # Gracefully handle pos arrays that carry velocity columns (N, 6)
        if pos.ndim == 2 and pos.shape[1] > 3:
            pos = pos[:, :3]

        # Recenter
        pos = pos - center[np.newaxis, :]

        pos_list.append(pos)
        mass_list.append(mass.ravel())

        if verbose:
            print(
                f"  Component '{key}': {len(mass):,} particles, "
                f"M_tot = {mass.sum():.4e}"
            )

    if not pos_list:
        return None

    return np.concatenate(pos_list, axis=0), np.concatenate(mass_list, axis=0)


def fitAgamaBFE(
    part: dict,
    center_coords: dict,
    nsnap: int,
    *,
    lmax: int | Sequence[int] = 6,
    mmax: int | Sequence[int] = 6,
    sym: str | Sequence[str] = "n",
    rmin_exp: float = 0.0,
    rmax_exp: float = 500.0,
    zmin_exp: float = 0.0,
    zmax_exp: float = 20.0,
    rmax_lmc: float = 500.0,
    gridsizeR: int = 25,
    gridsizez: int = 49,
    OUTPUT_PATH: str = "potential/",
    verbose: bool = False,
) -> None:
    """
    Fit Agama grid-based potential expansions to N-body particle data.

    Constructs Multipole (spherical harmonic) and, when a disk component is
    present, CylSpline (azimuthal) basis-function expansions for the Milky
    Way and, optionally, the LMC.  Fitted coefficients are written to
    individual files under ``OUTPUT_PATH``.

    **Unit responsibility.** The caller must ensure that the units used in
    ``part`` (positions in kpc, masses in the unit set by
    ``agama.setUnits``, etc.) are consistent with the Agama unit system.
    No unit conversion is performed inside this function.

    Component routing
    -----------------
    * ``MWhalo``, ``MWhaloiso``, ``MWbulge``
        → MW Multipole (spherical harmonic) expansion.
    * ``MWdisk``
        → MW CylSpline (azimuthal) expansion.
    * ``LMChalo``
        → LMC Multipole expansion (only when ``lmc_center`` is supplied).

    Components that are absent from ``part`` are silently skipped.

    Parameters
    ----------
    part : dict
        Particle data keyed by component name (e.g. ``'MWhalo'``).
        Each value is a dict with at least ``'pos'`` (N, 3) or (N, 6) and
        ``'mass'`` (N,) arrays.  If ``'pos'`` has > 3 columns only the
        first three (spatial coordinates) are used.
    center_coords : dict
        Center-of-mass coordinates.  Must contain ``'mw_center'`` and,
        if an LMC expansion is desired, ``'lmc_center'``.  Each entry
        may be:

        * shape ``(3,)``   – single position (used as-is),
        * shape ``(6,)``   – position + velocity (first 3 taken),
        * shape ``(N, 3)`` – time-series of positions (row *nsnap* used),
        * shape ``(N, 6)`` – time-series of pos+vel (row *nsnap*, :3).
    nsnap : int
        Snapshot index.  Selects the row from 2-D center arrays and is
        embedded in the output file names (zero-padded to 3 digits).
    lmax : int or sequence of int, optional
        Spherical-harmonic order(s) for Multipole fits.  Multiple values
        trigger one fit per value.  Default is 6.
    mmax : int or sequence of int, optional
        Azimuthal-harmonic order(s) for CylSpline fits.  Default is 6.
    sym : str or sequence of str, optional
        Symmetry flag(s).  Accepted single-character codes:

        * ``'n'`` – none
        * ``'a'`` – axisymmetric
        * ``'s'`` – spherical
        * ``'t'`` – triaxial

        Default is ``'n'``.
    rmin_exp : float, optional
        Inner radius for the expansion grid in kpc.  ``0`` lets Agama
        auto-detect from the particle distribution.  Default is 0.
    rmax_exp : float, optional
        Outer radius for the MW expansion grid in kpc.  Default is 500.
    zmin_exp : float, optional
        innermost grid point for the CylSpline grid in kpc.  Default is 0.
    zmax_exp : float, optional
        Maximum outermost grid for the CylSpline grid in kpc.  Default
        is 20. That means a grid between -20 and 20 kpc, so the total height is 40 kpc.
    rmax_lmc : float, optional
        Outer radius for the LMC expansion grid in kpc.  Default is 100.
    gridsizeR : int, optional
        Radial grid points.  Default is 25.
    gridsizez : int, optional
        Vertical grid points.  Default is 49.
    OUTPUT_PATH : str, optional
        Directory for output coefficient files.  Created if it does not
        exist.  Default is ``'potential/'``.
    verbose : bool, optional
        Print step-by-step diagnostics.  Default is False.

    Returns
    -------
    None
        All output is written to disk.

    Raises
    ------
    TypeError
        If ``center_coords`` is not a dict.
    KeyError
        If ``'mw_center'`` is missing from *center_coords*.
    ValueError
        If symmetry codes or harmonic orders are invalid, or if center
        arrays have unexpected shapes.
    IndexError
        If ``nsnap`` is out of range for a 2-D center array.
    RuntimeError
        If no fittable MW components are found in ``part``.
    """

    # ------------------------------------------------------------------
    # 0. Input validation
    # ------------------------------------------------------------------
    ALLOWED_SYMS = {"n": "none", "a": "axi", "s": "sph", "t": "triax"}
    nsnap_str = f"{int(nsnap):03d}"

    # --- symmetry ---
    syms = [sym] if isinstance(sym, str) else list(sym)
    for s in syms:
        if s not in ALLOWED_SYMS:
            raise ValueError(
                f"Unknown symmetry '{s}'. Allowed: {list(ALLOWED_SYMS)}"
            )

    # --- harmonic orders ---
    def _to_int_list(val: int | Sequence[int], name: str) -> list[int]:
        lst = [val] if isinstance(val, int) else list(val)
        if any(not isinstance(v, int) or v < 0 for v in lst):
            raise ValueError(f"{name} entries must be non-negative integers.")
        return lst

    lmax_list = _to_int_list(lmax, "lmax")
    mmax_list = _to_int_list(mmax, "mmax")

    # --- center coordinates ---
    if not isinstance(center_coords, dict):
        raise TypeError("center_coords must be a dictionary.")
    if "mw_center" not in center_coords:
        raise KeyError("center_coords must contain 'mw_center'.")

    mw_pos = _resolve_center(center_coords["mw_center"], nsnap, "mw_center")

    lmc_avail = "lmc_center" in center_coords
    lmc_pos = (
        _resolve_center(center_coords["lmc_center"], nsnap, "lmc_center")
        if lmc_avail
        else None
    )

    # --- rmin: 0 means auto-detect (passed as-is to Agama) ---
    rmin = rmin_exp if rmin_exp > 0.0 else 0.0

    # --- output directory ---
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Assemble particle arrays per expansion type
    # ------------------------------------------------------------------
    if verbose:
        print("=" * 72)
        print(f"Snapshot {nsnap_str} — assembling particle arrays")
        print("=" * 72)
        print(f"  MW center  : {mw_pos}")
        if lmc_avail:
            print(f"  LMC center : {lmc_pos}")

    # --- MW spherical (halo + bulge) ---
    if verbose:
        print("\n  [MW spherical components]")
    mw_sph = _extract_particles(part, _MW_SPHERICAL_KEYS, mw_pos, verbose=verbose)

    # --- MW disk (azimuthal) ---
    if verbose:
        print("\n  [MW disk components]")
    mw_disk = _extract_particles(part, _MW_DISK_KEYS, mw_pos, verbose=verbose)

    # --- LMC spherical ---
    lmc_sph = None
    if lmc_avail:
        if verbose:
            print("\n  [LMC components]")
        lmc_sph = _extract_particles(part, _LMC_KEYS, lmc_pos, verbose=verbose)
        if lmc_sph is None and verbose:
            print("    No LMC particles found despite lmc_center being provided.")

    # Guard: at least something to fit
    if mw_sph is None and mw_disk is None:
        raise RuntimeError(
            "No fittable MW components found in `part`. Expected at least one "
            f"of: {_MW_SPHERICAL_KEYS | _MW_DISK_KEYS}"
        )

    disk_fit = mw_disk is not None
    lmc_fit = lmc_sph is not None

    if verbose:
        n_mw_s = mw_sph[1].shape[0] if mw_sph else 0
        n_mw_d = mw_disk[1].shape[0] if mw_disk else 0
        n_lmc = lmc_sph[1].shape[0] if lmc_sph else 0
        print(f"\n  Particle counts  →  MW sph: {n_mw_s:,}  |  "
              f"MW disk: {n_mw_d:,}  |  LMC: {n_lmc:,}")
        print(f"  Disk expansion   →  {'enabled' if disk_fit else 'disabled'}")
        print(f"  LMC expansion    →  {'enabled' if lmc_fit else 'disabled'}")

    # ------------------------------------------------------------------
    # 2. Fit expansions over the (sym × lmax × mmax) grid
    # ------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 72)
        print("Fitting basis-function expansions")
        print("=" * 72)

    for s in syms:
        sym_label = ALLOWED_SYMS[s]

        for l_order in lmax_list:
            for m_order in mmax_list:

                tag = f"sym={sym_label}, lmax={l_order}, mmax={m_order}"

                # ---------- MW Multipole (spherical) ----------
                if mw_sph is not None:
                    pos_mul, m_mul = mw_sph
                    if verbose:
                        print(f"\n  → MW Multipole  [{tag}]")

                    p_MW_sph = agama.Potential(
                        type="Multipole",
                        particles=(pos_mul, m_mul),
                        lmax=l_order,
                        symmetry=sym_label,
                        rmin=rmin,
                        rmax=rmax_exp,
                        gridSizeR=gridsizeR,
                    )

                    fname_mul = (
                        f"{nsnap_str}.MW.{sym_label}_{l_order}.coef_mult"
                    )
                    outpath_mul = os.path.join(OUTPUT_PATH, fname_mul)
                    p_MW_sph.export(outpath_mul)

                    if verbose:
                        print(f"    Saved → {outpath_mul}")

                # ---------- MW CylSpline (azimuthal / disk) ----------
                if disk_fit:
                    pos_azi, m_azi = mw_disk
                    if verbose:
                        print(f"\n  → MW CylSpline  [{tag}]")

                    p_MW_azi = agama.Potential(
                        type="CylSpline",
                        particles=(pos_azi, m_azi),
                        mmax=m_order,
                        symmetry=sym_label,
                        rmin=rmin,
                        rmax=rmax_exp,
                        zmin=zmin_exp,
                        zmax=zmax_exp,
                        gridSizeR=gridsizeR,
                        gridsizez=gridsizez,
                    )

                    fname_azi = (
                        f"{nsnap_str}.MW.{sym_label}_{m_order}.coef_cylsp"
                    )
                    outpath_azi = os.path.join(OUTPUT_PATH, fname_azi)
                    p_MW_azi.export(outpath_azi)

                    if verbose:
                        print(f"    Saved → {outpath_azi}")

                # ---------- LMC Multipole (spherical) ----------
                if lmc_fit:
                    pos_lmc, m_lmc = lmc_sph
                    if verbose:
                        print(f"\n  → LMC Multipole [{tag}]")

                    p_LMC_sph = agama.Potential(
                        type="Multipole",
                        particles=(pos_lmc, m_lmc),
                        lmax=l_order,
                        symmetry=sym_label,
                        rmin=rmin,
                        rmax=rmax_lmc,
                        gridSizeR=gridsizeR,
                    )

                    fname_lmc = (
                        f"{nsnap_str}.LMC.{sym_label}_{l_order}.coef_mult"
                    )
                    outpath_lmc = os.path.join(OUTPUT_PATH, fname_lmc)
                    p_LMC_sph.export(outpath_lmc)

                    if verbose:
                        print(f"    Saved → {outpath_lmc}")

        if verbose:
            print("-" * 72)

    # ------------------------------------------------------------------
    # 3. Done
    # ------------------------------------------------------------------
    if verbose:
        print(
            f"\nFinished fitting potentials for snapshot {nsnap_str}. "
            f"All files saved to {OUTPUT_PATH}"
        )

    return None

# ---------------------------------------------------------------------------
# Construct HDF5 routing:
# ---------------------------------------------------------------------------

# ============================================================
# HDF5 WRITE
# ============================================================

def write_coef_to_h5(
    h5_path: Union[str, Path],
    coef_string: str,
    group_name: str = "snap_100",
    dataset_name: str = "coefs",
    overwrite: bool = False,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """
    Store an Agama coefficient text string in an HDF5 archive.

    The string is written as a single UTF-8 scalar dataset inside
    ``<group_name>/<dataset_name>``.  The archive is created if it does not
    already exist; existing archives are opened in append mode so that other
    groups are preserved.

    Parameters
    ----------
    h5_path : str or Path
        Destination HDF5 file.  Created if absent, appended to otherwise.
    coef_string : str
        Full text content of the Agama ``.coef_*`` file.
    group_name : str, optional
        HDF5 group to write into, by default ``"snap_100"``.
    dataset_name : str, optional
        Name of the scalar string dataset, by default ``"coefs"``.
    overwrite : bool, optional
        If ``True``, silently replace an existing dataset.
        If ``False`` (default) and the dataset already exists, raise
        ``RuntimeError``.
    metadata : dict, optional
        Key-value pairs attached as HDF5 *group* attributes, e.g.
        ``{"lmax": 8, "snap": 100}``.  Values must be HDF5-serialisable
        (str, int, float, array, …).

    Raises
    ------
    RuntimeError
        If ``overwrite=False`` and ``group_name/dataset_name`` already exists.
    """

    h5_path = Path(h5_path)
    dt = h5py.string_dtype(encoding="utf-8")

    mode = "a" if h5_path.exists() else "w"

    with h5py.File(h5_path, mode) as f:

        grp = f.require_group(group_name)

        if dataset_name in grp:
            if overwrite:
                del grp[dataset_name]
            else:
                raise RuntimeError(f"{group_name}/{dataset_name} exists")

        grp.create_dataset(dataset_name, data=coef_string, dtype=dt)

        if metadata:
            for k, v in metadata.items():
                grp.attrs[k] = v

def write_snapshot_coefs_to_h5(
    snapshot_ids: Sequence[int],
    coef_file_patterns: Sequence[str],
    h5_output_paths: Sequence[Union[str, Path]],
    group_fmt: str = "snap_{snap:03d}",
    dataset_name: str = "coefs",
    overwrite: bool = True,
    encoding: str = "utf-8",
) -> None:
    """
    Batch-write a collection of Agama coefficient files into one or more HDF5
    archives.

    For each snapshot index in *snapshot_ids* and for each
    ``(file_pattern, h5_path)`` pair in ``zip(coef_file_patterns,
    h5_output_paths)``, this function:

    1. Formats ``file_pattern % {"snap": snap_id}`` (or uses
       ``file_pattern.format(snap=snap_id)``) to obtain the source path.
    2. Reads the text content of that file.
    3. Calls :func:`write_coef_to_h5` to store the content under the group
       name produced by ``group_fmt.format(snap=snap_id)``.

    This is a convenience wrapper around the inner loop that is otherwise
    repeated for each potential type (spherical multipole, cylindrical
    spline, …).

    Parameters
    ----------
    snapshot_ids : sequence of int
        Ordered list of snapshot indices, e.g. ``range(0, 101)``.
    coef_file_patterns : sequence of str
        One pattern per output HDF5 file.  Each pattern is formatted with
        ``str.format(snap=snap_id)`` where ``snap_id`` is zero-padded to three
        digits, e.g.
        ``"potential/{snap:03d}.MW.none_8.coef_mult"``.
    h5_output_paths : sequence of str or Path
        One HDF5 output path per entry in *coef_file_patterns*.  Must have the
        same length.
    group_fmt : str, optional
        ``str.format``-style template for the HDF5 group name.  Receives the
        keyword ``snap=snap_id``, by default ``"snap_{snap:03d}"`` which
        produces ``"snap_000"``, ``"snap_001"``, …
    dataset_name : str, optional
        Name of the scalar dataset inside each group, by default ``"coefs"``.
    overwrite : bool, optional
        Forwarded to :func:`write_coef_to_h5`.  Defaults to ``True`` so that
        repeated runs overwrite previous data without errors.
    encoding : str, optional
        File encoding used when reading the source ``.coef_*`` files, by
        default ``"utf-8"``.

    Raises
    ------
    ValueError
        If *coef_file_patterns* and *h5_output_paths* have different lengths.
    FileNotFoundError
        If a formatted coefficient file path does not exist on disk.

    Examples
    --------
    Replicate the manual loop for three potential types:

    >>> write_snapshot_coefs_to_h5(
    ...     snapshot_ids=range(0, 101),
    ...     coef_file_patterns=[
    ...         "potential/{snap:03d}.MW.none_8.coef_mult",
    ...         "potential/{snap:03d}.MW.none_8.coef_cylsp",
    ...         "potential/{snap:03d}.LMC.none_8.coef_mult",
    ...     ],
    ...     h5_output_paths=[
    ...         "data/MW.none_8.coef_mult.h5",
    ...         "data/MW.none_8.coef_cylsp.h5",
    ...         "data/LMC.none_8.coef_mult.h5",
    ...     ],
    ... )
    """
    if len(coef_file_patterns) != len(h5_output_paths):
        raise ValueError(
            f"coef_file_patterns (len={len(coef_file_patterns)}) and "
            f"h5_output_paths (len={len(h5_output_paths)}) must have the same length."
        )

    for snap_id in snapshot_ids:
        group_name = group_fmt.format(snap=snap_id)
        for file_pattern, out_path in zip(coef_file_patterns, h5_output_paths):
            file_path = Path(file_pattern.format(snap=snap_id))
            if not file_path.exists():
                raise FileNotFoundError(
                    f"Coefficient file not found: {file_path} (snap={snap_id})"
                )
            coef_string = file_path.read_text(encoding=encoding)
            write_coef_to_h5(
                out_path,
                coef_string,
                group_name=group_name,
                dataset_name=dataset_name,
                overwrite=overwrite,
            )

# ============================================================
# HDF5 READ
# ============================================================

def load_coef_from_h5(
    h5_path: Union[str, Path],
    group_name: str = "snap_100",
    dataset_name: str = "coefs",
) -> str:
    """
    Load a coefficient text string from an HDF5 archive.

    Parameters
    ----------
    h5_path : str or Path
        Path to the HDF5 file.
    group_name : str, optional
        HDF5 group containing the dataset, by default ``"snap_100"``.
    dataset_name : str, optional
        Name of the scalar string dataset to read, by default ``"coefs"``.

    Returns
    -------
    str
        The full UTF-8 coefficient text, ready to be written to a temp file
        and passed to ``agama.Potential(file=...)``.

    Raises
    ------
    KeyError
        If *group_name* or *dataset_name* do not exist in the archive.
    """
    with h5py.File(h5_path, "r") as f:
        raw = f[group_name][dataset_name][()]
        if isinstance(raw, bytes):
            return raw.decode("utf-8")
        return str(raw)

# ============================================================
# RAM MATERIALIZATION
# ============================================================

def _get_fast_tmp_dir() -> str:
    """
    Return the best available temporary directory, preferring RAM-backed
    storage when possible.

    Preference order:

    1. ``/dev/shm``  – RAM-backed tmpfs on most Linux systems (including WSL2).
    2. System temp dir (``tempfile.gettempdir()``) – works everywhere and
       already points to fast storage on many platforms (``/tmp`` is often
       tmpfs on modern Linux; macOS and Windows use SSD-backed dirs).

    The directory is only selected if it exists **and** is writable, so
    restricted HPC nodes or permission-locked mounts are skipped silently.
    """
    shm = Path("/dev/shm")
    if shm.is_dir() and os.access(shm, os.W_OK):
        return str(shm)
    return tempfile.gettempdir()


def materialize_string_to_ram(coef_string: str) -> str:
    """
    Write a coefficient string to a fast temporary file and return its path.

    Prefers RAM-backed storage (``/dev/shm``) when available and writable,
    otherwise falls back to the platform's default temp directory.  Works
    on Linux, macOS, Windows, WSL2, and HPC clusters regardless of
    ``/dev/shm`` availability or permissions.

    Callers should still call :func:`cleanup_ram_file` when done.

    Parameters
    ----------
    coef_string : str
        Text content to materialise.

    Returns
    -------
    str
        Absolute path of the newly created file, e.g.
        ``"/dev/shm/agama_3f2a…hex.coef"`` or
        ``"/tmp/agama_3f2a…hex.coef"``.
    """
    tmp_dir = _get_fast_tmp_dir()
    filename = Path(tmp_dir) / f"agama_{uuid.uuid4().hex}.coef"
    filename.write_text(coef_string, encoding="utf-8")
    return str(filename)

def cleanup_ram_file(path: Union[str, Path]) -> None:
    """
    Remove a previously materialised temporary file, ignoring missing-file errors.

    Parameters
    ----------
    path : str or Path
        Path to the file to remove (typically under ``/dev/shm`` or the
        system temp directory).  Silently does nothing if the file no
        longer exists.
    """
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


# ============================================================
# LOAD SINGLE POTENTIAL
# ============================================================

def load_agama_potential_from_h5(
    agama: Any,
    h5_path: Union[str, Path],
    group_name: str = "snap_100",
    center: Optional[Union[str, Sequence[float], np.ndarray]] = None,
) -> Any:
    """
    Reconstruct a single Agama potential from a coefficient string stored in
    an HDF5 archive.

    The coefficient text is temporarily written to ``/dev/shm``, passed to
    ``agama.Potential(file=...)``, and the temp file is removed immediately
    afterwards.

    Parameters
    ----------
    agama : module
        The imported ``agama`` module.
    h5_path : str or Path
        Path to the HDF5 archive produced by :func:`write_coef_to_h5`.
    group_name : str, optional
        HDF5 group to read from, by default ``"snap_100"``.
    center : None, sequence of float, or ndarray, optional
        If provided, forwarded as the ``center`` keyword to
        ``agama.Potential``.  Must be a 3-element sequence ``[x, y, z]``.
        Pass ``None`` (default) to omit the keyword entirely.

    Returns
    -------
    agama.Potential
        The reconstructed Agama potential object.

    Raises
    ------
    AssertionError
        If *center* is provided but does not have exactly 3 elements.
    RuntimeError
        If no writable temporary directory is found (propagated from
        :func:`materialize_string_to_ram`).
    """
    assert center is None or len(center) == 3, (
        "center must be None or a 3-element sequence [x, y, z]."
    )

    coef_string = load_coef_from_h5(h5_path, group_name)
    path = materialize_string_to_ram(coef_string)

    try:
        if center is None:
            pot = agama.Potential(file=path)
        else:
            pot = agama.Potential(file=path, center=center)
    finally:
        cleanup_ram_file(path)

    return pot

# ============================================================
# LOAD EVOLVING POTENTIAL  (private helpers)
# ============================================================

def _extract_int_from_group(name: str) -> int:
    """
    Extract the first integer embedded in a group name string.

    Used as a sort key so that ``snap_000``, ``snap_001``, … are ordered
    numerically rather than lexicographically.

    Parameters
    ----------
    name : str
        HDF5 group name, e.g. ``"snap_042"`` or ``"snapshot_100"``.

    Returns
    -------
    int
        The first integer found in *name*, or a deterministic hash-based
        fallback if no integer is present.
    """
    m = re.search(r"(\d+)", name)
    if m:
        return int(m.group(1))
    return abs(hash(name)) % (10 ** 9)

def _materialize_to_shm_and_keep(coef_string: str) -> str:
    """
    Write *coef_string* to a fast temporary file and return the path
    **without** cleaning up.

    Unlike :func:`materialize_string_to_ram` this variant is intentionally
    non-ephemeral: the caller is responsible for later deletion (e.g. via a
    ``cleanup`` closure).  This is necessary when building evolving potentials
    where the files must persist until ``agama.Potential`` has finished
    reading them.

    Parameters
    ----------
    coef_string : str
        Text to write.

    Returns
    -------
    str
        Absolute path of the created temporary file.
    """
    tmp_dir = _get_fast_tmp_dir()
    filename = Path(tmp_dir) / f"agama_{uuid.uuid4().hex}.coef"
    filename.write_text(coef_string, encoding="utf-8")
    return str(filename)

def _load_coef_string_from_h5(
    h5_path: Union[str, Path],
    group_name: str,
    dataset_name: str = "coefs",
) -> str:
    """
    Load a coefficient string from a specific group in an HDF5 archive.

    A thin wrapper around :func:`load_coef_from_h5` for internal use.

    Parameters
    ----------
    h5_path : str or Path
        Path to the HDF5 archive.
    group_name : str
        Group to read.
    dataset_name : str, optional
        Dataset name within the group, by default ``"coefs"``.

    Returns
    -------
    str
        Decoded UTF-8 coefficient string.
    """
    with h5py.File(h5_path, "r") as f:
        raw = f[group_name][dataset_name][()]
        if isinstance(raw, bytes):
            return raw.decode("utf-8")
        return str(raw)


# ============================================================
# LOAD EVOLVING POTENTIAL  (public)
# ============================================================

def load_agama_evolving_potential_from_h5(
    agama: Any,
    h5_path: Union[str, Path],
    times: Sequence[float],
    *,
    group_names: Optional[Sequence[str]] = None,
    dataset_name: str = "coefs",
    center: Optional[Union[Sequence[float], np.ndarray]] = None,
    interpLinear: bool = True,
) -> Any:
    """
    Build a time-evolving Agama potential from an HDF5 archive.

    Each group in the archive corresponds to one snapshot.  The coefficient
    strings are materialised to ``/dev/shm``, an Agama ``Evolving`` config
    file is assembled, and the potential is created — after which all
    temporary files are removed.

    Parameters
    ----------
    agama : module
        The imported ``agama`` module.
    h5_path : str or Path
        HDF5 archive whose groups each contain a ``"coefs"`` dataset
        (or *dataset_name*) with a coefficient text string.
    times : sequence of float
        Simulation times corresponding, in order, to the groups that will be
        loaded.  Must have the same length as the number of groups used.
    group_names : sequence of str, optional
        Explicit list of HDF5 group names to use (and the order to use them).
        If ``None`` (default), all groups in the archive are used and sorted
        numerically by the integer embedded in their name (e.g. ``snap_042``
        → 42).
    dataset_name : str, optional
        Name of the scalar string dataset within each group, by default
        ``"coefs"``.
    center : sequence of float or ndarray, optional
        If provided, forwarded as the ``center`` keyword to
        ``agama.Potential``.  Agama accepts per-snapshot centers for evolving
        potentials.  Pass ``None`` (default) to omit.
    interpLinear : bool, optional
        Selects linear (``True``, default) vs. cubic-spline interpolation
        between snapshots for the evolving potential.

    Returns
    -------
    agama.Potential
        The time-evolving Agama potential object.

    Raises
    ------
    ValueError
        If the number of resolved group names differs from ``len(times)``.
    RuntimeError
        If ``/dev/shm`` is unavailable.

    Notes
    -----
    All ``.coef`` and config files written to ``/dev/shm`` are deleted once
    the potential has been constructed, even if construction fails.

    Examples
    --------
    >>> import agama
    >>> times = np.linspace(0, 10, 101)   # Gyr, one per snapshot
    >>> pot = load_agama_evolving_potential_from_h5(
    ...     agama,
    ...     "data/MW.none_8.coef_mult.h5",
    ...     times=times,
    ... )
    """

    h5_path = Path(h5_path)

    with h5py.File(h5_path, "r") as f:
        all_groups = list(f.keys())

    if group_names is None:
        group_names = sorted(all_groups, key=_extract_int_from_group)
    else:
        group_names = list(group_names)

    if len(group_names) != len(times):
        raise ValueError(
            f"len(group_names)={len(group_names)} does not match "
            f"len(times)={len(times)}."
        )

    ram_paths: list[str] = []
    config_path: Optional[str] = None

    try:
        for grp in group_names:
            coef_string = _load_coef_string_from_h5(
                h5_path, grp, dataset_name=dataset_name
            )
            path = _materialize_to_shm_and_keep(coef_string)
            ram_paths.append(path)

        timestamp_lines = [
            f"{t} {p}" for t, p in zip(times, ram_paths)
        ]
        config = (
            "[Potential]\n"
            "type = Evolving\n"
            f"interpLinear = {bool(interpLinear)}\n"
            "Timestamps\n"
            + "\n".join(timestamp_lines)
        )

        config_path = _materialize_to_shm_and_keep(config)

        if center is None:
            pot = agama.Potential(file=config_path)
        else:
            pot = agama.Potential(file=config_path, center=center)

    except Exception:
        for p in ram_paths:
            try:
                os.unlink(p)
            except Exception:
                pass
        raise

    finally:
        # always clean up — whether success or not
        for p in ram_paths:
            try:
                os.unlink(p)
            except Exception:
                pass
        if config_path is not None:
            try:
                os.unlink(config_path)
            except Exception:
                pass

    return pot