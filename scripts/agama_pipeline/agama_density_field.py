"""
Compute density fields on a regular 3-D grid directly from AGAMA
basis-function-expansion coefficients (``agama.Potential``), rather than
from raw particle data -- the AGAMA-side counterpart to
``particle_fields/density_field.py``'s particle-KDE approach. Evaluated on
the *same* per-component grid (``density_field.DEFAULT_GRID_PARAMS`` via the
shared ``normalize_grid_range`` helper) so the two can be compared
point-for-point.

Loads the per-snapshot coefficient files ``build_agama_expansion.py`` writes
under ``AGAMA_EXPANSIONS_PATH/suite_<sim_id>/`` (e.g. "000.MW.none_6.coef_mult",
"000.MW.none_6.coef_bulge", "000.MW.none_6.coef_cylsp",
"000.LMC.none_6.coef_mult"), the same way the prototype notebook
(agama_pipeline/XMC_Agama_BFE.ipynb) does, via ``agama.Potential(file=...)``.

AGAMA's BFE fit produces 4 potentials per snapshot: MWhalo ("coef_mult",
MWhalo/MWhaloiso particles only), MWbulge ("coef_bulge", its own tighter
grid), MWdisk ("coef_cylsp"), and LMChalo ("coef_mult"). Prior to
``agama_BFEs.fitAgamaBFE`` splitting the bulge into its own expansion, the
bulge was absorbed into the MW "coef_mult" component alongside the halo --
this module now matches density_field.py's 4-component (bulge/disk/halo/LMC)
split from raw particle data.

Author: github.com/jngaravitoc
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence, Union

import agama
import h5py
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR))  # agama_pipeline/ -- for build_agama_expansion
sys.path.append(str(THIS_DIR.parent))  # scripts/ -- for config
sys.path.append(str(THIS_DIR.parent / "particle_fields"))  # for density_field

from config import AGAMA_EXPANSIONS_PATH
from build_agama_expansion import resolve_suite_layout  # noqa: F401 -- re-exported
from density_field import DEFAULT_GRID_PARAMS, load_centers, normalize_grid_range  # noqa: F401

__all__ = [
    "AGAMA_COMPONENTS",
    "resolve_suite_layout",
    "load_centers",
    "load_agama_potentials",
    "compute_agama_grid_density",
    "write_agama_density_field",
    "read_agama_density_field",
    "compute_and_write_agama_density_field",
    "discover_agama_snapshots",
    "run_agama_density_fields",
]

AGAMA_COMPONENTS = ["MWhalo", "MWbulge", "MWdisk", "LMChalo"]

# {component: coefficient-file suffix, following build_agama_expansion.py's
# "{snap:03d}.{suffix}" naming}
_COEF_SUFFIX = {
    "MWhalo": "MW.none_6.coef_mult",
    "MWbulge": "MW.none_6.coef_bulge",
    "MWdisk": "MW.none_6.coef_cylsp",
    "LMChalo": "LMC.none_6.coef_mult",
}


def load_agama_potentials(
    sim_id: int,
    nsnap: int,
    lmc_center: np.ndarray,
    coef_dir: Optional[Union[str, Path]] = None,
) -> dict:
    """
    Load the 4 per-snapshot AGAMA potentials (MWhalo, MWbulge, MWdisk,
    LMChalo) that ``build_agama_expansion.py`` fit for one
    simulation/snapshot.

    Parameters
    ----------
    sim_id : int
    nsnap : int
        Snapshot index (matches the coefficient files' zero-padded prefix).
    lmc_center : ndarray, shape (3,)
        LMC center-of-mass position at this snapshot (e.g.
        ``density_field.load_centers(...)["lmc_center"][nsnap]``) -- the LMC
        potential is defined relative to its own center, so it must be
        recentered onto the same MW-centered coordinate frame the other two
        potentials (and the particle-KDE fields) use.
    coef_dir : str or Path, optional
        Directory containing the coefficient files. Defaults to
        ``config.AGAMA_EXPANSIONS_PATH / f"suite_{sim_id}"``.

    Returns
    -------
    dict
        ``{"MWhalo": agama.Potential, "MWbulge": agama.Potential,
        "MWdisk": agama.Potential, "LMChalo": agama.Potential}``.
    """
    coef_dir = Path(coef_dir) if coef_dir is not None else AGAMA_EXPANSIONS_PATH / f"suite_{sim_id}"

    potentials = {}
    for component, suffix in _COEF_SUFFIX.items():
        coef_file = coef_dir / f"{nsnap:03d}.{suffix}"
        if component == "LMChalo":
            potentials[component] = agama.Potential(file=str(coef_file), center=lmc_center)
        else:
            potentials[component] = agama.Potential(file=str(coef_file))
    return potentials


def compute_agama_grid_density(
    potential,
    grid_range,
    grid_bins: int,
    center: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate one AGAMA potential's density on a regular 3-D grid -- the same
    grid convention (coordinates, axis order, reshape) as
    ``density_field.compute_grid_density`` via the shared
    ``normalize_grid_range`` helper, so results are directly comparable
    point-for-point against the particle-KDE fields.

    Unlike the KDE approach, this is an exact evaluation of the fitted
    basis-function expansion -- there is no bandwidth/method choice here.

    Parameters
    ----------
    potential : agama.Potential
    grid_range : (float, float) or 3x (float, float)
        Same convention as ``density_field.compute_grid_density``: a single
        ``(min, max)`` pair applies to all three axes, or three separate
        pairs give independent ``(x, y, z)`` extents.
    grid_bins : int
        Number of grid points per axis.
    center : ndarray, shape (3,), optional
        Physical offset (kpc), added to the query points before evaluating
        ``potential.density(...)``, but *not* to the returned
        ``x_edges``/``y_edges``/``z_edges``. Needed for a potential loaded
        with ``agama.Potential(file=..., center=...)`` (e.g. LMChalo, see
        ``load_agama_potentials``): that ``center`` places the fitted
        expansion at that location in a shared Galactocentric frame, so the
        query grid must be offset into that same frame to actually land on
        the component -- confirmed on Model_13: the LMC sits 72-294 kpc
        from the origin depending on snapshot, so an unshifted grid (e.g.
        density_field.DEFAULT_GRID_PARAMS["LMChalo"]'s (-50,50) kpc range,
        which is only valid in the KDE pipeline's LMC-*recentered* particle
        frame) samples nowhere near the LMC's actual mass. Leaving this
        ``None`` (MWhalo/MWdisk, already MW-/Galactocentric-centered) keeps
        the returned coordinates and the query points identical, as before.

    Returns
    -------
    density : ndarray, shape (grid_bins, grid_bins, grid_bins)
        Msun/kpc^3.
    x_edges, y_edges, z_edges : ndarray, shape (grid_bins,)
        Relative to ``center`` when given (e.g. "kpc from the LMC"), the
        same convention density_field.py's KDE grids already use.
    """
    (x_range, y_range, z_range) = normalize_grid_range(grid_range)
    x_edges = np.linspace(x_range[0], x_range[1], grid_bins)
    y_edges = np.linspace(y_range[0], y_range[1], grid_bins)
    z_edges = np.linspace(z_range[0], z_range[1], grid_bins)

    xx, yy, zz = np.meshgrid(x_edges, y_edges, z_edges, indexing="ij")
    xyz = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    if center is not None:
        xyz = xyz + np.asarray(center)

    density = potential.density(xyz).reshape(grid_bins, grid_bins, grid_bins)

    return density, x_edges, y_edges, z_edges


def write_agama_density_field(
    filename,
    sim_id: int,
    simulation_suite: str,
    simulation_model: str,
    component: str,
    snapshot_label: str,
    density: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    z_edges: np.ndarray,
    grid_range,
    grid_bins: int,
    snapshot_time: Optional[float] = None,
    overwrite: bool = True,
) -> None:
    """
    Write one component/snapshot's AGAMA-BFE density field to a per-halo
    HDF5 file, mirroring ``density_field.write_density_field``'s group/attrs
    layout (``/{component}/{snapshot_label}/{density,x_edges,y_edges,z_edges}``)
    but with ``source="agama_bfe"`` instead of a KDE method/bandwidth, since
    neither applies to an exact basis-function evaluation.
    """
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if filename.exists() else "w"
    with h5py.File(filename, mode) as f:
        f.attrs["sim_id"] = sim_id
        f.attrs["simulation_suite"] = simulation_suite
        f.attrs["simulation_model"] = simulation_model

        grp = f.require_group(f"{component}/{snapshot_label}")
        datasets = {
            "density": density,
            "x_edges": x_edges,
            "y_edges": y_edges,
            "z_edges": z_edges,
        }
        for name, data in datasets.items():
            if name in grp:
                if overwrite:
                    del grp[name]
                else:
                    raise RuntimeError(f"{grp.name}/{name} exists")
            grp.create_dataset(name, data=data, compression="gzip")

        grp.attrs["grid_range"] = grid_range
        grp.attrs["grid_bins"] = grid_bins
        grp.attrs["source"] = "agama_bfe"
        grp.attrs["coef_suffix"] = _COEF_SUFFIX[component]
        if snapshot_time is not None:
            grp.attrs["snapshot_time"] = snapshot_time


def read_agama_density_field(filename, component: str, snapshot_label: str) -> dict:
    """Reader counterpart of :func:`write_agama_density_field`."""
    with h5py.File(filename, "r") as f:
        grp = f[f"{component}/{snapshot_label}"]
        result = {
            "density": np.array(grp["density"]),
            "x_edges": np.array(grp["x_edges"]),
            "y_edges": np.array(grp["y_edges"]),
            "z_edges": np.array(grp["z_edges"]),
        }
        result.update({key: grp.attrs[key] for key in grp.attrs})
        result.update({key: f.attrs[key] for key in f.attrs})
    return result


def compute_and_write_agama_density_field(
    sim_id: int,
    simulation_suite: str,
    simulation_model: str,
    component: str,
    nsnap: int,
    centers: dict,
    coef_dir=None,
    grid_params: Optional[dict] = None,
    output_path=None,
    overwrite: bool = True,
) -> dict:
    """
    Load one snapshot's AGAMA potentials, evaluate one component's density
    on its density_field.DEFAULT_GRID_PARAMS-matching grid, and write it to
    the per-halo HDF5 file -- the AGAMA-side analog of
    ``density_field.compute_and_write_density_field``.

    Parameters
    ----------
    sim_id : int
    simulation_suite, simulation_model : str
        config.py identity keys (e.g. "sheng", "sheng_iso").
    component : str
        One of :data:`AGAMA_COMPONENTS` ("MWhalo", "MWbulge", "MWdisk",
        "LMChalo").
    nsnap : int
        Snapshot index.
    centers : dict
        Output of ``density_field.load_centers`` (``time``, ``lmc_center``).
    coef_dir : str or Path, optional
        Forwarded to :func:`load_agama_potentials`.
    grid_params : dict, optional
        Per-component ``{"range": ..., "bins": ...}``. Defaults to
        ``density_field.DEFAULT_GRID_PARAMS`` (only ``range``/``bins`` are
        used; any ``method``/``bw_method`` entries are ignored -- they're
        KDE-specific and don't apply to an exact BFE evaluation).
    output_path : str or Path, optional
        Destination HDF5 file. Defaults to
        ``config.AGAMA_EXPANSIONS_PATH / f"agama_density_{sim_id:04d}.h5"``.
    overwrite : bool, optional

    Returns
    -------
    dict
        ``density``, ``x_edges``, ``y_edges``, ``z_edges``.
    """
    potentials = load_agama_potentials(
        sim_id, nsnap, centers["lmc_center"][nsnap], coef_dir=coef_dir,
    )

    # LMChalo's potential was loaded with center=lmc_center (Galactocentric
    # frame), so the query grid must be offset there too -- see
    # compute_agama_grid_density's `center` docstring. MWhalo/MWdisk are
    # already Galactocentric, no offset needed.
    center = centers["lmc_center"][nsnap] if component == "LMChalo" else None

    params = (grid_params or DEFAULT_GRID_PARAMS)[component]
    density, x_edges, y_edges, z_edges = compute_agama_grid_density(
        potentials[component], params["range"], params["bins"], center=center,
    )

    if output_path is None:
        output_path = AGAMA_EXPANSIONS_PATH / f"agama_density_{sim_id:04d}.h5"

    snapshot_time = float(centers["time"][nsnap]) if "time" in centers else None
    write_agama_density_field(
        output_path, sim_id, simulation_suite, simulation_model,
        component, f"snapshot_{nsnap:03d}",
        density, x_edges, y_edges, z_edges,
        params["range"], params["bins"],
        snapshot_time=snapshot_time, overwrite=overwrite,
    )

    return {"density": density, "x_edges": x_edges, "y_edges": y_edges, "z_edges": z_edges}


def discover_agama_snapshots(coef_dir) -> list[int]:
    """
    Find which snapshot indices have AGAMA coefficient files in ``coef_dir``.

    Unlike raw particle snapshots ("snapshot_NNN.hdf5", handled by
    ``compute_bfe_helpers.get_snapshot_suffixes``), AGAMA coefficient files
    are named number-first ("NNN.MW.none_6.coef_mult"), so that helper
    doesn't apply here -- this globs on the MWhalo suffix instead, since
    that component is written for every snapshot ``build_agama_expansion.py``
    fit.

    Parameters
    ----------
    coef_dir : str or Path

    Returns
    -------
    list of int
        Sorted snapshot indices.
    """
    coef_dir = Path(coef_dir)
    suffix = _COEF_SUFFIX["MWhalo"]
    return sorted(int(f.name.split(".")[0]) for f in coef_dir.glob(f"*.{suffix}"))


def run_agama_density_fields(
    sim_id: int,
    simulation_suite: str,
    simulation_model: str,
    components: Sequence[str] = AGAMA_COMPONENTS,
    snapshots: Optional[Sequence[int]] = None,
    coef_dir: Optional[Union[str, Path]] = None,
    grid_params: Optional[dict] = None,
    output_path=None,
) -> Path:
    """
    Compute and write AGAMA-BFE density fields for every (component,
    snapshot) pair of one simulation, into a single per-halo HDF5 file --
    the AGAMA-side analog of
    ``particle_fields/compute_density_fields.py``'s ``run_density_fields``.

    Parameters
    ----------
    sim_id : int
    simulation_suite, simulation_model : str
        config.py identity keys (e.g. "sheng", "sheng_iso").
    components : sequence of str, optional
        Defaults to all 4 (:data:`AGAMA_COMPONENTS`).
    snapshots : sequence of int, optional
        Defaults to every snapshot found via :func:`discover_agama_snapshots`
        for this sim's coefficient directory.
    coef_dir : str or Path, optional
        Forwarded to :func:`load_agama_potentials`. Defaults to
        ``config.AGAMA_EXPANSIONS_PATH / f"suite_{sim_id}"``.
    grid_params : dict, optional
        Forwarded to :func:`compute_and_write_agama_density_field`.
    output_path : str or Path, optional
        Destination HDF5 file. Defaults to
        ``config.AGAMA_EXPANSIONS_PATH / f"agama_density_{sim_id:04d}.h5"``.

    Returns
    -------
    Path
        The output file that was written to.
    """
    coef_dir = Path(coef_dir) if coef_dir is not None else AGAMA_EXPANSIONS_PATH / f"suite_{sim_id}"
    snapshots = snapshots if snapshots is not None else discover_agama_snapshots(coef_dir)

    centers = load_centers(simulation_suite, simulation_model, sim_id)

    output_path = (
        Path(output_path) if output_path is not None
        else AGAMA_EXPANSIONS_PATH / f"agama_density_{sim_id:04d}.h5"
    )

    for nsnap in snapshots:
        for component in components:
            compute_and_write_agama_density_field(
                sim_id, simulation_suite, simulation_model, component, nsnap,
                centers, coef_dir=coef_dir, grid_params=grid_params, output_path=output_path,
            )

    return output_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute AGAMA-BFE density fields (all components, all snapshots) "
        "for one XMC-Atlas simulation."
    )
    parser.add_argument(
        "simulation_suite", type=str, choices=["sheng"],
        help="Simulation suite key from config.py. Only 'sheng' is currently "
        "supported here, matching particle_fields/compute_density_fields.py.",
    )
    parser.add_argument(
        "simulation_model", type=str,
        help="Model key from config.py for the selected suite (e.g. sheng_iso).",
    )
    parser.add_argument("sim_id", type=int, help="Simulation ID")
    parser.add_argument(
        "--components", type=str, nargs="+", default=AGAMA_COMPONENTS,
        help=f"Components to compute (default: all of {AGAMA_COMPONENTS}).",
    )
    parser.add_argument(
        "--snapshots", type=int, nargs="+", default=None,
        help="Snapshot indices to compute (default: every snapshot found in the "
        "coefficient directory).",
    )
    parser.add_argument(
        "--coef-dir", type=str, default=None,
        help="Directory containing the coefficient files (default: "
        "config.AGAMA_EXPANSIONS_PATH/suite_<sim_id>).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output HDF5 path (default: config.AGAMA_EXPANSIONS_PATH/agama_density_<sim_id>.h5).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    output_path = run_agama_density_fields(
        args.sim_id, args.simulation_suite, args.simulation_model,
        components=args.components,
        snapshots=args.snapshots,
        coef_dir=args.coef_dir,
        output_path=args.output,
    )
    print(f"Wrote AGAMA density fields for sim_id={args.sim_id} to {output_path}")


if __name__ == "__main__":
    main()
