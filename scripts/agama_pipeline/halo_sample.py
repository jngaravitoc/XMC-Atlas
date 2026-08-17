"""
Sample dark-matter-halo particles from XMC-Atlas simulations, extract their
orbits across all snapshots, derive orbital parameters (energy, angular
momentum, pericenter/apocenter passages), and write results to disk.

Author: github.com/jngaravitoc

Usage: python halo_sample.py sim_id --n-sample 50

TODO:
    [] Peri/apocenter detection (find_peri_apo) is a placeholder -- discrete
       sign-change detection on the snapshot grid -- pending a finalized
       definition.
    [] Energy/angular-momentum time sampling defaults to "every snapshot";
       no downsampling logic yet.
    [] Batch/Slurm driver across all Sheng24 sim IDs deferred to a follow-up
       (see compute_agama_exp.sh for the pattern).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import h5py
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR.parent / "exp_pipeline"))

from ios_nbody_sims import load_particle_data
from compute_bfe_helpers import load_sheng24_exp_center, get_snapshot_suffixes
from sanity_checks import check_monotonic_contiguous_snapshots

__all__ = [
    "load_sim_context",
    "select_particle_sample",
    "extract_orbit_sample",
    "find_peri_apo",
    "compute_orbit_parameters",
    "write_orbit_trajectories",
    "write_orbit_parameters",
    "run_halo_sample",
]


def _energy(vel: np.ndarray, pot: np.ndarray) -> np.ndarray:
    """Specific orbital energy E = 0.5*v^2 + pot; broadcasts over any leading shape."""
    return 0.5 * np.sum(vel**2, axis=-1) + pot


def _ang_mom(pos: np.ndarray, vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Specific angular momentum vector L = r x v and its magnitude |L|."""
    L = np.cross(pos, vel)
    Lmag = np.linalg.norm(L, axis=-1)
    return L, Lmag


def load_sim_context(
    sim_id: int,
    snapshot_dir: str,
    snapname: str,
    centers_path: str,
    centers_file: str,
    suite: str = "Sheng24",
) -> dict:
    """
    Load the snapshot inventory and MW center-of-mass trajectory for one simulation.

    Parameters
    ----------
    sim_id : int
        Simulation identifier (row key in the centers file).
    snapshot_dir : str
        Directory containing the simulation's snapshot files.
    snapname : str
        Snapshot filename prefix (e.g. ``"snapshot"``).
    centers_path : str
        Directory containing the MW/LMC center-of-mass trajectory file.
    centers_file : str
        Filename of the center-of-mass trajectory file (e.g.
        ``"MW_LMC_orbits_iso.txt"``).
    suite : str, optional
        Simulation suite name, by default ``"Sheng24"`` (currently the only
        suite implemented).

    Returns
    -------
    dict
        ``times`` (T,), ``mw_center``/``mw_vel_center`` (T, 3), and
        ``snap_suffixes`` -- a list of zero-padded snapshot suffixes (e.g.
        ``"000"``), in time order.

    Raises
    ------
    ValueError
        If the snapshot suffixes on disk are not contiguous, or if the
        number of snapshots found does not match the number of times in the
        centers file.
    """
    if suite != "Sheng24":
        raise ValueError(f"suite {suite!r} not supported yet; only 'Sheng24' is implemented")

    centers = load_sheng24_exp_center(centers_path, centers_file, sim_id, return_vel=True)
    times = np.asarray(centers["time"])
    mw_center = np.asarray(centers["mw_center"])
    mw_vel_center = np.asarray(centers["mw_vel_center"])

    snap_suffixes = get_snapshot_suffixes(snapshot_dir, prefix=f"{snapname}_")
    ok, missing = check_monotonic_contiguous_snapshots(snap_suffixes)
    if not ok:
        raise ValueError(
            f"Snapshots in {snapshot_dir} are not contiguous; missing indices: {missing}"
        )
    if len(snap_suffixes) != len(times):
        raise ValueError(
            f"Found {len(snap_suffixes)} snapshots in {snapshot_dir} but the centers "
            f"file has {len(times)} times for sim_id={sim_id}"
        )

    return {
        "sim_id": sim_id,
        "suite": suite,
        "times": times,
        "mw_center": mw_center,
        "mw_vel_center": mw_vel_center,
        "snap_suffixes": snap_suffixes,
    }


def select_particle_sample(
    pid: np.ndarray,
    pos: np.ndarray,
    vel: np.ndarray,
    pot: np.ndarray,
    r_min: float = 20.0,
    r_max: float = 300.0,
    n_sample: int = 50,
    n_bins: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Select a sample of DM halo particle IDs that represents the population
    of orbits present in the halo -- including rare orbit types -- within a
    radial shell.

    Particles within ``[r_min, r_max]`` (instantaneous 3D position at the
    reference snapshot -- a phase-dependent proxy, not orbit-invariant, but
    the simplest option and the one currently used) are binned on a 2D
    ``(|L|, E)`` grid. Every occupied bin contributes at least one particle,
    so rare orbit families (e.g. near-radial or near-circular orbits) are
    guaranteed representation rather than being drowned out by however
    populous the typical orbit family is; the remaining sample budget is
    then filled with a uniform random draw from the rest of the eligible
    pool, which -- since particles are equal-mass -- reproduces the
    population's natural ``(|L|, E)`` density for the common orbit families.

    Parameters
    ----------
    pid : ndarray, shape (N,)
        Particle IDs.
    pos, vel : ndarray, shape (N, 3)
        Particle positions/velocities at the reference snapshot, recentered
        on the MW center of mass.
    pot : ndarray, shape (N,)
        Particle potential at the reference snapshot.
    r_min, r_max : float, optional
        Radial shell (kpc) eligible particles must fall within, by default
        20-300 kpc (matches the prototype notebook's cut).
    n_sample : int, optional
        Total number of particles to draw, by default 50.
    n_bins : int, optional
        Number of bins per ``(|L|, E)`` axis. Defaults to
        ``round(sqrt(max(1, n_sample // 3)))``, so the bin grid scales with
        the sample budget, leaving room to also fill in population-weighted
        draws rather than spending the whole budget on one-per-bin coverage.
    rng : numpy.random.Generator, optional
        Random number generator; a fresh one is created if not given. Pass
        an explicit, seeded generator for reproducible samples.

    Returns
    -------
    ndarray, shape (n_sample,)
        Particle IDs of the sampled particles (not positional indices).

    Raises
    ------
    ValueError
        If fewer than ``n_sample`` particles fall within the radial shell.
    """
    if rng is None:
        rng = np.random.default_rng()

    r = np.linalg.norm(pos, axis=1)
    mask = (r > r_min) & (r < r_max)
    n_eligible = int(mask.sum())

    if n_sample > n_eligible:
        raise ValueError(
            f"Requested n_sample={n_sample} but only {n_eligible} particles "
            f"fall within [{r_min}, {r_max}] kpc"
        )

    eligible_pid = pid[mask]
    energy = _energy(vel[mask], pot[mask])
    _, ang_mom = _ang_mom(pos[mask], vel[mask])

    if n_bins is None:
        n_bins = max(1, round(np.sqrt(max(1, n_sample // 3))))

    l_edges = np.linspace(ang_mom.min(), ang_mom.max(), n_bins + 1)
    e_edges = np.linspace(energy.min(), energy.max(), n_bins + 1)
    l_bin = np.clip(np.digitize(ang_mom, l_edges[1:-1]), 0, n_bins - 1)
    e_bin = np.clip(np.digitize(energy, e_edges[1:-1]), 0, n_bins - 1)
    bin_id = l_bin * n_bins + e_bin

    # Guarantee coverage: one particle from every occupied (|L|, E) bin.
    occupied_bins = np.unique(bin_id)
    if len(occupied_bins) > n_sample:
        occupied_bins = rng.choice(occupied_bins, size=n_sample, replace=False)

    core_idx = np.array(
        [rng.choice(np.flatnonzero(bin_id == b)) for b in occupied_bins], dtype=int
    )

    # Fill the remaining budget with a uniform draw over the rest of the
    # pool -- population-weighted, since bin population already reflects it.
    remaining_budget = n_sample - len(core_idx)
    if remaining_budget > 0:
        pool = np.setdiff1d(np.arange(n_eligible), core_idx)
        fill_idx = rng.choice(pool, size=remaining_budget, replace=False)
        sample_idx = np.concatenate([core_idx, fill_idx])
    else:
        sample_idx = core_idx

    return eligible_pid[sample_idx]


def extract_orbit_sample(
    snapshot_dir: str,
    snapname: str,
    snap_suffixes: Sequence[str],
    pid_sample: np.ndarray,
    mw_center: np.ndarray,
    mw_vel_center: np.ndarray,
    suite: str = "Sheng24",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract the phase-space trajectory and potential of a fixed particle
    sample across all snapshots, recentered on the MW center of mass.

    Generalizes the prototype notebook's ``extract_Sheng24_orbits``
    (PID -> ``searchsorted`` lookup per snapshot) to loop over every
    snapshot and to recenter on the MW's moving center of mass, the same
    convention used to build BFE coefficients in
    ``halos_bfe_from_same_basis.py``/``build_agama_expansion.py``
    (``data[comp]["pos"] -= exp_center[i]``).

    Parameters
    ----------
    snapshot_dir : str
        Directory containing the simulation's snapshot files.
    snapname : str
        Snapshot filename prefix.
    snap_suffixes : sequence of str
        Zero-padded snapshot suffixes, in time order (as returned by
        ``load_sim_context``).
    pid_sample : ndarray, shape (N,)
        Particle IDs to track, as returned by ``select_particle_sample``.
    mw_center, mw_vel_center : ndarray, shape (T, 3)
        MW center-of-mass position/velocity trajectory, one row per
        snapshot, aligned with ``snap_suffixes``.
    suite : str, optional
        Simulation suite name, by default ``"Sheng24"``.

    Returns
    -------
    pos, vel : ndarray, shape (N, T, 3)
        Particle positions/velocities relative to the MW center of mass.
    pot : ndarray, shape (N, T)
        Particle potential as read from the simulation (not recentered --
        it is a scalar per particle, not a vector quantity).
    """
    nsnaps = len(snap_suffixes)
    nsample = len(pid_sample)

    pos = np.zeros((nsample, nsnaps, 3))
    vel = np.zeros((nsample, nsnaps, 3))
    pot = np.zeros((nsample, nsnaps))

    for i, suffix in enumerate(snap_suffixes):
        nsnap = int(suffix)
        data = load_particle_data(
            snapshot_dir, snapname, ["MWhalo"], nsnap, suite=suite,
            quantities=["pid", "pos", "vel", "pot"],
        )["MWhalo"]

        order = np.argsort(data["pid"])
        sorted_pid = data["pid"][order]
        loc = order[np.searchsorted(sorted_pid, pid_sample)]

        assert np.array_equal(data["pid"][loc], pid_sample), (
            f"particle ID mismatch while extracting snapshot {suffix}"
        )

        pos[:, i, :] = data["pos"][loc] - mw_center[i]
        vel[:, i, :] = data["vel"][loc] - mw_vel_center[i]
        pot[:, i] = data["pot"][loc]

    return pos, vel, pot


def find_peri_apo(times: np.ndarray, r: np.ndarray) -> dict:
    """
    Locate pericenter/apocenter passages of a single particle's orbit from
    its Galactocentric distance evolution.

    NOTE: placeholder implementation, pending a finalized definition from
    the user. Detects passages as local extrema of ``r`` via sign changes of
    the discrete derivative on the snapshot grid itself -- no sub-cadence
    interpolation/refinement, so timing/radius resolution is limited to the
    simulation's snapshot cadence (``dt=0.02 Gyr`` for Sheng24). This is the
    single function to swap when the final algorithm is decided.

    Parameters
    ----------
    times : ndarray, shape (T,)
        Simulation times.
    r : ndarray, shape (T,)
        Galactocentric distance at each time.

    Returns
    -------
    dict
        ``peri_time``, ``peri_r``, ``apo_time``, ``apo_r`` -- 1D arrays, one
        entry per detected passage. Peri and apo counts (and counts between
        particles) may differ.
    """
    dr = np.diff(r)
    sign = np.sign(dr)
    # local minimum: derivative goes from negative to positive
    is_peri = np.r_[False, (sign[:-1] < 0) & (sign[1:] > 0), False]
    # local maximum: derivative goes from positive to negative
    is_apo = np.r_[False, (sign[:-1] > 0) & (sign[1:] < 0), False]

    return {
        "peri_time": times[is_peri],
        "peri_r": r[is_peri],
        "apo_time": times[is_apo],
        "apo_r": r[is_apo],
    }


def compute_orbit_parameters(
    times: np.ndarray,
    pos: np.ndarray,
    vel: np.ndarray,
    pot: np.ndarray,
) -> dict:
    """
    Derive energy, angular momentum, and pericenter/apocenter passages for a
    sample of particle orbits.

    Energy and angular momentum are computed at every snapshot (matching the
    prototype notebook's cadence); see module TODOs for the still-open
    question of downsampling to a coarser set of times.

    Parameters
    ----------
    times : ndarray, shape (T,)
        Simulation times.
    pos, vel : ndarray, shape (N, T, 3)
        Particle positions/velocities, recentered on the MW center of mass
        (as returned by ``extract_orbit_sample``).
    pot : ndarray, shape (N, T)
        Particle potential at each snapshot.

    Returns
    -------
    dict
        ``r`` (N, T) Galactocentric distance, ``energy`` (N, T),
        ``ang_mom`` (N, T, 4) with columns ``[Lx, Ly, Lz, |L|]``, and
        ``peri_apo`` -- a length-N list of ``find_peri_apo`` output dicts,
        one per particle.
    """
    r = np.linalg.norm(pos, axis=2)
    energy = _energy(vel, pot)
    L, Lmag = _ang_mom(pos, vel)
    ang_mom = np.concatenate([L, Lmag[..., None]], axis=2)

    peri_apo = [find_peri_apo(times, r[j]) for j in range(r.shape[0])]

    return {"r": r, "energy": energy, "ang_mom": ang_mom, "peri_apo": peri_apo}


def _write_datasets(grp: h5py.Group, datasets: dict, overwrite: bool) -> None:
    for name, data in datasets.items():
        if name in grp:
            if overwrite:
                del grp[name]
            else:
                raise RuntimeError(f"{grp.name}/{name} exists")
        grp.create_dataset(name, data=data, compression="gzip")


def write_orbit_trajectories(
    filename,
    sim_id: int,
    times: np.ndarray,
    pid_sample: np.ndarray,
    pos: np.ndarray,
    vel: np.ndarray,
    pot: np.ndarray,
    r_min: float,
    r_max: float,
    seed: Optional[int] = None,
    overwrite: bool = True,
) -> None:
    """
    Write particle trajectories in agama.orbit format.

    Positions and velocities are combined into a single ``(N, T, 6)``
    ``phase_space`` array with columns ``[x, y, z, vx, vy, vz]``, matching
    the on-disk/in-memory convention already used for
    ``agama.orbit(..., trajsize=T)`` trajectories elsewhere in this repo
    (see ``suites/Vasiliev23/scripts/resimulate_orbits.py``, which reshapes
    its output into exactly this ``(N, T, 6)`` shape), rather than the
    separate ``pos``/``vel`` datasets used by the prototype notebook.
    Orbital properties derived from these trajectories (energy, angular
    momentum, peri/apocenters) are written separately by
    ``write_orbit_parameters``.

    Parameters
    ----------
    filename : str or Path
        Destination HDF5 file. Created if absent, appended to otherwise.
    sim_id : int
        Simulation identifier; data is written under group
        ``suite_{sim_id:04d}``.
    times : ndarray, shape (T,)
    pid_sample : ndarray, shape (N,)
    pos, vel : ndarray, shape (N, T, 3)
        Positions/velocities recentered on the MW center of mass.
    pot : ndarray, shape (N, T)
        Kept alongside the trajectory since it is raw per-particle
        simulation output paired 1:1 with each phase-space sample, not a
        derived quantity.
    r_min, r_max, seed : optional
        Sampling parameters, stored as group attributes for provenance.
    overwrite : bool, optional
        Overwrite existing datasets in the group, by default True.
    """
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    phase_space = np.concatenate([pos, vel], axis=2)  # (N, T, 6)

    mode = "a" if filename.exists() else "w"
    with h5py.File(filename, mode) as f:
        grp = f.require_group(f"suite_{sim_id:04d}")
        _write_datasets(
            grp,
            {"pid": pid_sample, "time": times, "phase_space": phase_space, "pot": pot},
            overwrite,
        )
        grp.attrs["r_min"] = r_min
        grp.attrs["r_max"] = r_max
        grp.attrs["n_sample"] = len(pid_sample)
        if seed is not None:
            grp.attrs["seed"] = seed


def _pad_ragged(arrays: Sequence[np.ndarray]) -> np.ndarray:
    """Stack 1D arrays of different lengths into a 2D array, NaN-padded."""
    max_len = max((len(a) for a in arrays), default=0)
    out = np.full((len(arrays), max_len), np.nan)
    for i, a in enumerate(arrays):
        out[i, : len(a)] = a
    return out


def write_orbit_parameters(
    filename,
    sim_id: int,
    times: np.ndarray,
    pid_sample: np.ndarray,
    energy: np.ndarray,
    ang_mom: np.ndarray,
    peri_apo: Sequence[dict],
    r_min: float,
    r_max: float,
    seed: Optional[int] = None,
    overwrite: bool = True,
) -> None:
    """
    Write derived orbital properties (energy, angular momentum,
    pericenter/apocenter passages) to a separate HDF5 file from the
    trajectory file (see ``write_orbit_trajectories``).

    Pericenter/apocenter counts vary between particles (see the
    ``find_peri_apo`` placeholder), so each of
    ``peri_time``/``peri_r``/``apo_time``/``apo_r`` is stored as an
    ``(N, max_count)`` array, NaN-padded; ``n_peri``/``n_apo`` give the true
    per-particle passage count.

    Parameters
    ----------
    filename : str or Path
    sim_id : int
    times : ndarray, shape (T,)
    pid_sample : ndarray, shape (N,)
    energy : ndarray, shape (N, T)
    ang_mom : ndarray, shape (N, T, 4)
        Columns ``[Lx, Ly, Lz, |L|]``.
    peri_apo : sequence of dict, length N
        Per-particle output of ``find_peri_apo``.
    r_min, r_max, seed : optional
        Sampling parameters, stored as group attributes for provenance.
    overwrite : bool, optional
    """
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    peri_time = _pad_ragged([p["peri_time"] for p in peri_apo])
    peri_r = _pad_ragged([p["peri_r"] for p in peri_apo])
    apo_time = _pad_ragged([p["apo_time"] for p in peri_apo])
    apo_r = _pad_ragged([p["apo_r"] for p in peri_apo])
    n_peri = np.array([len(p["peri_time"]) for p in peri_apo])
    n_apo = np.array([len(p["apo_time"]) for p in peri_apo])

    mode = "a" if filename.exists() else "w"
    with h5py.File(filename, mode) as f:
        grp = f.require_group(f"suite_{sim_id:04d}")
        _write_datasets(
            grp,
            {
                "pid": pid_sample,
                "time": times,
                "energy": energy,
                "ang_mom": ang_mom,
                "peri_time": peri_time,
                "peri_r": peri_r,
                "apo_time": apo_time,
                "apo_r": apo_r,
                "n_peri": n_peri,
                "n_apo": n_apo,
            },
            overwrite,
        )
        grp.attrs["r_min"] = r_min
        grp.attrs["r_max"] = r_max
        grp.attrs["n_sample"] = len(pid_sample)
        if seed is not None:
            grp.attrs["seed"] = seed


def run_halo_sample(
    sim_id: int,
    snapshot_dir: str,
    snapname: str,
    centers_path: str,
    centers_file: str,
    orbits_output,
    params_output,
    suite: str = "Sheng24",
    n_sample: int = 50,
    r_min: float = 20.0,
    r_max: float = 300.0,
    seed: Optional[int] = None,
) -> None:
    """
    Run the full halo_sample pipeline for one simulation: load context,
    sample particles, extract orbits, derive orbital parameters, and write
    trajectories/parameters to their respective output files.

    Parameters
    ----------
    sim_id : int
        Simulation identifier.
    snapshot_dir : str
        Directory containing the simulation's snapshot files.
    snapname : str
        Snapshot filename prefix.
    centers_path, centers_file : str
        Location of the MW/LMC center-of-mass trajectory file.
    orbits_output : str or Path
        Output HDF5 path for orbit trajectories (agama.orbit format).
    params_output : str or Path
        Output HDF5 path for derived orbital parameters.
    suite : str, optional
    n_sample : int, optional
    r_min, r_max : float, optional
        Radial shell (kpc) used to pre-select eligible particles.
    seed : int, optional
        Seed for the particle-sampling RNG, for reproducibility.
    """
    rng = np.random.default_rng(seed)

    ctx = load_sim_context(
        sim_id, snapshot_dir, snapname, centers_path, centers_file, suite=suite
    )

    ref_nsnap = int(ctx["snap_suffixes"][0])
    ref_snap = load_particle_data(
        snapshot_dir, snapname, ["MWhalo"], ref_nsnap, suite=suite,
        quantities=["pid", "pos", "vel", "pot"],
    )["MWhalo"]
    ref_pos = ref_snap["pos"] - ctx["mw_center"][0]
    ref_vel = ref_snap["vel"] - ctx["mw_vel_center"][0]
    pid_sample = select_particle_sample(
        ref_snap["pid"], ref_pos, ref_vel, ref_snap["pot"],
        r_min=r_min, r_max=r_max, n_sample=n_sample, rng=rng,
    )

    pos, vel, pot = extract_orbit_sample(
        snapshot_dir, snapname, ctx["snap_suffixes"], pid_sample,
        ctx["mw_center"], ctx["mw_vel_center"], suite=suite,
    )

    params = compute_orbit_parameters(ctx["times"], pos, vel, pot)

    write_orbit_trajectories(
        orbits_output, sim_id, ctx["times"], pid_sample, pos, vel, pot,
        r_min=r_min, r_max=r_max, seed=seed,
    )
    write_orbit_parameters(
        params_output, sim_id, ctx["times"], pid_sample,
        params["energy"], params["ang_mom"], params["peri_apo"],
        r_min=r_min, r_max=r_max, seed=seed,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample DM halo particles, extract their orbits, and derive "
        "orbital parameters for one XMC-Atlas simulation."
    )
    parser.add_argument("sim_id", type=int, help="Simulation ID")
    parser.add_argument("--suite", type=str, default="Sheng24", choices=["Sheng24"])
    parser.add_argument(
        "--snapshot-dir", type=str, default=None,
        help="Directory containing the simulation's snapshots "
        "(default: suites/<suite>/data/Model_<sim_id>)",
    )
    parser.add_argument("--snapname", type=str, default="snapshot")
    parser.add_argument("--centers-path", type=str, default="suites/Sheng24/orbits")
    parser.add_argument("--centers-file", type=str, default="MW_LMC_orbits_iso.txt")
    parser.add_argument("--n-sample", type=int, default=50)
    parser.add_argument("--r-min", type=float, default=20.0)
    parser.add_argument("--r-max", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--orbits-output", type=str, default=None,
        help="Output HDF5 path for orbit trajectories (default: halo_sample_orbits.h5)",
    )
    parser.add_argument(
        "--params-output", type=str, default=None,
        help="Output HDF5 path for orbital parameters (default: halo_sample_params.h5)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    snapshot_dir = args.snapshot_dir or f"suites/{args.suite}/data/Model_{args.sim_id}"
    orbits_output = args.orbits_output or "halo_sample_orbits.h5"
    params_output = args.params_output or "halo_sample_params.h5"

    run_halo_sample(
        args.sim_id,
        snapshot_dir=snapshot_dir,
        snapname=args.snapname,
        centers_path=args.centers_path,
        centers_file=args.centers_file,
        orbits_output=orbits_output,
        params_output=params_output,
        suite=args.suite,
        n_sample=args.n_sample,
        r_min=args.r_min,
        r_max=args.r_max,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
