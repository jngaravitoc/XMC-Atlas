"""
Minimal utilities to create a Gizmo-like snapshot and fit Agama multipole/CylSpline
potentials from it, with a small benchmark routine in ``if __name__ == "__main__"``.

Assumptions / caveats
---------------------   
- Particle positions MUST already be centered and rotated into the desired frame
  before being passed to these functions (no centering/rotation is performed here).
- If a disk-like component is modeled with CylSpline, the disk should lie near the
  XY plane (small tilts are acceptable; large misalignments will degrade the fit).
- If gas temperature is not provided, gas is treated as "hot" and included with
  the multipole (dark-matter-like) component.
- rmax_sel is required and controls the selection aperture for particles.
- This module focuses on validating inputs, splitting gas by temperature (when
  available), and saving Agama coefficient files. It raises informative errors
  for invalid inputs.

Style
-----
- Numpy-style docstrings, PEP8-compliant, Python 3.13 type hints.
"""

from __future__ import annotations

import os
import time
from typing import Iterable, Mapping, Sequence

import numpy as np
import agama

# Units for Gadget: assume 1e10 Msol, kpc, km/s
agama.setUnits(mass=1e10, length=1, velocity=1)  # Msol, kpc, km/s



def compute_agama_coefs(snapshot_dir, snapname, expansion_center, npart, dt, runtime_log):
    import agama
    from agama_external_sims import create_GizmoLike_snapshot, fit_potential
    #pos, mass, nsnap  = load_mwhalo(snapname, snapshot_dir, npart)
    #pos, mass, nsnap  = load_snapshot(snapname, snapshot_dir, outpath, npart)
 
    halo_data = LoadSim(snapshot_dir, snapname, expansion_center, suite)
    mw_halo_particles = halo_data.load_halo(halo='MW', quantities=['pos', 'vel', 'mass'], npart=npart)
    # Load center
    pos_center, vel_center, nsnap = load_center(snapname, expansion_center)
    mwhalo_recenter = nba.com.CenterHalo(mw_halo_particles)
    mwhalo_recenter.recenter(pos_center, [0,0,0])    

    mw_disk_particles = halo_data.load_mw_disk(quantities=['pos', 'vel', 'mass'])
    mw_disk_recenter = nba.com.CenterHalo(mw_disk_particles)
    mw_disk_recenter.recenter(pos_center, [0,0,0])    


    sim_time = nsnap * dt # Gyrs # TODO get this from header! 
    print(sim_time)
    # re-center
    
    # Compute coefficients
    start_time = time.time()
    print(mw_halo_particles.keys())    

    # build snapshot WITHOUT gas to demonstrate robustness when gas absent
    
    snapshot = create_GizmoLike_snapshot(
        pos_dark=mw_halo_particles['pos'],
        mass_dark=mw_halo_particles['mass'],
        pos_star=mw_disk_particles['pos'],
        mass_star=mw_disk_particles['mass'],
        #pos_gas=np.ones((100, 3)),
        #mass_gas=np.ones(100),
        # temperature_gas=temperature_gas,
    )

    print("Running fit_potential on synthetic snapshot...")
    t0 = time.perf_counter()
    outputs = fit_potential(
        snapshot,
        nsnap=nsnap,
        sym=["n"],
        pole_l=[4],
        rmax_sel=600.0,
        rmax_exp=500.0,
        save_dir="./demo_output",
        file_ext="spline",
        verbose=True,
        halo='MW_iso_beta1',
    )
    dt = time.perf_counter() - t0

    print("\nBenchmark complete.")
    print(f"Elapsed time: {dt:.3f} s")
    print("Generated files summary:")
    for key, files in outputs.items():
        print(f"  {key}: {len(files)} files")
        for f in files[:3]:
            print(f"    - {f}")
        if len(files) > 3:
            print(f"    ... (+{len(files)-3} more)")
    print("Done.")



    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("*Done computing Agama coefficients")



def create_GizmoLike_snapshot(
    pos_dark: np.ndarray,
    mass_dark: np.ndarray,
    pos_star: np.ndarray | None = None,
    mass_star: np.ndarray | None = None,
    pos_gas: np.ndarray | None = None,
    mass_gas: np.ndarray | None = None,
    temperature_gas: np.ndarray | None = None,
) -> dict:
    """
    Create a minimal Gizmo-like snapshot dictionary.

    Parameters
    ----------
    pos_dark
        (N_dark, 3) array of dark matter positions.
    mass_dark
        (N_dark,) array of dark matter masses.
    pos_star, mass_star
        Optional star positions and masses (N_star, 3) and (N_star,).
    pos_gas, mass_gas
        Optional gas positions and masses (N_gas, 3) and (N_gas,).
    temperature_gas
        Optional gas temperatures (N_gas,). If provided, used by `fit_potential`
        to split cold/hot gas.

    Returns
    -------
    dict
        Snapshot dictionary with keys 'dark', 'star', 'gas', 'host'. Each species
        entry contains arrays named `host.distance` and `mass` (and `temperature`
        for gas if provided).
    """
    pos_dark = np.asarray(pos_dark, dtype=float)
    mass_dark = np.asarray(mass_dark, dtype=float)

    if pos_dark.ndim != 2 or pos_dark.shape[1] != 3:
        raise ValueError("pos_dark must be shape (N, 3).")
    if mass_dark.shape[0] != pos_dark.shape[0]:
        raise ValueError("mass_dark length must match pos_dark rows.")

    snapshot: dict = {
        "dark": {
            "host.distance": pos_dark,
            "mass": mass_dark,
        },
        # include star/gas only if provided; leave empty dicts optional
        "star": {},
        "gas": {},
    }

    if pos_star is not None or mass_star is not None:
        if pos_star is None or mass_star is None:
            raise ValueError("Both pos_star and mass_star must be provided to add stars.")
        pos_star = np.asarray(pos_star, dtype=float)
        mass_star = np.asarray(mass_star, dtype=float)
        if pos_star.ndim != 2 or pos_star.shape[1] != 3:
            raise ValueError("pos_star must be shape (N, 3).")
        if mass_star.shape[0] != pos_star.shape[0]:
            raise ValueError("mass_star length must match pos_star rows.")
        snapshot["star"]["host.distance"] = pos_star
        snapshot["star"]["mass"] = mass_star

    if pos_gas is not None or mass_gas is not None:
        if pos_gas is None or mass_gas is None:
            raise ValueError("Both pos_gas and mass_gas must be provided to add gas.")
        pos_gas = np.asarray(pos_gas, dtype=float)
        mass_gas = np.asarray(mass_gas, dtype=float)
        if pos_gas.ndim != 2 or pos_gas.shape[1] != 3:
            raise ValueError("pos_gas must be shape (N, 3).")
        if mass_gas.shape[0] != pos_gas.shape[0]:
            raise ValueError("mass_gas length must match pos_gas rows.")
        snapshot["gas"]["host.distance"] = pos_gas
        snapshot["gas"]["mass"] = mass_gas
        if temperature_gas is not None:
            temperature_gas = np.asarray(temperature_gas, dtype=float)
            if temperature_gas.shape[0] != pos_gas.shape[0]:
                raise ValueError("temperature_gas length must match pos_gas rows.")
            snapshot["gas"]["temperature"] = temperature_gas

    return snapshot


def fit_potential(
    part: Mapping[str, Mapping[str, np.ndarray]],
    nsnap: int,
    *,
    sym: str | Sequence[str] = "n",
    pole_l: int | Sequence[int] = 4,
    rmax_sel: float | None = None,
    rmax_exp: float = 500.0,
    file_ext: str = "spline",
    save_dir: str = "./",
    halo: str | None = None,
    spec_ind: Mapping[str, Iterable[int]] | None = None,
    kind: str = "whole",
    verbose: bool = True,
    subsample_factor: float = 1.0,
    cold_temp_log10_thresh: float = 4.5,
) -> dict:
    """
    Fit combined Agama potentials (Multipole + CylSpline) from a Gizmo-like snapshot.

    See function docstring in earlier code block for parameter details.
    """
    if rmax_sel is None or rmax_sel <= 0:
        raise ValueError("rmax_sel is required and must be > 0.")

    allowed_syms = {"n": "none", "a": "axi", "s": "sph", "t": "triax"}
    if isinstance(sym, str):
        syms = [sym]
    else:
        syms = list(sym)
    for s in syms:
        if s not in allowed_syms:
            raise ValueError(f"Unknown symmetry '{s}'. Allowed: {list(allowed_syms)}")

    if isinstance(pole_l, int):
        pole_ls = [pole_l]
    else:
        pole_ls = list(pole_l)
    if any((not isinstance(l, int) or l < 0) for l in pole_ls):
        raise ValueError("pole_l entries must be non-negative integers.")

    if kind not in ("whole", "dark", "bar"):
        raise ValueError("kind must be one of {'whole', 'dark', 'bar'}.")

    # --- detect actual particle species: require both 'mass' and 'host.distance' ---
    species_keys = [
        k for k, v in part.items() if isinstance(v, dict) and ("mass" in v) and ("host.distance" in v)
    ]
    if not species_keys:
        raise ValueError(
            "No particle species found in `part`. Expected at least one species "
            "dictionary containing 'mass' and 'host.distance' (e.g. 'dark')."
        )
    if verbose:
        ignored = set(part.keys()) - set(species_keys)
        if ignored:
            print("Ignoring non-particle keys in snapshot:", ignored)

    # Build default spec_ind mapping only for detected species
    spec_ind_map: dict[str, np.ndarray] = {}
    for sp in species_keys:
        n_particles = int(np.asarray(part[sp]["mass"]).shape[0])
        if spec_ind and sp in spec_ind:
            spec_ind_map[sp] = np.asarray(list(spec_ind[sp]), dtype=int)
        else:
            spec_ind_map[sp] = np.arange(n_particles, dtype=int)

    # Compute distances and vectors for available species (dark/star/gas)
    dist: dict[str, np.ndarray] = {}
    dist_vectors: dict[str, np.ndarray] = {}
    for sp in ("dark", "star", "gas"):
        if sp in part and "host.distance" in part[sp]:
            pos = np.asarray(part[sp]["host.distance"])
            idx = spec_ind_map.get(sp, np.arange(pos.shape[0], dtype=int))
            # guard index bounds
            if np.any(idx >= pos.shape[0]) or np.any(idx < 0):
                raise IndexError(f"spec_ind for species '{sp}' contains invalid indices.")
            pos_sel = pos[idx]
            dist_vectors[sp] = pos_sel
            dist[sp] = np.linalg.norm(pos_sel, axis=1)
        else:
            dist_vectors[sp] = np.empty((0, 3), dtype=float)
            dist[sp] = np.empty((0,), dtype=float)

    # Select particles within rmax_sel for each species
    selected: dict[str, np.ndarray] = {}
    for sp in ("dark", "star", "gas"):
        selected[sp] = dist[sp] < rmax_sel

    # Prepare masses (apply subsample_factor)
    masses: dict[str, np.ndarray] = {}
    for sp in ("dark", "star", "gas"):
        if sp in part and "mass" in part[sp]:
            m_all = np.asarray(part[sp]["mass"])
            idx = spec_ind_map.get(sp, np.arange(m_all.shape[0], dtype=int))
            if np.any(idx >= m_all.shape[0]) or np.any(idx < 0):
                raise IndexError(f"spec_ind for species '{sp}' contains invalid indices.")
            masses[sp] = m_all[idx] * float(subsample_factor)
        else:
            masses[sp] = np.empty((0,), dtype=float)

    # Gas temperature handling: if present, split cold/hot; else treat all as hot
    gas_has_temp = "gas" in part and "temperature" in part["gas"]
    if gas_has_temp:
        temp_all = np.asarray(part["gas"]["temperature"])
        temp_idx = spec_ind_map.get("gas", np.arange(temp_all.shape[0], dtype=int))
        if np.any(temp_idx >= temp_all.shape[0]) or np.any(temp_idx < 0):
            raise IndexError("spec_ind for 'gas' contains invalid indices.")
        temp_sel = temp_all[temp_idx]
        if np.any(temp_sel <= 0.0):
            raise ValueError("All gas temperatures must be > 0 to compute log10.")
        log10_temp = np.log10(temp_sel)
        tsel = log10_temp < cold_temp_log10_thresh
    else:
        tsel = np.zeros_like(dist["gas"], dtype=bool)

    # Build arrays for cylspline (bar: stars + cold gas) and multipole (dark + hot gas)
    pos_bar_list: list[np.ndarray] = []
    m_bar_list: list[np.ndarray] = []

    # Stars (if available) go to CylSpline (bar)
    if "star" in part and masses["star"].size > 0:
        pos_star_sel = dist_vectors["star"][selected["star"]]
        m_star_sel = masses["star"][selected["star"]]
        if pos_star_sel.size > 0:
            pos_bar_list.append(pos_star_sel)
            m_bar_list.append(m_star_sel)
            if verbose:
                print(f"Selected {pos_star_sel.shape[0]} star particles within {rmax_sel}.")

    # Gas selections
    if "gas" in part and masses["gas"].size > 0:
        pos_gas_sel = dist_vectors["gas"][selected["gas"]]
        m_gas_sel = masses["gas"][selected["gas"]]
        if pos_gas_sel.size > 0:
            tsel_sel = tsel[selected["gas"]]
            pos_gas_cold = pos_gas_sel[tsel_sel]
            m_gas_cold = m_gas_sel[tsel_sel]
            pos_gas_hot = pos_gas_sel[~tsel_sel]
            m_gas_hot = m_gas_sel[~tsel_sel]
            if pos_gas_cold.size > 0:
                pos_bar_list.append(pos_gas_cold)
                m_bar_list.append(m_gas_cold)
                if verbose:
                    print(f"Selected {pos_gas_cold.shape[0]} cold gas particles for CylSpline (disk).")
        else:
            pos_gas_hot = np.empty((0, 3), dtype=float)
            m_gas_hot = np.empty((0,), dtype=float)
    else:
        pos_gas_hot = np.empty((0, 3), dtype=float)
        m_gas_hot = np.empty((0,), dtype=float)

    # Dark matter selection for multipole
    if "dark" in part and masses["dark"].size > 0:
        pos_dark_sel = dist_vectors["dark"][selected["dark"]]
        m_dark_sel = masses["dark"][selected["dark"]]
    else:
        pos_dark_sel = np.empty((0, 3), dtype=float)
        m_dark_sel = np.empty((0,), dtype=float)

    # Multipole component = dark + hot gas
    if pos_gas_hot.size:
        pos_mul = np.vstack([pos_dark_sel, pos_gas_hot]) if pos_dark_sel.size else pos_gas_hot
        m_mul = np.hstack([m_dark_sel, m_gas_hot]) if m_dark_sel.size else m_gas_hot
    else:
        pos_mul = pos_dark_sel
        m_mul = m_dark_sel

    if pos_mul.size == 0 and kind in ("whole", "dark"):
        raise ValueError("No particles selected for multipole component (dark+hot).")

    # CylSpline (bar) component = stars + cold gas if any
    pos_bar = np.vstack(pos_bar_list) if pos_bar_list else np.empty((0, 3), dtype=float)
    m_bar = np.hstack(m_bar_list) if m_bar_list else np.empty((0,), dtype=float)

    # Prepare output directory using rmax_sel rounded as folder name
    rmax_ctr = int(round(rmax_sel))
    out_dir = os.path.join(save_dir, "potential", f"{10}kpc") # 10kpc refers to apertured for COM-veloc 
    os.makedirs(out_dir, exist_ok=True)

    output_files: dict[str, list[str]] = {"multipole": [], "cylspline": []}

    # Iterate over requested symmetries and pole orders and fit
    for s in syms:
        sym_label = allowed_syms[s]
        for l in pole_ls:
            if verbose:
                print(f"Fitting symmetry='{s}' ({sym_label}), order={l}  (rmax_exp={rmax_exp})")

            # Multipole: only if kind != 'bar'
            if kind in ("whole", "dark"):
                if pos_mul.size == 0:
                    if verbose:
                        print("Skipping multipole (no particles).")
                else:
                    p_dark = agama.Potential(
                        type="Multipole",
                        particles=(pos_mul, m_mul),
                        lmax=l,
                        symmetry=s,
                        rmin=0.1,
                        rmax=rmax_exp,
                    )
                    nsnap_str = f"{int(nsnap):03d}"
                    fname_mul = f"{nsnap_str}.dark.{sym_label}_{l}"
                    if halo:
                        fname_mul += f".{halo}"
                    fname_mul += f".coef_mul_{file_ext}"
                    path_mul = os.path.join(out_dir, fname_mul)
                    p_dark.export(path_mul)
                    output_files["multipole"].append(path_mul)
                    if verbose:
                        print(f"Saved multipole coeffs -> {path_mul}")

            # CylSpline: only if we have bar material and kind != 'dark'
            if kind in ("whole", "bar"):
                if pos_bar.size == 0:
                    if verbose:
                        print("No bar (CylSpline) particles selected; skipping CylSpline.")
                else:
                    p_bar = agama.Potential(
                        type="CylSpline",
                        particles=(pos_bar, m_bar),
                        mmax=l,
                        symmetry=s,
                        rmin=0.1,
                        rmax=rmax_exp,
                    )
                    nsnap_str = f"{int(nsnap):03d}"
                    fname_bar = f"{nsnap_str}.bar.{sym_label}_{l}"
                    if halo:
                        fname_bar += f".{halo}"
                    fname_bar += f".coef_cylsp_{file_ext}"
                    path_bar = os.path.join(out_dir, fname_bar)
                    p_bar.export(path_bar)
                    output_files["cylspline"].append(path_bar)
                    if verbose:
                        print(f"Saved CylSpline coeffs -> {path_bar}")

    if verbose:
        print("Done fitting potential models.")

    return output_files


# ---------------------------
# Benchmark / example routine
# ---------------------------
def _sample_spherical(how_many: int, scale_radius: float = 50.0) -> np.ndarray:
    """
    Sample 'how_many' spherical positions with an approximate declining density.
    Produces radii from an exponential-like tail and random directions.
    Returns array shape (N, 3).
    """
    u = np.random.random(how_many)
    r = -scale_radius * np.log(1.0 - u * 0.95)  # truncated tail
    cos_theta = 2.0 * np.random.random(how_many) - 1.0
    phi = 2.0 * np.pi * np.random.random(how_many)
    sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)
    x = r * sin_theta * np.cos(phi)
    y = r * sin_theta * np.sin(phi)
    z = r * cos_theta
    return np.vstack((x, y, z)).T


def _sample_disk(how_many: int, r_scale: float = 3.0, z_sigma: float = 0.2) -> np.ndarray:
    """
    Sample 'how_many' disk-like positions (exponential radial profile, thin gaussian z).
    Returns array shape (N, 3).
    """
    u = np.random.random(how_many)
    R = -r_scale * np.log(1.0 - u)
    phi = 2.0 * np.pi * np.random.random(how_many)
    x = R * np.cos(phi)
    y = R * np.sin(phi)
    z = np.random.normal(loc=0.0, scale=z_sigma, size=how_many)
    x += np.random.normal(scale=0.01 * r_scale, size=how_many)
    y += np.random.normal(scale=0.01 * r_scale, size=how_many)
    return np.vstack((x, y, z)).T


if __name__ == "__main__":
    np.random.seed(13)

    # particle counts (modest for quick demo)
    N_dark = 20000
    N_star = 5000
    N_gas = 3000

    print("Sampling synthetic snapshot...")
    pos_dark = _sample_spherical(N_dark, scale_radius=30.0)
    mass_dark = np.ones(N_dark) * (1.0e6 / N_dark)

    pos_star = _sample_disk(N_star, r_scale=3.0, z_sigma=0.15)
    mass_star = np.ones(N_star) * (5.0e5 / N_star)

    pos_gas = _sample_disk(N_gas, r_scale=4.0, z_sigma=0.3)
    mass_gas = np.ones(N_gas) * (1.0e5 / N_gas)

    # Assign random temperatures (log-uniform between 10^3 and 10^6 K)
    logt = np.random.uniform(3.0, 6.0, size=N_gas)
    temperature_gas = 10.0 ** logt

    # build snapshot WITHOUT gas to demonstrate robustness when gas absent
    snapshot = create_GizmoLike_snapshot(
        pos_dark=pos_dark,
        mass_dark=mass_dark,
        pos_star=pos_star,
        mass_star=mass_star,
        pos_gas=pos_gas,
        mass_gas=mass_gas,
        # temperature_gas=temperature_gas,
    )

    print("Running fit_potential on synthetic snapshot...")
    t0 = time.perf_counter()
    outputs = fit_potential(
        snapshot,
        nsnap=0,
        sym=["n", "a"],
        pole_l=[2, 4],
        rmax_sel=600.0,
        rmax_exp=100.0,
        save_dir="./demo_output",
        file_ext="spline",
        verbose=True,
    )
    dt = time.perf_counter() - t0

    print("\nBenchmark complete.")
    print(f"Elapsed time: {dt:.3f} s")
    print("Generated files summary:")
    for key, files in outputs.items():
        print(f"  {key}: {len(files)} files")
        for f in files[:3]:
            print(f"    - {f}")
        if len(files) > 3:
            print(f"    ... (+{len(files)-3} more)")
    print("Done.")
