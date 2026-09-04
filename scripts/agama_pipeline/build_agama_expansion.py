"""
Pipeline to compute optimal basis for the Sheng+24 simulation suite.

Author: github.com/jngaravitoc

Usage: python build_basis.py sim_id halo_component

TODO:
    [] Check output files into folders 
    [] Optimize fit params for bulge and LMC
        - [] Chose amplitude best init amplitude for bulge and LMC's halos in the fit
    [] Check if functions are well distributed across scripts
    [] Complete Pipeline tests

"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
# third-party
THIS_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = THIS_DIR.parent
sys.path.append(str(SCRIPTS_DIR))
sys.path.append(str(SCRIPTS_DIR / "exp_pipeline"))
sys.path.append(str(SCRIPTS_DIR / "exp_fields"))
sys.path.append(str(SCRIPTS_DIR / "exp_visuals"))
sys.path.append(str(SCRIPTS_DIR / "agama_pipeline"))

from config import PROJECT_DIR, SIMS_PARAMS_PATH, get_simulation_path
from ios_nbody_sims import load_particle_data
from sanity_checks import check_monotonic_contiguous_snapshots
from compute_bfe_helpers import load_sheng24_exp_center, get_snapshot_suffixes
from agama_BFEs import fitAgamaBFE, write_snapshot_coefs_to_h5


# Particle-component names AGAMA's pipeline routes into a fit (see
# agama_BFEs.py's _MW_SPHERICAL_KEYS | _MW_DISK_KEYS | _MW_BULGE_KEYS |
# _LMC_KEYS). Anything outside this set is not a valid `components` entry.
VALID_COMPONENTS = {"MWhalo", "LMChalo", "MWdisk", "MWbulge"}


def resolve_output_dir(output_dir):
    """
    Resolve an output directory to an absolute path.

    Parameters
    ----------
    output_dir : str or Path
        Destination directory. Used as-is if already absolute; otherwise
        treated as relative to ``PROJECT_DIR / "scripts"``.

    Returns
    -------
    Path
        The resolved absolute directory (not created here).
    """
    output_path = Path(output_dir)
    if output_path.is_absolute():
        return output_path
    return PROJECT_DIR / "scripts" / output_path


def resolve_suite_layout(simulation_suite, simulation_model, sim_id):
    """
    Resolve the on-disk snapshot layout for one simulation.

    Parameters
    ----------
    simulation_suite : str
        Suite key from config.py, one of ``"sheng"`` or ``"gc21"``.
    simulation_model : str
        Model key within the suite (e.g. ``"sheng_iso"``, ``"3b0"``).
    sim_id : int
        Simulation ID (only used for ``"sheng"``, to locate ``Model_<sim_id>``).

    Returns
    -------
    dict
        ``{"snapshot_path": Path, "snapname": str, "suite_label": str}``.

    Raises
    ------
    ValueError
        If *simulation_suite* is not ``"sheng"`` or ``"gc21"``.
    """
    base_path = get_simulation_path(simulation_suite, simulation_model)

    if simulation_suite == "sheng":
        return {
            "snapshot_path": base_path / f"Model_{sim_id}",
            "snapname": "snapshot",
            "suite_label": "Sheng24",
        }

    if simulation_suite == "gc21":
        mwlmc_tag = simulation_model[1:]
        model_number = simulation_model[0]
        return {
            "snapshot_path": base_path / "out",
            "snapname": f"MWLMC{model_number}_100M_{mwlmc_tag}_vir_OM3_G4",
            "suite_label": "GC21",
        }

    raise ValueError(f"Unsupported simulation suite: {simulation_suite}")


def load_gc21_centers(simulation_model):
    """
    Load GC21's MW/LMC orbit centers from their text files.

    Parameters
    ----------
    simulation_model : str
        GC21 model key (e.g. ``"3b0"``); the first character is the model
        number, the rest is the "vB"-style tag used in the orbit filenames.

    Returns
    -------
    dict
        ``{"time": ndarray, "mw_center": ndarray (N, 3), "lmc_center":
        ndarray (N, 3)}``.

    Raises
    ------
    FileNotFoundError
        If either orbit file is missing for this model.
    """
    orbit_dir = PROJECT_DIR / "suites" / "GC21" / "orbits"
    model_tag = simulation_model.upper().replace("B", "b")
    mw_file = orbit_dir / f"GC21M{model_tag}_orbit_mw.txt"
    lmc_file = orbit_dir / f"GC21M{model_tag}_orbit_lmc.txt"

    if not mw_file.is_file() or not lmc_file.is_file():
        raise FileNotFoundError(
            f"GC21 orbit files not found for model '{simulation_model}': {mw_file}, {lmc_file}"
        )

    mw_data = np.loadtxt(mw_file)
    lmc_data = np.loadtxt(lmc_file)

    return {
        "time": mw_data[:, 0],
        "mw_center": mw_data[:, 1:4],
        "lmc_center": lmc_data[:, 1:4],
    }


def main(
    simulation_suite,
    simulation_model,
    SIM_ID,
    ncoefs,
    components,
    output_dir):
    """
    Build AGAMA basis-function-expansion coefficients for one simulation,
    one snapshot at a time, via :func:`agama_BFEs.fitAgamaBFE`.

    Parameters
    ----------
    simulation_suite : str
        Suite key from config.py, one of ``"sheng"`` or ``"gc21"``.
    simulation_model : str
        Model key within the suite (e.g. ``"sheng_iso"``, ``"3b0"``).
    SIM_ID : int
        Simulation ID used to select orbit/center rows.
    ncoefs : int
        Number of coefficient samples. Currently unused by this function
        body (accepted for CLI/API compatibility).
    components : sequence of str
        Particle components to load and fit, forwarded to
        ``load_particle_data`` and ``fitAgamaBFE``. Either ``["all"]``
        (expands to every valid component: ``sorted(VALID_COMPONENTS)`` --
        ``"LMChalo"``, ``"MWbulge"``, ``"MWdisk"``, ``"MWhalo"``) or an
        explicit subset of those same names.
    output_dir : str or Path
        Destination directory for the fitted coefficient files, forwarded
        to :func:`resolve_output_dir`. Created if it doesn't exist.

    Raises
    ------
    ValueError
        If *components* contains a name outside :data:`VALID_COMPONENTS`
        (and isn't the ``["all"]`` sentinel), or if *simulation_suite* is
        not ``"sheng"``/``"gc21"``.
    """
    suite = simulation_suite.lower()
    layout = resolve_suite_layout(suite, simulation_model, SIM_ID)
    SNAPSHOT_PATH = layout["snapshot_path"]
    SNAPNAME = layout["snapname"]
    suite_label = layout["suite_label"]
    SIM_PARAMS_FILE = 'MW_LMC_orbits_iso.txt'

    nsnap = 0  # ??

    outpath = resolve_output_dir(output_dir)
    outpath.mkdir(parents=True, exist_ok=True)
    
    #figure_name = '{}_{:04d}_density_profile_evolution.png'.format(component, SIM_ID)
    

    
    #--------------------------
    # profile fitting params
    #--------------------------
    
    rmin = 0.1 # should be 4*softening
    nbins = 101 # <- Should this be fixed? 

    
    #  
    lmax = int(12)
    nmax = int(20)
    rmapping = 1.0
    
    #--------------------------
    # Pipeline params:
    #--------------------------
    paranoid = True
    cwd_path = os.getcwd()
    #--------------------------
    # log ouput.
    # 

    #------------------------------------------

    #--------------------------
    # 1. Load halo centers
    # -------------------------


    if suite == "sheng":
        sim_params = load_sheng24_exp_center(
            str(SIMS_PARAMS_PATH), 
            SIM_PARAMS_FILE,
            SIM_ID,
            return_vel=False,
        )

        tsim = sim_params['time']
        mw_center = sim_params['mw_center']
        lmc_center = sim_params['lmc_center']

        centers = {
            "mw_center": mw_center,
            "lmc_center": lmc_center,
        }
    elif suite == "gc21":
        sim_params = load_gc21_centers(simulation_model)
        tsim = sim_params["time"]
        centers = {
            "mw_center": sim_params["mw_center"],
            "lmc_center": sim_params["lmc_center"],
        }
    else:
        raise ValueError(f"Suite {suite} not implemented")

    
    print("-> Done loading simulation centers")

    snap_suffixes = get_snapshot_suffixes(str(SNAPSHOT_PATH), prefix=SNAPNAME+"_")
    NSNAPS = len(snap_suffixes)
    
    if paranoid == True:
        snap_check, missing_snaps = check_monotonic_contiguous_snapshots(snap_suffixes)
        print(f"[paranoid] {NSNAPS} snapshots found in {SNAPSHOT_PATH}")
        print(f"[paranoid] Snapshots are monotonic:", snap_check)
        if snap_check == False:
            print(f"[error] snapshots {missing_snaps} are missing")
        assert NSNAPS == len(tsim), "number of snapshots found differ from size of centers"
    
    #------------------------------------------
    
    #----------------------------------------
    # 2. Load particle data and recenter halo
    # ----------------------------------------

    
    gadget_particle_mass = 1e10
    units = [('mass', 'Msun', gadget_particle_mass),
             ('length', 'kpc', 1.0),
             ('velocity', 'km/s', 1.0),
             ('G', 'mixed', 43007.1)]

    if components == ['all']:
        components = sorted(VALID_COMPONENTS)

    unknown = set(components) - VALID_COMPONENTS
    if unknown:
        raise ValueError(
            f"Unknown component(s) {sorted(unknown)}; expected 'all' or a "
            f"subset of {sorted(VALID_COMPONENTS)}"
        )

    for i in range(NSNAPS):
        snapshot = SNAPNAME + "{:03d}.hdf5".format(i)
        data = load_particle_data(str(SNAPSHOT_PATH), SNAPNAME, components, nsnap=i, suite=suite_label, msun_units=True)
        fitAgamaBFE(
            part=data,
            center_coords= centers,
            nsnap = i,
            OUTPUT_PATH=str(outpath))
    

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build spherical basis functions for external simulations."
    )

    # Required positional arguments
    parser.add_argument(
        "simulation_suite",
        type=str,
        choices=["sheng", "gc21"],
        help="Simulation suite key from config.py (e.g. sheng, gc21)",
    )

    parser.add_argument(
        "simulation_model",
        type=str,
        help="Model key from config.py for the selected suite (e.g. sheng_iso, 3b0)",
    )

    parser.add_argument(
        "sim_id",
        type=int,
        help="Simulation ID used to select orbit/center rows.",
    )
    
	# Optional arguments
    parser.add_argument(
        "--coefs_freq",
        type=int,
        nargs="?",
        default=1,
        help="Number of coefficient samples (default: 10)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        nargs="?",
        default=str(SCRIPTS_DIR / "agama_expansions"),
        help="Directory output name for datasets"
    )

    parser.add_argument(
        "--components",
        type=str,
        nargs="+",
        default=["all"],
        help=(
            "Particle components to fit (default: ['all'], expands to "
            f"{sorted(VALID_COMPONENTS)}). Otherwise an explicit subset of "
            f"{sorted(VALID_COMPONENTS)}; anything else raises ValueError."
        ),
    )

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    
    main(
        simulation_suite=args.simulation_suite,
        simulation_model=args.simulation_model,
        SIM_ID=args.sim_id,
        ncoefs=args.coefs_freq,
        components=args.components,
        output_dir=args.output_dir,
    )

   
    final_output_dir = resolve_output_dir(args.output_dir)

    write_snapshot_coefs_to_h5(
    snapshot_ids = range(0, 101),
    coef_file_patterns=[
    str(final_output_dir / "{snap:03d}.MW.none_6.coef_mult"),
    str(final_output_dir / "{snap:03d}.MW.none_6.coef_bulge"),
    str(final_output_dir / "{snap:03d}.MW.none_6.coef_cylsp"),
    str(final_output_dir / "{snap:03d}.LMC.none_6.coef_mult"),],
    h5_output_paths=[
        str(final_output_dir / "MW.none_6.coef_mult.h5"),
        str(final_output_dir / "MW.none_6.coef_bulge.h5"),
        str(final_output_dir / "MW.none_6.coef_cylsp.h5"),
        str(final_output_dir / "LMC.none_6.coef_mult.h5"),]
    )
