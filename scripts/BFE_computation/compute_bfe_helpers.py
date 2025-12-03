import os
import csv
import logging
import yaml
import pyEXP

def check_snaps_in_folder(folder_path, expected_files):
    """
    Check that all expected snapshots exist inside a folder.

    Parameters
    ----------
    folder_path : str or Path
        Path to the directory to check.
    expected_files : list of str
        Filenames expected to appear in the folder (exact matches).

    Returns
    -------
    missing_files : list of str
        Files that were expected but not found.

    Raises
    ------
    FileNotFoundError
        If any expected file is missing.
    """
    missing = []
    
    for fname in expected_files:
        full_path = os.path.join(folder_path, fname)
        if not os.path.isfile(full_path):
            missing.append(fname)

    if missing:
        raise FileNotFoundError(
            f"The following files are missing in {folder_path}:\n{missing}"
        )

    return True

def check_coefficients_path(outpath):
    if os.path.isdir(outpath):
        logging.info(f"> Coefficients {outpath} folder exists")
    else:
        logging.info(f"> Creating coefficients folder in: {outpath}")
        os.makedirs(outpath, exist_ok=True)

def sample_snapshots(initial_snap, final_snap, nsnaps_to_compute_exp):
    snaps_to_compute_exp = np.arange(initial_snap, final_snap+1, 1, dtype=int)
    nsnaps = len(snaps_to_compute_exp)

    assert snaps_to_compute_exp[0] == initial_snap
    assert snaps_to_compute_exp[-1] == final_snap
    if nsnaps_to_compute_exp:
        nsample = round(nsnaps / nsnaps_to_compute_exp)
        snaps_to_compute_exp = snaps_to_compute_exp[::nsample]
    
    nsnaps_sample = len(snaps_to_compute_exp)
    logging.info("Computing coefficients in {} snapshots".format(nsnaps_sample))
    return snaps_to_compute_exp

def load_GC21_exp_center(origin_dir, simulation_filename, suite, component, return_vel=False ):
    """
    Loads the center of the GC21 simulations

    Paramters:
    ----------
    centers_parh : str
        filename with the centers.

    Returns:
    --------
    
    halo_com_pos : np.ndarray, shape (3,N)
    halo_com_vel : np.ndarray, shape (3,N) (optional)

    TODO: This could be skipped by loading once the snapshots and caching the orbit to avoid
    reading at every snapshot.
    
    """
    
    center_file = read_simulations_files(simulation_filename, suite, component, quantity='expansion_center')
    origin_file = os.path.join(origin_dir, center_file)
    if not os.path.isfile(origin_file):
        raise FileNotFoundError(f"> Origins file not found in {origin_file}")

    density_center = np.loadtxt(origin_file)[:,0:3]
    
    if return_vel == True:
        velocity_center = np.loadtxt(center_file)[:,3:6]
        return density_center, velocity_center
    else:
        return density_center

def read_simulations_files(sims_file_path, suite, component, quantity):
    """
    Read a basis lookup table and return the basis filename matching suite & component.

    Parameters
    ----------
    sims_file_path : str
        Path to the text/CSV file containing the lookup table.
    suite : str
        Suite name to match.
    component : str
        Component name to match (e.g. 'MWhaloiso').
	quantity : str
		either basis or expansion_center

    Returns
    -------
    str
        Basis filename associated with the given suite and component.

    Raises
    ------
    ValueError
        If the suite/component pair is not found.
    """
    with open(sims_file_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["suite"].strip() == suite and row["component"].strip() == component:
                return row[quantity].strip()

    raise ValueError(f"No entry found for suite='{suite}' and component='{component}'.")

def load_exp_basis(sim_files, basis_path, component, suite, variance=False):
    """
    Load a basis configuration from a YAML file and initialize a Basis object.

    Parameters
    ----------
    conf_name : str
        Path to the YAML configuration file. If the provided filename does not 
        end with `.yaml`, the extension is automatically appended.

    Returns
    -------
    basis : pyEXP.basis.Basis
        An initialized Basis object created from the configuration.

    Raises
    ------
    FileNotFoundError
        If the specified YAML file does not exist.
    """

    basis_name = read_simulations_files(sim_files, suite, component, quantity="basis")
    basis_file = os.path.join(basis_path, basis_name) 
    if not os.path.isfile(basis_file):
        raise FileNotFoundError("Basis file not found: {}".format(basis_file))

    with open(basis_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Build basis from configuration
    basis = pyEXP.basis.Basis.factory(config)

    if variance == True:
        logging.info("Enabling coefficients variance computation")
        basis.enableCoefCovarince(True, 100)
    elif type(variance) != bool:
        loggin.error("variance must be a boolean")
    return basis
