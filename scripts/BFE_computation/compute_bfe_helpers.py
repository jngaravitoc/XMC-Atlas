import os
import csv
import yaml
import pyEXP
import logging


def setup_logger(logfile="bfe_computation.log"):
    logging.basicConfig(
        filename=logfile,
        filemode="w",                     # overwrite each run; use "a" to append
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO                # or DEBUG for more detail
    )


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

def load_config_file(config_file):
    """
    Load YAML configuration parameters.
    
    Paramters:
    ----------

    config_file : str
        yaml file name

    Returns:
    --------
    
    dictionary

    """
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def load_basis(basis_path, component, suite):
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

    basis_file = read_simulations_files(basis_path, suite, component, quantity="basis")
    if not os.path.exists(basis_file):
        raise FileNotFoundError(f"Basis file not found: {halo_basis_yaml}")

    with open(basis_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Build basis from configuration
    basis = pyEXP.basis.Basis.factory(config)
    return basis
