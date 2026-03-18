from pathlib import Path
import os

# Add this paths in your ~/.bashrc file as:
# export XMC_ATLAS_SIMS="/path/to/sims_folder/"
# Base directory
PROJECT_DIR =  Path(os.environ.get("XMC_ATLAS"))
SIMS_DIR =  Path(os.environ.get("XMC_ATLAS_SIMS"))


# Data directories
DATA_PATH = PROJECT_DIR / "data"
EXP_EXPANSIONS_PATH = PROJECT_DIR / "scripts/exp_expansions"
DENSITY_FIELDS_PATHS = PROJECT_DIR / "scripts/exp_expansions/density_fields"
SIMULATIONS_PATH = SIMS_DIR 
SIMS_PARAMS_PATH = PROJECT_DIR/ "suites/Sheng24/orbits"
TEMP_DATA_PATH = PROJECT_DIR / "temp_data"
FIGURES_PATH = PROJECT_DIR / "temp_figures"


# Create directories if needed
