from pathlib import Path
import os

# Base directory
PROJECT_DIR =  Path(os.environ.get("XMC_ATLAS"))


# Data directories
DATA_PATH = PROJECT_DIR / "data"
EXP_EXPANSIONS_PATH = PROJECT_DIR / "scripts/exp_expansions"
DENSITY_FIELDS_PATHS = PROJECT_DIR / "scripts/exp_expansions/density_fields"

TEMP_DATA_PATH = PROJECT_DIR / "temp_data"
FIGURES_PATH = PROJECT_DIR / "temp_figures"


# Create directories if needed
