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
AGAMA_EXPANSIONS_PATH = PROJECT_DIR / "scripts/agama_expansions"
AGAMA_ORBITS_PATH = PROJECT_DIR / "scripts/agama_orbits"
PARTICLE_FIELDS_PATH = PROJECT_DIR / "scripts/particle_fields_data"
VALIDATION_PATH = PROJECT_DIR / "scripts/validation_data"
SIMULATIONS_PATH = SIMS_DIR
SIMS_PARAMS_PATH = PROJECT_DIR/ "suites/Sheng24/orbits"
TEMP_DATA_PATH = PROJECT_DIR / "temp_data"
FIGURES_PATH = PROJECT_DIR / "temp_figures"

# Simulation suite/model directories under SIMS_DIR.
# Update these subfolder names if your on-disk naming differs.
SIMULATION_SUITE_PATHS = {
	"sheng": {
		"sheng_iso": SIMS_DIR / "Sheng" / "LHC_iso_low",
		"sheng_radial": SIMS_DIR / "Sheng" / "LHC_radial_low",
	},
	"gc21": {
		"3b0": SIMS_DIR / "GC21" / "MWLMC3_b0",
		"3b1": SIMS_DIR / "GC21" / "MWLMC3_b1",
		"4b0": SIMS_DIR / "GC21" / "MWLMC4_b0",
		"4b1": SIMS_DIR / "GC21" / "MWLMC4_b1",
		"5b0": SIMS_DIR / "GC21" / "MWLMC5_b0",
		"5b1": SIMS_DIR / "GC21" / "MWLMC5_b1",
		"6b0": SIMS_DIR / "GC21" / "MWLMC6_b0",
		"6b1": SIMS_DIR / "GC21" / "MWLMC6_b1",
	},
}


def get_simulation_path(suite, model):
	"""Return the configured simulation folder for a suite/model pair."""
	suite_key = suite.lower()
	if suite_key not in SIMULATION_SUITE_PATHS:
		available = ", ".join(sorted(SIMULATION_SUITE_PATHS))
		raise KeyError(f"Unknown simulation suite '{suite}'. Available: {available}")

	models = SIMULATION_SUITE_PATHS[suite_key]
	if model not in models:
		available = ", ".join(sorted(models))
		raise KeyError(
			f"Unknown model '{model}' for suite '{suite}'. Available: {available}"
		)

	return models[model]
    
# Create directories if needed
