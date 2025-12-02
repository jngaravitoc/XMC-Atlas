import yaml
import pytest
from compute_bfe_helpers import load_config_file


def test_load_config(tmp_path):
    # -------------------------------------------------------------
    # Create a temporary YAML file using the example provided
    # -------------------------------------------------------------
    config_text = """
paths: 
  snapshot_dir: "/n/nyx3/garavito/XMC-Atlas-sims/GC21/MW/out/"
  output_dir: "/n/nyx3/garavito/projects/XMC-Atlas/GC21/coefficients/" 
  coefficients_filename: "test_coefs_GC21_MWLMC5_DM_halo_10_8_wunits.h5"
  expansion_center_dir: "/n/nyx3/garavito/projects/XMC-Atlas/scripts/output/MW/"

simulations:
  snapname: MW_100M_beta1_vir_OM3_G4
  component: MWHaloiso
  initial_snap: 0
  final_snap: 1
  nsnaps_to_compute_exp: null
  npart_per_snapshot: 1000000
  exp_center: /n/nyx3/garavito/projects/XMC-Atlas/scripts/output/MW/

exp:
  basis_paths: /n/nyx3/garavito/projects/XMC-Atlas/GC21/basis/
  compute_bfe_variance: false
  units: "gadget"

agama:
  rmax_exp: null
  rmax_sel: null
  pole_l: null
  sym: null
"""

    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_text)

    # -------------------------------------------------------------
    # Load using your load_config function
    # -------------------------------------------------------------
    cfg = load_config_file(config_file)

    # -------------------------------------------------------------
    # Assertions: top-level keys
    # -------------------------------------------------------------
    assert "paths" in cfg
    assert "simulations" in cfg
    assert "exp" in cfg
    assert "agama" in cfg

    # -------------------------------------------------------------
    # Paths validation
    # -------------------------------------------------------------
    paths = cfg["paths"]
    assert paths["snapshot_dir"].endswith("/MW/out/")
    assert "coefficients_filename" in paths

    # -------------------------------------------------------------
    # Simulation validation
    # -------------------------------------------------------------
    sims = cfg["simulations"]
    assert sims["snapname"] == "MW_100M_beta1_vir_OM3_G4"
    assert sims["component"] == "MWHaloiso"
    assert sims["initial_snap"] == 0
    assert sims["final_snap"] == 1
    assert sims["nsnaps_to_compute_exp"] is None

    # -------------------------------------------------------------
    # Choose expansion type: EXP vs AGAMA
    # -------------------------------------------------------------
    assert cfg["expansion_type"] == "EXP"

    # -------------------------------------------------------------
    # EXP configuration
    # -------------------------------------------------------------
    exp_cfg = cfg["exp"]
    assert exp_cfg["units"] == "gadget"
    assert exp_cfg["compute_bfe_variance"] is False

    # -------------------------------------------------------------
    # AGAMA configuration still exists but unused
    # -------------------------------------------------------------
    agama_cfg = cfg["agama"]
    assert agama_cfg["rmax_exp"] is None
    assert agama_cfg["sym"] is None

