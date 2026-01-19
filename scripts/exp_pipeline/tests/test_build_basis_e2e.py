import subprocess
import sys
import os
from pathlib import Path

import numpy as np
import yaml
import pytest

# unmcoment this if this ever becomes a package!
#@pytest.mark.e2e
def test_build_basis_halo_0108(tmp_path):
    """
    End-to-end test for build_basis.py.

    This test:
    1. Runs: python build_basis.py 108 halo
    2. Verifies that all expected files are created in /test_sheng24
    3. Validates the contents of the YAML basis file
    4. Loads the coefficient HDF5 file with pyEXP and checks:
       - coefficient array shape
       - Power() output
       - Times() output
    """

    # ------------------------------------------------------------------
    # Arrange: define paths
    # ------------------------------------------------------------------
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "build_spherical_basis.py"

    assert script_path.exists(), "build_basis.py not found at repo root"

    scripts_dir = repo_root 

    workdir = scripts_dir
    output_dir = scripts_dir / "test_sheng24_108"
    
    # ------------------------------------------------------------------
    # Act: run the script
    # ------------------------------------------------------------------
    
    cmd = [
	sys.executable,
    str(script_path),
    "108",           # first positional argument
    "halo",          # second positional argument
    "--ncoefs", "10", #third positional argument
    "--fit-type", "initial",  # optional flag with value
    #"--compute-mise", "False",     # optional boolean flag
	"--output_dir", "test_sheng24_108"
    ]

    subprocess.run(
        cmd,
        cwd=workdir,
        check=True,
    )
    
    # ------------------------------------------------------------------
    # Assert: expected files exist
    # ------------------------------------------------------------------
    expected_files = [
        "basis_halo_0108.yaml",
        "bfe_halo_0108_density_profiles_sheng24.h5",
        "halo_0108_density_profiles_sheng24.h5",
        "cache_halo_0108.txt",
        "modelname_halo_0108.txt",
        "halo_0108_coefficients.h5",
    ]

    for fname in expected_files:
        fpath = output_dir / fname
        assert fpath.exists(), f"Missing expected file: {fname}"

    # ------------------------------------------------------------------
    # Assert: YAML file content
    # ------------------------------------------------------------------
    yaml_file = output_dir / "basis_halo_0108.yaml"
    with open(yaml_file) as f:
        basis = yaml.safe_load(f)

    expected_yaml = {
        "id": "sphereSL",
        "parameters": {
            "numr": 1000,
            "rmin": 0.1,
            "rmax": 500,
            "Lmax": 1,
            "nmax": 10,
            "rmapping": 1.0,
            "modelname": "modelname_halo_0108.txt",
            "cachename": "cache_halo_0108.txt",
        },
    }

    assert basis == expected_yaml

    # ------------------------------------------------------------------
    # Assert: coefficient file via pyEXP
    # ------------------------------------------------------------------
    import pyEXP

    coef_file = output_dir / "halo_0108_coefficients.h5"
    coefs = pyEXP.coefs.Coefs.factory(str(coef_file))

    # -- getAllCoefs shape
    all_coefs = coefs.getAllCoefs()
    assert all_coefs.shape == (3, 10, 11)

    # -- Power()
    expected_power = np.array(
        [
            [7.60876888e02, 4.64057692e-03],
            [7.60259401e02, 3.04282229e-02],
            [7.59922961e02, 2.66688575e-01],
            [7.58942161e02, 9.83373248e-01],
            [7.57140971e02, 2.87211569e00],
            [7.53041425e02, 6.94142882e00],
            [7.45225316e02, 1.46885336e01],
            [7.31392921e02, 2.72377281e01],
            [7.08888739e02, 4.66039492e01],
            [6.76353293e02, 7.21568392e01],
            [6.35691477e02, 1.03410675e02],
        ]
    )

    np.testing.assert_allclose(
        coefs.Power(),
        expected_power,
        rtol=1e-5,
        atol=0.0,
    )

    # -- Times()
    expected_times = np.array(
        [
            0.0,
            0.203125,
            0.3984375,
            0.6015625,
            0.796875,
            1.0,
            1.203125,
            1.3984375,
            1.6015625,
            1.796875,
            2.0,
        ]
    )

    np.testing.assert_allclose(
        coefs.Times(),
        expected_times,
        rtol=0.0,
        atol=0.0,
    )

