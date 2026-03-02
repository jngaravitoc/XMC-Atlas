# Scripts

Code for building basis function expansions (BFE), computing coefficients,
validating density reconstructions, and generating diagnostic dashboards for
the XMC-Atlas simulations.

---

## Pipeline overview

1. **Build a basis** from the initial N-body snapshot density profile.
2. **Compute BFE coefficients** for every snapshot (optionally MPI-parallel).
3. **Validate** the reconstruction against KDE densities (MISE / MIRSE metrics).
4. **Generate dashboards** (density maps, error panels, videos).

---

## Top-level scripts

| Script | Description |
|--------|-------------|
| `build_spherical_basis.py` | Build an optimal spherical (EXP) basis for Sheng+24 halos. |
| `build_agama_expansion.py` | Build an optimal Agama-based basis expansion for Sheng+24 halos. |
| `compute_basis.py` | Compute EXP BFE coefficients from a YAML config (load basis, snapshots, write coefs). |
| `compute_bfe.py` | Compute EXP BFE coefficients (alternative entry point). |
| `compute_MW_orbit.py` | Compute MW centre-of-mass orbit using NBA and pynbody methods. |
| `compute_density_profile.py` | Compute density profiles of MW halo/bulge after COM recentering. |
| `density_dashboard.py` | Compute BFE fields on a grid and produce KDE vs BFE density comparison dashboards. |
| `spherical_basis_density_dashboards.py` | CLI tool: generate density dashboard figures and optional MP4 video for a given halo. |
| `halos_bfe_from_same_basis.py` | Compute BFE for multiple halos sharing a common basis (MPI-parallel). |
| `generate_surface_density_maps.py` | Generate xy/xz surface density maps for the LMC disk (cylindrical basis). |
| `agama_acceleration_errors.py` | Assess Agama potential acceleration errors at random particle positions. |

### Shell helpers

| Script | Description |
|--------|-------------|
| `compute_agama_exp.sh` | Loop over halo IDs and run `build_agama_expansion.py` for each. |
| `run_dashboard.sh` | Run `spherical_basis_density_dashboards.py` in parallel for multiple halos. |

### Notebooks

| Notebook | Description |
|----------|-------------|
| `com_smooth.ipynb` | Centre-of-mass smoothing exploration. |
| `draft_bfe_pipeline.ipynb` | Prototyping notebook for the BFE pipeline. |
| `test_dashboard.ipynb` | Interactive testing of dashboard functions and field I/O. |

---

## `exp_pipeline/` — core library

Reusable modules imported by the top-level scripts.

| Module | Description |
|--------|-------------|
| `field_projections.py` | `FieldProjections` class — grid/mesh setup, BFE field evaluation, KDE density. |
| `field_io.py` | `write_fields` / `read_fields` — HDF5 I/O for BFE field dictionaries. |
| `plot_helpers.py` | `plot_profiles`, `density_dashboard` — pure matplotlib plotting functions. |
| `basis_utils.py` | Load, validate, and write pyEXP basis YAML configs; create basis from model. |
| `basis_fidelity.py` | Spherically-averaged BFE density profiles and radial MISE fidelity. |
| `compute_bfe_helpers.py` | Snapshot listing, centre loading, sim file reading helpers. |
| `exp_coefficients.py` | Serial and MPI-parallel EXP coefficient computation. |
| `expansion_units.py` | Unit system definitions for EXP and Agama. |
| `fit_density.py` | Fit analytic density profiles (power-law halo) to particle data. |
| `grid.py` | `Grid3D` class — Cartesian / spherical / cylindrical grid generation. |
| `ios_nbody_sims.py` | `LoadSim`, `load_particle_data` — load XMC-Atlas simulation snapshots. |
| `makemodel.py` | Write model table files and create EXP models from density profiles. |
| `metrics.py` | MISE, MIRSE, KDTree density, and grid-based error metrics. |
| `sanity_checks.py` | Validate profile monotonicity and snapshot contiguity. |
| `data_products.py` | Write/append density profiles to HDF5. |
| `build_basis_helpers.py` | GC21 MW halo driver, config writing, profile plotting. |
| `disk_basis.py` | Generate the cylindrical MW disk basis (MPI-parallel). |
| `test_compute_bfe.py` | Pytest tests for BFE computation helpers. |

---

## `agama_pipeline/` — Agama BFE pipeline

| Module | Description |
|--------|-------------|
| `agama_BFEs.py` | Agama Multipole/CylSpline BFE fitting; HDF5 coefficient I/O; RAM-backed potential reconstruction. |
| `agama_coefficients.py` | Create Gizmo-like snapshots and fit Agama potentials; benchmark routines. |
| `agama_external_sims.py` | Fit Agama potentials from external (non-XMC) simulation data. |
| `agama_acceleration_errors.py` | Acceleration-error assessment (duplicate of top-level script). |

---

## `validation/` — BFE validation

| Module | Description |
|--------|-------------|
| `metrics.py` | Comprehensive fidelity metrics: MISE, MIRSE, goodness-of-fit, radial profiles, projected maps. |
| `fields.py` | Compute and plot 2-D BFE density reconstructions with LMC orbit overlay. |
| `fields_helpers.py` | Spherical grid, BFE density profiles, 2-D field computation helpers. |
| `pointmesh_pyexp_test.py` | Test pyEXP point-mesh field evaluation on a 2-D grid. |

---

## `basis_computation/` — standalone basis construction

Older / standalone scripts for building bases (some overlap with `exp_pipeline/`).

| Module | Description |
|--------|-------------|
| `build_basis.py` | Build bases for GC21 MW halo and Sheng+24 sims. |
| `basis_utils.py` | Load/write/validate pyEXP basis YAML configs. |
| `build_basis_helpers.py` | Write basis config files; MPI basis construction helpers. |
| `density_fit_arpit.py` | Double power-law density fitting with uneven grids. |
| `disk_basis.py` | Generate MW cylindrical disk basis (MPI-parallel). |

---

## Other subdirectories

| Directory | Contents |
|-----------|----------|
| `BFE_computation/` | `EXP_external_sims.py` — compute BFE coefficients for external simulations. |
| `COM_test/` | `com_comparison.py` — compare centre-of-mass methods (NBA, pynbody). |
| `disbatch_scripts/` | `make_disbatch_tasks.sh` — generate disBatch task files for MW orbit jobs. |
| `galaxy_models/` | Jupyter notebooks for TNG halo bases, disk bases, and data reading tests. |
| `test_sheng24/` | YAML configs, cached basis/coefficient HDF5 files (data, no scripts). |
| `orbit_integration/` | (empty) |
| `exp_expansions/` | Computed basis files, coefficient outputs, and field data. |

