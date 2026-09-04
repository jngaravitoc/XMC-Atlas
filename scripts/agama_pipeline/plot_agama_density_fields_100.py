"""
Load the AGAMA-BFE density fields computed by agama_density_field.py for
the 100-sim sample (submit_agama_density_fields_100.sh) and make one QA
figure per simulation: a component-rows x snapshot-cols grid of
log10(density), mean-projected along z -- same layout/conventions as
particle_fields/plot_density_fields_100.py, but for the AGAMA side (4
components -- MWhalo, MWbulge, MWdisk, LMChalo; see
agama_density_field.py's module docstring).

Only reads already-computed HDF5 output (agama_density_field.read_agama_density_field)
-- does no BFE evaluation itself, so this is cheap even across all 100 sims.

Usage:
    scripts/run_atlas_python.sh scripts/agama_pipeline/plot_agama_density_fields_100.py
    scripts/run_atlas_python.sh scripts/agama_pipeline/plot_agama_density_fields_100.py --sim-ids 13 28 49
    scripts/run_atlas_python.sh scripts/agama_pipeline/plot_agama_density_fields_100.py --snapshots 0 25 50 75

Author: github.com/jngaravitoc
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless: no display on cluster nodes

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR))  # agama_pipeline/ -- for agama_density_field
sys.path.append(str(THIS_DIR.parent))  # scripts/ -- for config

import plot_config  # noqa: F401 -- applies XMC-Atlas.mplstyle (text.usetex=True)
import agama_density_field as adf
from config import AGAMA_EXPANSIONS_PATH, FIGURES_PATH

COMPONENTS = adf.AGAMA_COMPONENTS  # ["MWhalo", "MWbulge", "MWdisk", "LMChalo"]
SNAPSHOTS = [0, 50, 90]

# Same 100-sim sample used for submit_agama_density_fields_100.sh (and, the
# original AGAMA-coefficients batch, job 9725): first 100 lines of
# sim_ids_sheng_iso.txt (this script already lives in agama_pipeline/).
DEFAULT_SIM_IDS_FILE = THIS_DIR / "sim_ids_sheng_iso.txt"


def load_default_sim_ids(n: int = 100) -> list[int]:
    with open(DEFAULT_SIM_IDS_FILE) as f:
        return [int(line) for line in f.read().splitlines()[:n]]


def plot_sim_agama_density_fields(
    sim_id: int,
    snapshots: Sequence[int] = SNAPSHOTS,
    components: Sequence[str] = COMPONENTS,
    input_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """
    Load one sim's AGAMA-BFE density fields and make a component-rows x
    snapshot-cols grid figure -- same layout/conventions as
    particle_fields/plot_density_fields_100.py's plotting: log10(density)
    mean-projected along z, colorbar fixed per component (row) across the
    plotted snapshots, bins/z-depth shown in each panel's title.

    Parameters
    ----------
    sim_id : int
    snapshots : sequence of int, optional
        Defaults to :data:`SNAPSHOTS` (0, 50, 90).
    components : sequence of str, optional
        Defaults to all 4 AGAMA components (:data:`COMPONENTS`).
    input_path : Path, optional
        HDF5 file to read. Defaults to
        ``config.AGAMA_EXPANSIONS_PATH / f"agama_density_{sim_id:04d}.h5"``
        (agama_density_field.py's default output path).
    output_dir : Path, optional
        Directory to save the figure into. Defaults to
        ``config.FIGURES_PATH / "agama_density_fields_100"``.

    Returns
    -------
    Path or None
        The saved figure path, or None if ``input_path`` doesn't exist (the
        sim hasn't been computed yet, or the batch job failed for it --
        skipped with a warning rather than raising, since this is meant to
        run across many sims where a few missing/failed ones shouldn't stop
        the rest).
    """
    input_path = input_path or AGAMA_EXPANSIONS_PATH / f"agama_density_{sim_id:04d}.h5"
    output_dir = output_dir or FIGURES_PATH / "agama_density_fields_100"

    if not Path(input_path).exists():
        print(f"[skip] sim_id={sim_id}: {input_path} not found")
        return None

    results = {}
    for component in components:
        for nsnap in snapshots:
            snapshot_label = f"snapshot_{nsnap:03d}"
            try:
                results[(component, nsnap)] = adf.read_agama_density_field(
                    input_path, component, snapshot_label
                )
            except KeyError:
                print(f"[skip] sim_id={sim_id}, component={component}: "
                      f"{snapshot_label} not found in {input_path}")

    if not results:
        return None

    # Precompute log10 projections and each component's colorbar range
    # across the plotted snapshots, so every panel in a row shares the same
    # vmin/vmax (same reasoning as plot_density_fields_100.py). Unlike KDE
    # (a sum of positive Gaussian kernels, so always well-behaved), a
    # truncated BFE isn't guaranteed non-negative and can ring numerically
    # right at/near r=0 -- confirmed on Model_13 MWhalo, where 0.08% of
    # cells (near the polar axis) reach ~1e11-1e12 while the rest of the
    # grid is many orders of magnitude smaller. True min/max would let those
    # handful of cells set the whole row's color scale, washing out
    # everything else to a single flat color. Using the 1st-99th percentile
    # instead (like clipping outliers before display) keeps the real
    # dynamic range visible; the outlier cells just saturate to the
    # edge color rather than dominating. np.log10 of a non-positive density
    # (also possible for a BFE, unlike KDE) is NaN, so use the NaN-aware
    # percentile.
    projections = {
        key: np.log10(r["density"].mean(axis=2)) for key, r in results.items()
    }
    component_vrange = {}
    for component in components:
        proj_list = [
            projections[(component, nsnap)] for nsnap in snapshots
            if (component, nsnap) in projections
        ]
        if proj_list:
            stacked = np.concatenate([p.ravel() for p in proj_list])
            if np.all(np.isnan(stacked)):
                continue
            component_vrange[component] = tuple(np.nanpercentile(stacked, [1, 99]))

    n_rows, n_cols = len(components), len(snapshots)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows), squeeze=False)

    for row, component in enumerate(components):
        if component not in component_vrange:
            continue
        vmin, vmax = component_vrange[component]
        for col, nsnap in enumerate(snapshots):
            ax = axes[row, col]
            if (component, nsnap) not in results:
                ax.axis("off")
                continue
            r = results[(component, nsnap)]

            z_depth = r["z_edges"][-1] - r["z_edges"][0]
            proj = projections[(component, nsnap)]
            extent = [r["x_edges"][0], r["x_edges"][-1], r["y_edges"][0], r["y_edges"][-1]]

            im = ax.imshow(
                proj.T, origin="lower", extent=extent, aspect="equal", cmap="cubehelix",
                vmin=vmin, vmax=vmax,
            )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$\log_{10}$ density [M$_\odot$/kpc$^3$]")

            bins = r["grid_bins"]
            ax.set_title(
                f"{component} (AGAMA BFE, bins={bins}), snapshot {nsnap:03d}\n"
                f"mean over z-depth={z_depth:.0f} kpc"
            )
            ax.set_xlabel("x [kpc]")
            if col == 0:
                ax.set_ylabel("y [kpc]")

    fig.suptitle(f"sim_id={sim_id}")
    fig.tight_layout()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    snap_tag = "_".join(f"{s:03d}" for s in snapshots)
    output_path = output_dir / f"agama_density_{sim_id:04d}_snapshots_{snap_tag}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the AGAMA-BFE density fields computed for the 100-sim sample."
    )
    parser.add_argument(
        "--sim-ids", type=int, nargs="+", default=None,
        help="Sim IDs to plot (default: the 100-sim sample from "
        f"{DEFAULT_SIM_IDS_FILE.name}).",
    )
    parser.add_argument(
        "--snapshots", type=int, nargs="+", default=SNAPSHOTS,
        help=f"Snapshot indices to plot as columns (default: {SNAPSHOTS}).",
    )
    parser.add_argument("--components", type=str, nargs="+", default=COMPONENTS)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    sim_ids = args.sim_ids if args.sim_ids is not None else load_default_sim_ids()
    output_dir = Path(args.output_dir) if args.output_dir else None

    n_written = 0
    for sim_id in sim_ids:
        output_path = plot_sim_agama_density_fields(
            sim_id, snapshots=args.snapshots, components=args.components, output_dir=output_dir,
        )
        if output_path is not None:
            print(f"[ok] sim_id={sim_id} -> {output_path}")
            n_written += 1

    print(f"\nWrote {n_written}/{len(sim_ids)} figures.")


if __name__ == "__main__":
    main()
