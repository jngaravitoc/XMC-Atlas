#!/usr/bin/env python3
"""
Generate density-comparison dashboard figures from pre-computed fields.

This script reads BFE and KDE density fields previously written by
``compute_fields.py`` and produces per-snapshot dashboard figures
comparing the two, along with MISE / MIRSE error metrics.
Optionally creates an MP4 video from the generated frames.

Usage
-----
    python make_dashboard.py --halo_id 108 --output_dir ./output
    python make_dashboard.py --halo_id 108 --output_dir ./output --rvir 270
    python make_dashboard.py --halo_id 108 --output_dir ./output --make_video --fps 15
"""

import sys
import os
import argparse
import glob

import numpy as np
import matplotlib.pyplot as plt

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

# Add sibling directories to path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
sys.path.append(os.path.join(_THIS_DIR, "../exp_pipeline/"))
sys.path.append(os.path.join(_THIS_DIR, "../exp_fields/"))

from field_io import read_fields, read_merged_kde_density
from plot_helpers import density_dashboard
from metrics import mise, mirse

import h5py


# ------------------------------------------------------------------
# Video helper
# ------------------------------------------------------------------

def create_video_from_images(halo_id, output_dir, fps=10, video_filename=None):
    """Create an MP4 video from saved dashboard PNG files.

    Parameters
    ----------
    halo_id : int
    output_dir : str
    fps : int, optional
    video_filename : str, optional
    """
    if not HAS_IMAGEIO:
        print("\nWarning: imageio not installed.  Skipping video creation.")
        print("Install with: pip install imageio")
        return None

    pattern = os.path.join(output_dir, f"halo_{halo_id:04d}_density_field_*.png")
    png_files = sorted(glob.glob(pattern))

    if not png_files:
        print(f"\nWarning: No PNG files found matching {pattern}")
        return None

    print(f"\nCreating video from {len(png_files)} frames...")

    if video_filename is None:
        video_filename = os.path.join(
            output_dir, f"halo_{halo_id:04d}_density_evolution.mp4"
        )
    else:
        video_filename = os.path.join(output_dir, video_filename)

    try:
        images = [imageio.imread(f) for f in png_files]
        imageio.mimsave(video_filename, images, fps=fps, codec="libx264")
        print(f"Video created successfully: {video_filename}")
        return video_filename
    except Exception as e:
        print(f"Error creating video: {e}")
        return None


# ------------------------------------------------------------------
# Dashboard generation
# ------------------------------------------------------------------

def generate_dashboards(halo_id, output_dir, rvir=270,
                        grid_bins=20, dpi=300,
                        make_video=False, fps=10):
    """Generate density dashboard figures from pre-computed field files.

    Expects the following files to exist inside *output_dir*:

    * ``halo_XXXX_BFE_fields.h5``  — written by ``compute_fields.py``
    * ``halo_XXXX_kde_density.h5`` — merged KDE file

    Parameters
    ----------
    halo_id : int
    output_dir : str
    rvir : float, optional
    grid_bins : int, optional
    dpi : int, optional
    make_video : bool, optional
    fps : int, optional
    """
    os.makedirs(output_dir, exist_ok=True)

    bfe_file = os.path.join(output_dir, f"halo_{halo_id:04d}_BFE_fields.h5")
    kde_file = os.path.join(output_dir, f"halo_{halo_id:04d}_kde_density.h5")

    # Validate inputs
    if not os.path.isfile(bfe_file):
        raise FileNotFoundError(
            f"BFE fields file not found: {bfe_file}\n"
            "Run compute_fields.py first."
        )
    if not os.path.isfile(kde_file):
        raise FileNotFoundError(
            f"Merged KDE density file not found: {kde_file}\n"
            "Run compute_fields.py first."
        )

    # Read time keys from BFE file
    with h5py.File(bfe_file, "r") as f:
        time_keys = sorted(f.keys(), key=float)
        grid_shape = tuple(f.attrs.get("grid_shape", ()))
    nbins = grid_shape[0] if grid_shape else grid_bins

    # Read all KDE snapshots
    kde_data, _kde_attrs = read_merged_kde_density(kde_file)
    kde_snapshots = sorted(kde_data.keys())

    n_snaps = min(len(time_keys), len(kde_snapshots))
    print(f"Generating dashboards for {n_snaps} snapshots...")

    mise_dens_arr = np.zeros(n_snaps)
    mise_logdens_arr = np.zeros(n_snaps)
    mirse_dens_arr = np.zeros(n_snaps)

    for i in range(n_snaps):
        t_key = time_keys[i]
        t_val = float(t_key)
        snap_key = kde_snapshots[i]

        # Read BFE density for this snapshot
        dens_bfe = read_fields(bfe_file, "dens", t_key)
        if dens_bfe.ndim == 1:
            dens_bfe = dens_bfe.reshape(nbins, nbins, nbins)

        # KDE density
        kd_dens = kde_data[snap_key]

        # Error metrics
        mise_val = mise(dens_bfe, kd_dens, axis=0)
        mise_log = mise(np.log10(dens_bfe), np.log10(kd_dens), axis=0)
        mirse_val = mirse(dens_bfe, kd_dens, axis=0)

        mise_dens_arr[i] = np.median(mise_val)
        mise_logdens_arr[i] = np.median(mise_log)
        mirse_dens_arr[i] = np.median(mirse_val)

        # Plot
        fig = density_dashboard(
            kd_dens, dens_bfe,
            mise_log, mirse_val,
            mean_axis=0, rvir=rvir,
        )
        fig.suptitle(f"Halo {halo_id:04d}; Time = {t_val:.2f} Gyr", fontsize=12)

        out_png = os.path.join(
            output_dir,
            f"halo_{halo_id:04d}_density_field_center_{i:03d}.png",
        )
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {os.path.basename(out_png)}")

    # Save error metrics
    mises_file = os.path.join(output_dir, f"halo_{halo_id:04d}_mises.npy")
    np.save(
        mises_file,
        np.column_stack([mise_dens_arr, mise_logdens_arr, mirse_dens_arr]),
    )
    print(f"Error metrics saved to {os.path.basename(mises_file)}")

    # Optional video
    if make_video:
        create_video_from_images(halo_id, output_dir, fps=fps)

    print("\nDashboards generated successfully!")
    print(f"Output files in: {output_dir}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate density dashboard figures from pre-computed fields",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python make_dashboard.py --halo_id 108 --output_dir ./output
  python make_dashboard.py --halo_id 108 --output_dir ./output --rvir 270 --dpi 150
  python make_dashboard.py --halo_id 108 --output_dir ./output --make_video --fps 15
        """,
    )
    parser.add_argument("--halo_id", type=int, required=True,
                        help="Halo model ID (e.g. 108)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory containing field HDF5 files / output PNGs")
    parser.add_argument("--rvir", type=float, default=270,
                        help="Virial radius in kpc (default: 270)")
    parser.add_argument("--grid_bins", type=int, default=20,
                        help="Number of grid bins (default: 20)")
    parser.add_argument("--dpi", type=int, default=300,
                        help="DPI for saved figures (default: 300)")
    parser.add_argument("--make_video", action="store_true",
                        help="Create an MP4 video from the dashboard frames")
    parser.add_argument("--fps", type=int, default=10,
                        help="Frames per second for video (default: 10)")
    args = parser.parse_args()

    generate_dashboards(
        halo_id=args.halo_id,
        output_dir=args.output_dir,
        rvir=args.rvir,
        grid_bins=args.grid_bins,
        dpi=args.dpi,
        make_video=args.make_video,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
