#!/usr/bin/env python3
"""
Generate density dashboard figures for different halo IDs.

This script generates density dashboard figures showing KDE vs BFE reconstructions
for a specified halo and saves them to a user-specified output directory.
Optionally creates an MP4 video from the generated frames.

Usage:
    python generate_density_dashboards.py --halo_id 108 --output_dir ./output
    python generate_density_dashboards.py --halo_id 108 --output_dir ./output --rvir 270
    python generate_density_dashboards.py --halo_id 108 --output_dir ./output --make_video --fps 15
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import glob
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

from density_dashboard import compute_dashboard, compute_bfe_fields
from field_io import write_fields, read_fields
from ios_nbody_sims import load_particle_data
from compute_bfe_helpers import load_sheng24_exp_center
from basis_utils import load_basis

import pyEXP


def create_video_from_images(halo_id, output_dir, fps=10, video_filename=None):
    """
    Create an MP4 video from saved dashboard PNG files.
    
    Parameters
    ----------
    halo_id : int
        The halo model ID.
    output_dir : str
        Directory containing the PNG files.
    fps : int, optional
        Frames per second for the video (default is 10).
    video_filename : str, optional
        Name of the output video file. If None, uses 'halo_XXXX.mp4'.
    """
    
    if not HAS_IMAGEIO:
        print("\nWarning: imageio not installed. Skipping video creation.")
        print("Install with: pip install imageio")
        return
    
    # Find all PNG files matching the pattern
    pattern = os.path.join(output_dir, f"halo_{halo_id:04d}_density_field_*.png")
    png_files = sorted(glob.glob(pattern))
    
    if not png_files:
        print(f"\nWarning: No PNG files found matching {pattern}")
        return
    
    print(f"\nCreating video from {len(png_files)} frames...")
    
    # Default video filename
    if video_filename is None:
        video_filename = os.path.join(output_dir, f"halo_{halo_id:04d}_density_evolution.mp4")
    else:
        video_filename = os.path.join(output_dir, video_filename)
    
    try:
        # Read images and create video
        images = []
        for png_file in png_files:
            img = imageio.imread(png_file)
            images.append(img)
        
        # Write video
        imageio.mimsave(video_filename, images, fps=fps, codec='libx264')
        print(f"Video created successfully: {video_filename}")
        return video_filename
        
    except Exception as e:
        print(f"Error creating video: {e}")
        return None


def compute_fields_in_grid(halo_id, grid_range=(-100, 100), grid_bins=20):
    """
    Load basis and coefficients for a halo, build a 3D grid, and compute
    BFE density fields for every time snapshot.

    This function expects the current working directory to already be
    ``exp_expansions/basis/`` (it does **not** change directories itself).

    Parameters
    ----------
    halo_id : int
        The halo model ID (e.g., 108).
    grid_range : tuple of float, optional
        (min, max) extent of the Cartesian grid in kpc (default ``(-100, 100)``).
    grid_bins : int, optional
        Number of bins along each axis (default 20).

    Returns
    -------
    dens_bfe_list : list
        List of BFE density arrays, one per time snapshot.
    FP : object
        Field-point object returned by ``compute_bfe_fields``.
    times : array-like
        Array of time values for each snapshot.
    points : dict
        Nested dictionary ``{time: {field_name: ndarray, ...}, ...}``
        returned by ``compute_bfe_fields``.
    grid_arrays : list of ndarray
        The 3-D meshgrid arrays (output of ``np.meshgrid``).
    """
    # Load basis
    config_name = f"basis_halo_{halo_id:04d}.yaml"
    print(f"Loading basis from {config_name}...")
    basis = load_basis(config_name)
    print(f"  Basis loaded")

    # Load coefficients
    coefs_file = f"../coefficients/halo_{halo_id:04d}_coefficients_center.h5"
    print(f"Loading coefficients from {coefs_file}...")
    coefs = pyEXP.coefs.Coefs.factory(coefs_file)
    times = coefs.Times()
    print(f"  Found {len(times)} time snapshots")

    # Print power information
    power = coefs.Power()
    print(f"  Power range: {power}")

    # Create grid
    dbins = np.linspace(grid_range[0], grid_range[1], grid_bins)
    grid_arrays = np.meshgrid(dbins, dbins, dbins, indexing='ij')
    grid = np.stack(grid_arrays)
    print(f"Grid created: {grid_bins} x {grid_bins} x {grid_bins}")

    # Compute BFE fields once (already evaluated for all times)
    print(f"\nComputing BFE fields...")
    dens_bfe_list, FP, points = compute_bfe_fields(grid, basis, coefs, times)
    print("Field points computed")

    return dens_bfe_list, FP, times, points, grid_arrays


def generate_dashboards(halo_id, output_dir, suite="Sheng24", rvir=270, 
                        grid_range=(-100, 100), grid_bins=20, dpi=300, make_video=False, fps=10, 
                        write_fields_file=True):
    """
    Generate density dashboard figures for a specified halo.
    
    Parameters
    ----------
    halo_id : int
        The halo model ID (e.g., 108).
    output_dir : str
        Directory where output figures will be saved.
    suite : str, optional
        Simulation suite name (default is 'Sheng24').
    rvir : float, optional
        Virial radius in kpc (default is 270).
    grid_range : tuple, optional
        Min and max values for grid range (default is (-100, 100)).
    grid_bins : int, optional
        Number of bins for grid (default is 20).
    dpi : int, optional
        DPI for saved figures (default is 300).
    make_video : bool, optional
        Whether to create an MP4 video from the frames (default is False).
    fps : int, optional
        Frames per second for video creation (default is 10).
    write_fields_file : bool, optional
        Whether to write the BFE fields to an HDF5 file (default is True).
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Store current directory to restore later
    original_dir = os.getcwd()
    
    try:
        
        # Load sim centers
        print(f"Loading center data...")
        sim_centers = load_sheng24_exp_center(
            origin_dir="../suites/Sheng24/orbits",
            centers_filename="MW_LMC_orbits_iso.txt",
            sim_id=halo_id,
            return_vel=True
        )
        print(f"  Center data loaded")
        
        # Change to expansion directory
        os.chdir("./exp_expansions/basis/")
        
        # Compute density fields
        dens_bfe_list, FP, times, points, grid_arrays = compute_fields_in_grid(
            halo_id, grid_range, grid_bins
        )

        # Write BFE fields to HDF5
        if write_fields_file:
            fields_file = os.path.join(
                original_dir, output_dir,
                f"halo_{halo_id:04d}_BFE_fields.h5"
            )
            write_fields(points, fields_file, grid=grid_arrays[0])
            del points  # free memory after writing
        
        mise_dens = np.zeros(len(times))
        mise_logdens = np.zeros(len(times))
        mirse_dens = np.zeros(len(times))
        # Generate dashboards for each time snapshot
        print(f"\nGenerating dashboards for {len(times)} time snapshots...")
        for i in range(len(times)):
            print(f"Loading particle data for halo {halo_id:04d}...")
            p = load_particle_data(
                f"/n/nyx3/garavito/XMC-Atlas-sims/Sheng/Model_{halo_id:03d}",
                snapname="snapshot",
                components=["MWhalo"],
                nsnap=i,
                suite=suite,
                quantities=["pos", "mass", "pot"]
            )
            print(f"  Loaded MWhalo particles")

            kde_output_file = None
            if write_fields_file:
                kde_output_file = os.path.join(
                    original_dir,
                    output_dir,
                    f"halo_{halo_id:04d}_kde_density_{i:03d}.h5"
                )

            fig, m1, m2, m3 = compute_dashboard(
                FP=FP,
                dens_bfe=dens_bfe_list[i],
                pos=p['MWhalo']['pos']-sim_centers["mw_center"][i],
                mass=p['MWhalo']['mass'],
                rvir=rvir,
                return_mises = True,
                kde_filename=kde_output_file,
                snapshot_name=f"snapshot_{i:03d}",
            )
            
            mise_dens[i] = np.median(m1)
            mise_logdens[i] = np.median(m2) 
            mirse_dens[i] = np.median(m3) 
            # Add title
            fig.suptitle(f"Halo {halo_id:04d}; Time = {times[i]:.2f} Gyr", fontsize=12)
            
            # Save figure
            output_file = os.path.join(
                original_dir,
                output_dir,
                f"halo_{halo_id:04d}_density_field_center_{i:03d}.png"
            )
            
            fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            print(f"saved to {os.path.basename(output_file)}")
        output_file_mises = os.path.join(
            original_dir,
            output_dir,
            f"halo_{halo_id:04d}_mises.npy"
        )
        np.save(output_file_mises, np.array([mise_dens, mise_logdens, mirse_dens]).T)
        print(f"\nDashboards generated successfully!")
        print(f"Output files saved to: {os.path.join(original_dir, output_dir)}")
        
        # Create video if requested
        if make_video:
            video_path = create_video_from_images(halo_id, os.path.join(original_dir, output_dir), fps=fps)
        
    finally:
        # Restore original directory
        os.chdir(original_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Generate density dashboard figures for halo IDs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_density_dashboards.py --halo_id 108 --output_dir ./dashboards
  python generate_density_dashboards.py --halo_id 108 --output_dir ./out --rvir 270 --dpi 150
  python generate_density_dashboards.py --halo_id 108 --output_dir ./out --make_video --fps 15
  python generate_density_dashboards.py --halo_id 108 --output_dir ./out --make_video --fps 8
        """
    )
    
    parser.add_argument(
        '--halo_id',
        type=int,
        required=True,
        help='Halo model ID (e.g., 108)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for saving figures'
    )
    
    parser.add_argument(
        '--suite',
        type=str,
        default='Sheng24',
        help='Simulation suite name (default: Sheng24)'
    )
    
    parser.add_argument(
        '--rvir',
        type=float,
        default=270,
        help='Virial radius in kpc (default: 270)'
    )
    
    parser.add_argument(
        '--grid_range',
        type=float,
        nargs=2,
        default=[-100, 100],
        metavar=('MIN', 'MAX'),
        help='Grid range in kpc (default: -100 100)'
    )
    
    parser.add_argument(
        '--grid_bins',
        type=int,
        default=20,
        help='Number of grid bins (default: 20)'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for saved figures (default: 300)'
    )
    
    parser.add_argument(
        '--make_video',
        action='store_true',
        help='Create an MP4 video from the generated frames'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=10,
        help='Frames per second for video (default: 10)'
    )
    
    parser.add_argument(
        '--no_write_fields',
        action='store_true',
        help='Skip writing the BFE fields HDF5 file'
    )
    
    args = parser.parse_args()
    
    generate_dashboards(
        halo_id=args.halo_id,
        output_dir=args.output_dir,
        suite=args.suite,
        rvir=args.rvir,
        grid_range=tuple(args.grid_range),
        grid_bins=args.grid_bins,
        dpi=args.dpi,
        make_video=args.make_video,
        fps=args.fps,
        write_fields_file=not args.no_write_fields
    )


if __name__ == "__main__":
    main()



