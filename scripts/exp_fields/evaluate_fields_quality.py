#!/usr/bin/env python3
"""
Evaluate the quality of BFE field reconstructions for different halo shapes
and save the metrics evolution results to an HDF5 file.
"""
import os
import h5py
import numpy as np
import sys


def read_fields_quality_metrics(filename, halo_id):
    """
    Read metrics evolution results for a given halo from an HDF5 file.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file containing metrics.
    halo_id : int or str
        Identifier for the halo to read.

    Returns
    -------
    dict
        Dictionary of metric arrays for the specified halo.
    """
    import h5py
    metrics = {}
    group_name = f"halo_{halo_id}"
    with h5py.File(filename, "r") as h5f:
        if group_name not in h5f:
            raise KeyError(f"Halo group '{group_name}' not found in file.")
        group = h5f[group_name]
        for key in group.keys():
            metrics[key] = group[key][...]
    return metrics


def read_all_fields_quality_metrics(filename):
    """
    Read metrics evolution results for all halos from an HDF5 file.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file containing metrics.

    Returns
    -------
    dict
        Dictionary mapping halo_id to its metrics dictionary.
    """
    import h5py
    all_metrics = {}
    with h5py.File(filename, "r") as h5f:
        for group_name in h5f:
            metrics = {}
            group = h5f[group_name]
            for key in group.keys():
                metrics[key] = group[key][...]
            # Extract halo_id from group_name (assumes format 'halo_{halo_id}')
            if group_name.startswith("halo_"):
                halo_id = group_name[5:]
            else:
                halo_id = group_name
            all_metrics[halo_id] = metrics
    return all_metrics


def compute_halos_fields_quality(halo_ids):
    # List of halo IDs or shapes to evaluate (replace with actual IDs as needed)
    # Output HDF5 file
    filename = "fields_quality_metrics.h5"
    output_file = os.path.join(DENSITY_FIELDS_PATHS, filename)

    with h5py.File(output_file, "a") as h5f:
        for halo_id in halo_ids:
            fq = FieldsQuality(halo_id)
            metrics = fq.metrics_evolution()
            group = h5f.create_group(f"halo_{halo_id}")
            for key, arr in metrics.items():
                group.create_dataset(key, data=np.asarray(arr))

    print(f"Metrics for halos {halo_ids} written to {output_file}")

if __name__ == "__main__":
    halo = int(sys.argv[1])
    sys.path.append("../")
    from field_projections import FieldsQuality
    from config import DENSITY_FIELDS_PATHS 
    compute_halos_fields_quality(halo_ids=[halo])

