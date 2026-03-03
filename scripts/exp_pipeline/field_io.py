#!/usr/bin/env python3
"""
HDF5 I/O utilities for BFE field dictionaries.
"""

import numpy as np
import h5py


def write_fields(points, filename, grid=None, field_shape=None):
    """
    Write a ``fields.points`` dictionary to an HDF5 file.

    The dictionary is expected to have the structure returned by
    ``pyEXP.field.FieldGenerator.points``::

        {
            time_0: {field_name: ndarray, ...},
            time_1: {field_name: ndarray, ...},
            ...
        }

    Each time snapshot is stored as a separate HDF5 group named after its
    time value, and every field array is stored as a dataset inside that
    group.

    Parameters
    ----------
    points : dict
        Nested dictionary ``{time: {field_name: ndarray, ...}, ...}``.
    filename : str
        Path to the output HDF5 file.
    grid : array-like, optional
        The grid used to compute the fields (e.g. one array from
        ``np.meshgrid``).  If provided, its shape is stored in the file
        header attribute ``grid_shape``.
    field_shape : tuple of int, optional
        Explicit shape to record in the file header attribute
        ``field_shape``.  Ignored when *grid* is also given.
    """
    with h5py.File(filename, "w") as f:
        # Store grid / field shape metadata in root attributes
        if grid is not None:
            f.attrs["grid_shape"] = np.asarray(grid).shape
        elif field_shape is not None:
            f.attrs["field_shape"] = np.asarray(field_shape)

        for time, fields_dict in points.items():
            grp = f.create_group(str(time))
            for field_name, data in fields_dict.items():
                grp.create_dataset(
                    field_name,
                    data=np.asarray(data),
                    compression="gzip",
                    compression_opts=4,
                )
    print(f"Fields written to {filename}")


def read_fields(filename, field, time):
    """
    Read a single field array from an HDF5 file written by :func:`write_fields`.

    If the file contains a ``grid_shape`` or ``field_shape`` attribute the
    returned array is automatically reshaped to that shape.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file.
    field : str
        Name of the field dataset to read (e.g. ``'dens'``, ``'potl'``).
    time : float or str
        Time snapshot key.  Converted to ``str`` for the HDF5 group lookup.

    Returns
    -------
    data : numpy.ndarray
        The field array, reshaped to the stored grid/field shape when
        available.
    """
    with h5py.File(filename, "r") as f:
        data = np.array(f[str(time)][field])

        if "grid_shape" in f.attrs:
            data = data.reshape(f.attrs["grid_shape"])
        elif "field_shape" in f.attrs:
            data = data.reshape(f.attrs["field_shape"])

    return data


def write_kde_density(kd_dens, filename, grid_shape, snapshot_name, Ndens=64):
    """
    Write a KDE density field to an HDF5 file.

    Parameters
    ----------
    kd_dens : ndarray, shape (nbins, nbins, nbins)
        KDE density array returned by
        :meth:`~field_projections.FieldProjections.kde_density`.
    filename : str
        Path to the output HDF5 file.
    grid_shape : tuple of int
        Shape of the 3-D grid, e.g. ``(nbins, nbins, nbins)``.
    snapshot_name : str
        Identifier of the snapshot used to compute the density field.
    Ndens : int, optional
        Number of nearest neighbours used in the KDE (default 64).
    """
    with h5py.File(filename, "w") as f:
        f.attrs["grid_shape"] = np.asarray(grid_shape)
        f.attrs["snapshot_name"] = snapshot_name
        f.attrs["Ndens"] = Ndens

        f.create_dataset(
            "kde_density",
            data=np.asarray(kd_dens),
            compression="gzip",
            compression_opts=4,
        )
    print(f"KDE density written to {filename}")


def read_kde_density(filename):
    """
    Read a KDE density field from an HDF5 file written by
    :func:`write_kde_density`.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file.

    Returns
    -------
    kd_dens : ndarray
        The density array, reshaped to the stored ``grid_shape``.
    attrs : dict
        Header attributes (``grid_shape``, ``snapshot_name``, ``Ndens``).
    """
    with h5py.File(filename, "r") as f:
        data = np.array(f["kde_density"])
        attrs = {key: f.attrs[key] for key in f.attrs}

        if "grid_shape" in attrs:
            data = data.reshape(attrs["grid_shape"])

    return data, attrs
