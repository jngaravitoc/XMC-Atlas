#!/usr/bin/env python3
"""
Grid-based field evaluation and KDE density computation using pyEXP.
"""

import numpy as np
import pyEXP


class FieldProjections:
    """Evaluate BFE fields and KDE densities on a regular 3-D Cartesian grid.

    Parameters
    ----------
    grid : ndarray, shape (3, N, N, N)
        Stacked meshgrid arrays (x, y, z).
    basis : pyEXP basis object
        The spherical basis expansion.
    coefs : pyEXP coefs object
        Coefficient container.
    times : array-like
        Snapshot times.
    """

    def __init__(self, grid, basis, coefs, times):
        self.grid = grid
        self.basis = basis
        self.coefs = coefs
        self.times = times
        assert np.shape(self.grid)[0] == 3
        assert np.shape(self.grid)[1] == np.shape(self.grid)[2] == np.shape(self.grid)[3]
        self.nbins = np.shape(self.grid)[1]
        self.mesh = np.zeros((self.nbins**3, 3))
        self.mesh[:, 0] = self.grid[0].flatten()
        self.mesh[:, 1] = self.grid[1].flatten()
        self.mesh[:, 2] = self.grid[2].flatten()

    def compute_fields_in_points(self):
        """Evaluate all BFE fields at every grid point for every time.

        Returns
        -------
        points : dict
            ``{time: {field_name: ndarray, ...}, ...}``
        """
        fields = pyEXP.field.FieldGenerator(self.times, self.mesh)
        points = fields.points(self.basis, self.coefs)
        return points

    def twod_field(self, points, time, field):
        """Extract one or more 3-D field arrays for a single snapshot.

        Parameters
        ----------
        points : dict
            Output of :meth:`compute_fields_in_points`.
        time : float
            Must be present in ``self.times``.
        field : str or list of str
            Field name(s) to extract.

        Returns
        -------
        list of ndarray
            Each array has shape ``(nbins, nbins, nbins)``.
        """
        available_fields = [
            'azi force', 'dens', 'dens m=0',
            'dens m>0', 'mer force', 'potl',
            'potl m=0', 'potl m>0', 'rad force',
        ]

        if isinstance(field, str):
            field = [field]

        for f in field:
            if f not in available_fields:
                raise ValueError(f"field value {f!r} not available")

        if time not in self.times:
            raise ValueError("Requested time not in times")

        fields = []
        for f in field:
            field_arr = np.array(points[time][f]).reshape(
                self.nbins, self.nbins, self.nbins
            )
            fields.append(field_arr)
        return fields

    def kde_density(self, pos, mass, Ndens=64):
        """Compute KDE density at every grid point.

        Parameters
        ----------
        pos : ndarray, shape (Np, 3)
            Particle positions.
        mass : ndarray, shape (Np,)
            Particle masses.
        Ndens : int, optional
            Number of nearest neighbours (default 64).

        Returns
        -------
        kd_dens : ndarray, shape (nbins, nbins, nbins)
        """
        kddens = pyEXP.util.KDdensity(mass=mass, pos=pos, Ndens=Ndens)
        kd_dens = np.zeros(self.nbins**3)
        for i in range(self.nbins**3):
            kd_dens[i] = kddens.getDensityAtPoint(
                self.mesh[i, 0], self.mesh[i, 1], self.mesh[i, 2]
            )
        return kd_dens.reshape(self.nbins, self.nbins, self.nbins)
