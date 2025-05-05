import numpy as np
import importlib.resources 
import gala.potential as gp
from astropy import units as u

class L18:
    """
    Laporte+18 potentials as defined in Section 2 in (https://arxiv.org/pdf/1608.04743)
    """

    def __init__(self, LMC_model):
        """
        Initialize the Laporte18 halo model potentials.

        Parameters
        ---------
        LMC_model : str, optional
            The LMC model to use ('LMC3' or 'LMC4'). Default is 'LMC3'.
        """
        self.m_disk = 6.5e10
        self.hR_disk = 3.5
        self.hz_disk = 0.53
        self.c_bulge = 0.7
        self.m_bulge = 1.e10
        self.rs_mw = 28
        self.m_halo = 9.3e11
        self.LMC_model = LMC_model


        if LMC_model == 'LMC1':
            self.m_lmc = 0.34e11
            self.c_lmc = 3.3
        elif LMC_model == 'LMC2':
            self.m_lmc = 0.63e11
            self.c_lmc = 8.3
        elif LMC_model == 'LMC3':
            self.m_lmc = 1.07e11
            self.c_lmc = 13.5
        elif LMC_model == 'LMC4':
            self.m_lmc = 1.39e11
            self.c_lmc = 16.8
        elif LMC_model == 'LMC5':
            self.m_lmc = 2.76e11
            self.c_lmc = 27.2
        elif LMC_model == 'LMC6':
            self.m_lmc = 4e11
            self.c_lmc = 34.6

    def milkyway(self, **kwargs):
        """
        Construct the rigid Milky Way potential.

        Parameters
        ----------
        kwargs : dict
            Dictionary containing 'origin'.

        Returns
        -------
        gala.potential.CCompositePotential
            The composite MW potential.
        """
        mw_pot = gp.CCompositePotential()

        mw_pot['halo'] = gp.HernquistPotential(
            m=self.m_halo * u.Msun,
            c=self.rs_mw * u.kpc,
            units=[u.kpc, u.Gyr, u.Msun, u.radian]
        )
        mw_pot['bulge'] = gp.HernquistPotential(
            m=self.m_bulge * u.Msun,
            c=self.c_bulge * u.kpc,
            units=[u.kpc, u.Gyr, u.Msun, u.radian],
        )
        mw_pot['disk'] = gp.MN3ExponentialDiskPotential(
            m=self.m_disk * u.Msun,
            h_R=self.hR_disk * u.kpc,
            h_z=self.hz_disk * u.kpc,
            units=[u.kpc, u.Gyr, u.Msun, u.radian],
        )
        return mw_pot

    def lmc(self):
        """Return the LMC potential using a Hernquist profile."""
        return gp.HernquistPotential(
            m=self.m_lmc * u.Msun,
            c=self.c_lmc * u.kpc,
            units=[u.kpc, u.Gyr, u.Msun, u.radian]
        )
