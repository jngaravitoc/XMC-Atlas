import numpy as np
import importlib.resources 
import gala.potential as gp
from astropy import units as u

class GC21:
    """
    Cranes potentials as defined in Section 3 of Patel+24.
    """

    def __init__(self, LMC_model):
        """
        Initialize the CranesPOT class with default model parameters.

        Parameters
        ----------
        LMC_model : str, optional
            The LMC model to use ('LMC3' or 'LMC4'). Default is 'LMC3'.
        """
        self.m_disk = 5.78e10
        self.a_disk = 2.4
        self.b_disk = 0.5
        self.c_bulge = 0.7
        self.m_bulge = 1.4e10
        self.MWvmax = 203.28453679656351 * u.km / u.s
        self.mw_pmass = 1e10
        self.lmc_pmass = 1e10
        self.rs_mw = 40.85
        self.m_halo = 1.57e12
        self.c_halo = 15.0
        self.LMC_model = LMC_model


        if LMC_model == 'LMC1':
            self.m_lmc = 0.8e11
            self.c_lmc = 10.4
        elif LMC_model == 'LMC2':
            self.m_lmc = 1e11
            self.c_lmc = 12.7
        elif LMC_model == 'LMC3':
            self.m_lmc = 1.8e11
            self.c_lmc = 20.0
        elif LMC_model == 'LMC4':
            self.m_lmc = 2.5e11
            self.c_lmc = 25.2


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
        cranes_pot = gp.CCompositePotential()

        cranes_pot['halo'] = gp.HernquistPotential(
            m=self.m_halo * u.Msun,
            c=self.rs_mw * u.kpc,
            units=[u.kpc, u.Gyr, u.Msun, u.radian]
        )
        cranes_pot['bulge'] = gp.HernquistPotential(
            m=self.m_bulge * u.Msun,
            c=self.c_bulge * u.kpc,
            units=[u.kpc, u.Gyr, u.Msun, u.radian],
        )
        cranes_pot['disk'] = gp.MiyamotoNagaiPotential(
            m=self.m_disk * u.Msun,
            a=self.a_disk * u.kpc,
            b=self.b_disk * u.kpc,
            units=[u.kpc, u.Gyr, u.Msun, u.radian],
        )
        return cranes_pot

    def lmc(self):
        """Return the LMC potential using a Hernquist profile."""
        return gp.HernquistPotential(
            m=self.m_lmc * u.Msun,
            c=self.c_lmc * u.kpc,
            units=[u.kpc, u.Gyr, u.Msun, u.radian]
        )
