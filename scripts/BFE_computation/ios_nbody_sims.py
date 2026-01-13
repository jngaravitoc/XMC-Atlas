import logging
import numpy as np
from compute_bfe_helpers import load_GC21_exp_center
import nba


class LoadSim:
    """
    Class for loading particle data from XMC-Atlas simulation snapshots.
    """
    def __init__(self, snapshot_dir, snapname, suite):
        """
        Parameters
        ----------
        snapshot_dir : str
            Path to the snapshot directory.
        snapname : str
            Snapshot file name.
        suite : str
            Suite name (Sheng24 or GC21)
	
        TODO: Downsampling will increase the mass of the particles, this should be represented 
        in the unit value for the mass.
        """
        self.snapshot_dir = snapshot_dir
        self.snapname = snapname
        #self.GC21 = nba.ios.ReadGC21(self.snapshot_dir, self.snapname)
        self.gadget = nba.ios.ReadGadgetSim(self.snapshot_dir, self.snapname)
        self.suite = suite
        if suite == "GC21":
            self.sim = nba.ios.ReadGC21(self.snapshot_dir, self.snapname)
        elif suite == "Sheng24":
            self.sim = nba.ios.ReadSheng24(self.snapshot_dir, self.snapname)
        else:
            raise ValueError "Requested suite not found please choose between Sheng24 or GC21"

    def load_snap_time(self):
        snap_time = self.gadget.read_header()['Time']
        return snap_time

    def load_mw_bulge(self, quantities=None, npart=None):
        if quantities is None:
            quantities = ['pos', 'mass']
        bulge_data = self.sim.read_halo(quantities, halo='MW', ptype='bulge')
        return bulge_data
        
    def load_mw_disk(self, quantities=None, npart=None):
        if quantities is None:
            quantities = ['pos', 'mass']
        disk_data = self.sim.read_halo(quantities, halo='MW', ptype='disk')
        return disk_data

    def load_halo(self, halo,  quantities=None, npart=None):
        """
        halo : str
            halo to which load the data
        """
        if quantities is None:
            quantities = ['pos', 'mass']

        if halo in ('MW', 'LMC'):
            if npart:
                halo_data = self.sim.read_halo(quantities, halo=halo, ptype='dm', randomsample=npart)
            else:   
                halo_data = self.sim.read_halo(quantities, halo=halo, ptype='dm')

        elif halo == 'MWnoLMC':
            if self.suite == 'Sheng24':
                raise ValueError "halo MWnoLMC not available for the Sheng24 suite"
                sys.exit()

            halo_data = self.gadget.read_snapshot(quantities, 'dm')
            all_npart = len(halo_data['mass'])

            # TODO implement this random sampling in NBA
            if npart:
                sample_factor = all_npart / npart
                rand_part = np.random.choice(all_npart, size=npart, replace=False)
                halo_data['pos'] = halo_data['pos'][rand_part]
                halo_data['mass'] = halo_data['mass'][rand_part] * sample_factor
            
        else:
            raise ValueError(f"Unsupported halo type: {halo}")

        return halo_data

def load_particle_data(snapshot_dir, snapname, components, nsnap, suite, **kwargs):
    """
    Load particle data for one or more galaxy components from a snapshot.

    Parameters
    ----------
    snapshot_dir : str
        Path to the directory containing the simulation snapshots.
    snapname : str
        Base name of the snapshot (without numeric index or extension).
    components : str or list of str
        Component name(s) to load. Must be one or more of:
        ['MWhaloiso', 'MWhalo', 'LMChalo', 'MWdisk', 'MWbulge'].
    nsnap : int
        Snapshot number.
    **kwargs : dict
        Additional keyword arguments forwarded to the particle-loading functions.
        Common options include:
            - npart : int
                Number of particles to sample.

    Returns
    -------
    data : dict
        Nested dictionary of particle data:
        {
            component_name: {
                "pos": ndarray,
                "mass": ndarray
            },
            ...
        }

    snap_time : float
        Simulation time corresponding to the snapshot.
    """

    # ----------------------------
    # Validate and normalize input
    # ----------------------------
    allowed_components = {'MWhaloiso', 'MWhalo', 'LMChalo', 'MWdisk', 'MWbulge'}

    invalid = set(components) - allowed_components
    if invalid:
        raise ValueError(
            f"Invalid component(s): {invalid}. "
            f"Allowed values are {sorted(allowed_components)}"
        )

    # ----------------------------
    # Load snapshot
    # ----------------------------
    full_snapname = f"{snapname}_{nsnap:03d}.hdf5"
    load_data = LoadSim(snapshot_dir, full_snapname, suite)

    logging.info("--------------------------------")
    logging.info(f"Loading snapshot: {full_snapname}")

    data = {}

    # ----------------------------
    # Load each component
    # ----------------------------
    for component in components:

        if component == 'MWhaloiso':
            particle_data = load_data.load_halo('MWnoLMC', **kwargs)

        elif component == 'MWhalo':
            particle_data = load_data.load_halo('MW', **kwargs)

        elif component == 'LMChalo':
            particle_data = load_data.load_halo('LMC', **kwargs)

        elif component == 'MWdisk':
            particle_data = load_data.load_mw_disk(**kwargs)

        elif component == 'MWbulge':
            particle_data = load_data.load_mw_bulge(**kwargs)

        else:
            # This should never happen due to validation above
            raise RuntimeError(f"Unhandled component: {component}")

        # Enforce required output structure
        data[component] = {
            "pos": particle_data["pos"],
            "mass": particle_data["mass"]
        }

        logging.info(f"Loaded component: {component}")
        
        data[component]['snapshot_time'] = load_data.load_snap_time()
    # ----------------------------
    # Load snapshot time (once)
    # ----------------------------
    logging.info("Done loading snapshot time")
    return data
