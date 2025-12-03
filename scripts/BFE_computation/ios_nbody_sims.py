import logging
import numpy as np
import nba
from compute_bfe_helpers import load_GC21_exp_center

class LoadSim:
    def __init__(self, snapshot_dir, snapname):
        """
        XMC-Atlas class to load particle data from simulations suites
        TODO: add doctring
        """
        self.snapshot_dir = snapshot_dir
        self.snapname = snapname

    def load_snap_time(self):
        sim = nba.ios.ReadGadgetSim(self.snapshot_dir, self.snapname)
        snap_time = sim.read_header()['Time']
        return snap_time

    def load_mw_bulge(self, quantities=['pos', 'mass']):
        GC21_bulge = nba.ios.ReadGC21(self.snapshot_dir, self.snapname)
        bulge_data = GC21_bulge.read_halo(quantities, halo='MW', ptype='bulge')
        return bulge_data
        
    def load_mw_disk(self, quantities=['pos', 'mass']):
        GC21_disk = nba.ios.ReadGC21(self.snapshot_dir, self.snapname)
        disk_data = GC21_disk.read_halo(quantities, halo='MW', ptype='disk')
        return disk_data

    def load_halo(self, halo,  quantities=['pos', 'mass'], npart=None):
        """
        halo : str
            halo to which load the data
        """
        if halo == 'MW' or'LMC':
            GC21_halo = nba.ios.ReadGC21(self.snapshot_dir, self.snapname)
            if npart:
                halo_data = GC21_halo.read_halo(quantities, halo='MW', ptype='dm', randomsample=npart)
            else:   
                halo_data = GC21_halo.read_halo(quantities, halo='MW', ptype='dm')

        elif halo == 'MWnoLMC':
            MWhalo = nba.ios.ReadGadgetSim(self.snapshot_dir, self.snapname)
            halo_data = MWhalo.read_snapshot(quantities, 'dm')
            all_npart = len(halo_data['mass'])

            # TODO implement this random sampling in NBA
            if npart:
                sample_factor = all_npart / npart
                rand_part = np.random.randint(0, all_npart, npart)
                halo_data['pos'] = halo_data['pos'][rand_part]
                halo_data['mass'] = halo_data['mass'][rand_part] * sample_factor
            
            print("*Done loading {}".format(self.snapname))

        return halo_data

# TODO move this function to bfe_computation_helper.py
def load_particle_data(snapshot_dir, snapname, component, nsnap, **kwargs):
    """
    Load particle data

    Params:

    **kwargs:
        npart : int
            samples halo particles
    Returns:
    --------

    Particle data:

    snap_time:
    """
    full_snapname = snapname + "_{:03d}.hdf5".format(nsnap)
    load_data = LoadSim(snapshot_dir, full_snapname)
    # Load center
    print("--------------------------------")
    if component=='MWHaloiso':
        particle_data = load_data.load_halo('MWnoLMC', **kwargs)
    
    elif component=='MWHalo':
        particle_data = load_data.load_halo('MW', **kwargs)
    
    elif component=='LMChalo':
        particle_data = load_data.load_halo('LMC', **kwargs)
     
    elif component=='MWdisk':
        particle_data = load_data.load_mw_disk(**kwargs)
    
    elif component=='MWbulge':
        particle_data = load_data.load_mw_bulge(**kwargs)
    else:
        raise ValueError("Component not defined")

    logging.info("Done loading snapshot")
    

    # TODO does this need to be done here?
    snap_times = load_data.load_snap_time() 
    logging.info("Done loading snap-time data")

    return particle_data, snap_times
