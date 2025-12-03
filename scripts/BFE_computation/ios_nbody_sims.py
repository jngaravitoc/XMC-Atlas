import numpy as np
import nba

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
            npart = len(halo_data['mass'])

            # TODO implement this random sampling in NBA
            if npart_rand:
                npart_sample = npart_rand
                sample_factor = npart / npart_sample
                rand_part = np.random.randint(0, npart, npart_sample)
                halo_data['pos'] = halo_data['pos'][rand_part]
                halo_data['mass'] = halo_data['mass'][rand_part] * sample_factor
            
            print("*Done loading {}".format(snapname))

        return halo_data

def recentering(origin_dir, particle_data, nsnap, component, suite):
    """
    Recenters particle data

    Parameters:
    -----------
    particle data : np.ndarray, shape (N, 3)
    snapname : str
        string with the snapshot name 
        density_center : np.ndarray, shape (3)

    TODO: adds velocity center if needed!
    """
    # Load expansion centers
    expansion_center = load_GC21_exp_center(origin_dir, nsnap, component, suite)
    particle_data['pos'] = particle_data['pos'] -  expansion_center 
    return particle_data

# TODO move this function to bfe_computation_helper.py
def load_particle_data(origin_dir, component, suite, nsnap, **kwargs):
    """
    Load particle data

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

    logging.info("Done loading snapshot")
    
    # TODO check: are we passing here nsnap too?
    particle_data = recentering(origin_dir, particle_data, nsnap, component, suite)
    logging.info("Done re-centering data")
    # TODO does this need to be done here?
    snap_times = load_data.load_snap_time() 
    logging.info("Done loading snap-time data")

    return particle_data, snap_times
