# TODO load bulge and disk particles
import numpy as np
import nba
import os

def check_files_in_folder(folder_path, expected_files):
    """
    Check that all expected files exist inside a folder.

    Parameters
    ----------
    folder_path : str or Path
        Path to the directory to check.
    expected_files : list of str
        Filenames expected to appear in the folder (exact matches).

    Returns
    -------
    missing_files : list of str
        Files that were expected but not found.

    Raises
    ------
    FileNotFoundError
        If any expected file is missing.
    """
    missing = []

    for fname in expected_files:
        full_path = os.path.join(folder_path, fname)
        if not os.path.isfile(full_path):
            missing.append(fname)

    if missing:
        raise FileNotFoundError(
            f"The following files are missing in {folder_path}:\n{missing}"
        )

    return True



class LoadSim:
    def __init__(self, snapshot_dir, snapname, origin_filename, suite):
        """
        XMC-Atlas class to load particle data from simulations suites
        """

        self.snapshot_dir = snapshot_dir
        self.snapname = snapname
        # TODO: Re-center halos this vvv
        self.origin_filename = origin_filename
        self.suite = suite

    def load_snap_time(self):
        sim = nba.ios.ReadGadgetSim(self.snapshot_dir, self.snapname)
        snap_time = sim.read_header()['Time']
        return snap_time

    def load_mw_bulge(self, quantities=['pos', 'mass']):
        if self.suite == 'GC21':
            GC21_bulge = nba.ios.ReadGC21(self.snapshot_dir, self.snapname)
            bulge_data = GC21_bulge.read_halo(quantities, halo='MW', ptype='bulge')
        """
        elif self.suite == 'Sheng':
            Sheng_bulge = nba.ios.ReadGadgetSim(self.snapshot_dir, self.snapname)
            bulge_data = Sheng_bulge.read_halo(quantities)
        """
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



def test_read_time():
    SNAPSHOT_DIR = "/n/nyx3/garavito/XMC-Atlas-sims/GC21/MW/out/"
    SNAP_NAME = "MW_100M_beta1_vir_OM3_G4_150.hdf5"
    sim = LoadSim(SNAPSHOT_DIR, SNAP_NAME, "NONE", "NONE")
    time = sim.load_snap_time()
    assert time == 150*0.02

if __name__ == "__main__":
    test_read_time()


