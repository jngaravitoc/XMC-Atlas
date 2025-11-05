# TODO load bulge and disk particles
import numpy as np
import nba
import pyEXP

class LoadSim:
    def init(self, snapshot_dir, snapname, origin_filename):
        #self.suite = suite
        self.snapshot_dir = snapsht_dir
        self.snapname = snapname
        self.origin_filename = origin_filename

    def load_mw_bulge(self, quantities=['pos', 'mass']):
        GC21_bulge = nba.ios.ReadGC21(self.snapshot_dir, self.snapname)
        bulge_data = GC21_bulge.read_halo(quantities, halo='MW', ptype='bulge')
        return bulge_data

    def load_mw_disk(self, quantities=['pos', 'mass']):
        GC21_disk = nba.ios.ReadGC21(self.snapshot_dir, self.snapname)
        disk_data = GC21_disk.read_halo(quantities, halo='MW', ptype='disk')
        return disk_data

    def load_halo(self, quantities=['pos', 'mass'], npart=None):
        if halo == 'MW' | 'LMC':
            C21_halo = nba.os.ReadGC21(self.snapshot_dir, self.snapname)
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
