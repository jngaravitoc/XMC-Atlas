import sys
import numpy as np
import nba

sys.path.append("../")

from ios_nbody_sims import load_particle_data


def test_load_particle_data():
    
    SNAPSHOT_PATH = ""
    SNAPNAME = "snapshot_"
    suite = 'Sheng24'
    nsnap = 0
    data = load_particle_data(SNAPSHOT_PATH, SNAPNAME, 'MWhalo', suite)
    npart_halo = len(data['mass'])
    data = load_particle_data(SNAPSHOT_PATH, SNAPNAME, 'MWdisk', suite)
    npart_disk = len(data['mass'])
    data = load_particle_data(SNAPSHOT_PATH, SNAPNAME, 'MWbulge', suite)
    npart_bulge = len(data['mass'])
    data = load_particle_data(SNAPSHOT_PATH, SNAPNAME, 'LMChalo', suite)
    npart_lmc = len(data['mass'])

    assert npart_halo == 704000
    assert npart_disk == 6800
    assert npart_bulge == 5000
    assert npart_lmc == 145690
    

    SNAPSHOT_PATH = ""
    SNAPNAME = "snapshot_"
    suite = 'GC21'
    nsnap = 0
    data = load_particle_data(SNAPSHOT_PATH, SNAPNAME, 'MWhalo', suite)
    npart_halo = len(data['mass'])
    data = load_particle_data(SNAPSHOT_PATH, SNAPNAME, 'MWdisk', suite)
    npart_disk = len(data['mass'])
    data = load_particle_data(SNAPSHOT_PATH, SNAPNAME, 'MWbulge', suite)
    npart_bulge = len(data['mass'])
    data = load_particle_data(SNAPSHOT_PATH, SNAPNAME, 'LMChalo', suite)
    npart_lmc = len(data['mass'])

    assert npart_halo = 100_000_000
    #assert npart_disk = 6800
    #assert npart_bulge = 5000
    #assert npart_lmc = 145690 145690
