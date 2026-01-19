import sys

sys.path.append("../")

from ios_nbody_sims import load_particle_data


def test_load_particle_data():
    
    SNAPSHOT_PATH = "../../../../XMC-Atlas-sims/Sheng/Model_108"
    SNAPNAME = "snapshot"
    suite = "Sheng24"
    nsnap = 0
    data = load_particle_data(SNAPSHOT_PATH, SNAPNAME, ['MWhalo'], nsnap, suite=suite)
    npart_halo = len(data['MWhalo']['mass'])
    data = load_particle_data(SNAPSHOT_PATH, SNAPNAME, ['MWdisk'], nsnap, suite=suite)
    npart_disk = len(data['MWdisk']['mass'])
    data = load_particle_data(SNAPSHOT_PATH, SNAPNAME, ['MWbulge'], nsnap, suite=suite)
    npart_bulge = len(data['MWbulge']['mass'])
    data = load_particle_data(SNAPSHOT_PATH, SNAPNAME, ['LMChalo'], nsnap, suite=suite)
    npart_lmc = len(data['LMChalo']['mass'])
    
    assert npart_halo == 704000
    assert npart_disk == 68000
    assert npart_bulge == 5000
    assert npart_lmc == 145690
    

    SNAPSHOT_PATH = "../../../../XMC-Atlas-sims/GC21/MWLMC3_b0/out"
    SNAPNAME = "MWLMC3_100M_b0_vir_OM3_G4"
    suite = 'GC21'
    nsnap = 0
    data = load_particle_data(SNAPSHOT_PATH, SNAPNAME, ['MWhalo'], nsnap, suite)
    npart_halo = len(data['MWhalo']['mass'])
    data = load_particle_data(SNAPSHOT_PATH, SNAPNAME, ['MWdisk'], nsnap, suite)
    npart_disk = len(data['MWdisk']['mass'])
    data = load_particle_data(SNAPSHOT_PATH, SNAPNAME, ['MWbulge'], nsnap, suite)
    npart_bulge = len(data['MWbulge']['mass'])
    data = load_particle_data(SNAPSHOT_PATH, SNAPNAME, ['LMChalo'], nsnap, suite)
    npart_lmc = len(data['LMChalo']['mass'])

    assert npart_halo == 100_000_000
    assert npart_disk == 5_780_000
    assert npart_bulge == 1_400_000
    assert npart_lmc == 6_666_666
    
