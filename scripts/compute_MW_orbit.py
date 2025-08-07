"""
Script to compute the COM of the MW's halo

"""
import numpy as np
import nba
import pynbody


snapshot = "/mnt/home/nico/ceph/gadget_runs/MWLMC/MWLMC5_b0/out/"
snapname = "MWLMC5_100M_b0_vir_OM3_G4_110.hdf5"
nsnaps=1
outpath = './'
# load data

GC21_disk = nba.ios.ReadGC21(snapshot, snapname)
GC21_disk_data = GC21_disk.read_halo(
    ['pos', 'vel', 'mass', 'pid', 'pot'], 
    halo='MW', ptype='disk'
    )

GC21_bulge = nba.ios.ReadGC21(snapshot, snapname)

GC21_bulge_data = GC21_bulge.read_halo(
    ['pos', 'vel', 'mass', 'pid', 'pot'], 
    halo='MW', ptype='bulge'
    )

GC21_mwhalo = nba.ios.ReadGC21(snapshot, snapname)
GC21_mwhalo_data = GC21_mwhalo.read_halo(
    ['pos', 'vel', 'mass', 'pid', 'pot'], 
    halo='MW', ptype='dm')


# compute COM with nba

nba_disk_pot = np.zeros((nsnaps, 6))
nba_bulge_pot = np.zeros((nsnaps, 6))
nba_mwhalo_pot = np.zeros((nsnaps, 6))
pb_disk_pot = np.zeros((nsnaps, 3))
pb_bulge_pot = np.zeros((nsnaps, 3))
pb_halo_pot = np.zeros((nsnaps, 3))
pb_halo_ssc = np.zeros((nsnaps, 3))

disk_com = nba.com.CenterHalo(GC21_disk_data)
bulge_com = nba.com.CenterHalo(GC21_bulge_data)
mwhalo_com = nba.com.CenterHalo(GC21_mwhalo_data)

r=8 # found to be good for velocity COM
for i in range(nsnaps):
    nba_bulge_pot[i, :3], nba_bulge_pot[i, 3:] = bulge_com.min_potential(rcut=r)
    nba_disk_pot[i, :3], nba_disk_pot[i, 3:] = disk_com.min_potential(rcut=r)
    nba_mwhalo_pot[i, :3], nba_mwhalo_pot[i, 3:] = mwhalo_com.min_potential(rcut=r)

# compute COM with pynbody

MWdisk_pb = nba.ios.pynbody_routines.createPBhalo(GC21_disk_data)
MWbulge_pb = nba.ios.pynbody_routines.createPBhalo(GC21_bulge_data)
MWhalo_pb = nba.ios.pynbody_routines.createPBhalo(GC21_mwhalo_data)

for i in range(nsnaps):
    pb_disk_pot[i] = np.array(pynbody.analysis.center(MWdisk_pb, return_cen=True, with_velocity=True, mode='pot', cen_size='1 kpc'))
    pb_bulge_pot[i] = np.array(pynbody.analysis.center(MWbulge_pb, return_cen=True, with_velocity=True, mode='pot', cen_size='1 kpc'))
    pb_halo_pot[i] = np.array(pynbody.analysis.center(MWhalo_pb, return_cen=True, with_velocity=True, mode='pot', cen_size='1 kpc'))
    pb_halo_ssc[i] = np.array(pynbody.analysis.center(MWhalo_pb, return_cen=True, with_velocity=True, mode='ssc', cen_size='1 kpc'))

# Write centers:
np.savetxt(outpath + "nba_disk_center.txt", nba_disk_pot)
    

