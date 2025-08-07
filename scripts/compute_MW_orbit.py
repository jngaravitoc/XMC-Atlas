# process_com_snapshot.py
import sys
import numpy as np
import pynbody
import nba

def process_snapshot(snapname, snapshot_dir, outpath):
    GC21_disk = nba.ios.ReadGC21(snapshot_dir, snapname)
    disk_data = GC21_disk.read_halo(['pos', 'vel', 'mass', 'pid', 'pot'], halo='MW', ptype='disk')

    GC21_bulge = nba.ios.ReadGC21(snapshot_dir, snapname)
    bulge_data = GC21_bulge.read_halo(['pos', 'vel', 'mass', 'pid', 'pot'], halo='MW', ptype='bulge')

    GC21_halo = nba.ios.ReadGC21(snapshot_dir, snapname)
    halo_data = GC21_halo.read_halo(['pos', 'vel', 'mass', 'pid', 'pot'], halo='MW', ptype='dm')

    # NBA COM
    disk_com = nba.com.CenterHalo(disk_data)
    bulge_com = nba.com.CenterHalo(bulge_data)
    halo_com = nba.com.CenterHalo(halo_data)

    r = 8
    nba_disk_pot = np.array(disk_com.min_potential(rcut=r)).flatten()
    nba_bulge_pot = np.array(bulge_com.min_potential(rcut=r)).flatten()
    nba_halo_pot = np.array(halo_com.min_potential(rcut=r)).flatten()

    # Pynbody COM
    disk_pb = nba.ios.pynbody_routines.createPBhalo(disk_data)
    bulge_pb = nba.ios.pynbody_routines.createPBhalo(bulge_data)
    halo_pb = nba.ios.pynbody_routines.createPBhalo(halo_data)

    pb_disk_pot = np.array(pynbody.analysis.center(disk_pb, return_cen=True, with_velocity=True, mode='pot', cen_size='1 kpc'))
    pb_bulge_pot = np.array(pynbody.analysis.center(bulge_pb, return_cen=True, with_velocity=True, mode='pot', cen_size='1 kpc'))
    pb_halo_pot = np.array(pynbody.analysis.center(halo_pb, return_cen=True, with_velocity=True, mode='pot', cen_size='1 kpc'))
    pb_halo_ssc = np.array(pynbody.analysis.center(halo_pb, return_cen=True, with_velocity=True, mode='ssc', cen_size='1 kpc'))

    # Write to file
    tag = snapname.replace(".hdf5", "")
    np.savetxt(f"{outpath}/{tag}_nba_disk_pot.txt", nba_disk_pot.reshape(1, -1))
    np.savetxt(f"{outpath}/{tag}_nba_bulge_pot.txt", nba_bulge_pot.reshape(1, -1))
    np.savetxt(f"{outpath}/{tag}_nba_halo_pot.txt", nba_halo_pot.reshape(1, -1))
    np.savetxt(f"{outpath}/{tag}_pb_disk_pot.txt", pb_disk_pot.reshape(1, -1))
    np.savetxt(f"{outpath}/{tag}_pb_bulge_pot.txt", pb_bulge_pot.reshape(1, -1))
    np.savetxt(f"{outpath}/{tag}_pb_halo_pot.txt", pb_halo_pot.reshape(1, -1))
    np.savetxt(f"{outpath}/{tag}_pb_halo_ssc.txt", pb_halo_ssc.reshape(1, -1))

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python process_com_snapshot.py SNAPNAME SNAPSHOT_DIR OUTPATH")
        sys.exit(1)

    snapname = sys.argv[1]
    snapshot_dir = sys.argv[2]
    outpath = sys.argv[3]
    process_snapshot(snapname, snapshot_dir, outpath)

