# computes_MW_orbit.py
import os
import sys
import numpy as np
import pynbody
import nba
import re

sim = "MWLMC4"

def process_snapshot(snapname, snapshot_dir, outpath):
    GC21_bulge = nba.ios.ReadGC21(snapshot_dir, snapname)
    bulge_data = GC21_bulge.read_halo(['pos', 'vel', 'mass', 'pid', 'pot'], halo='MW', ptype='bulge')

    GC21_halo = nba.ios.ReadGC21(snapshot_dir, snapname)
    halo_data = GC21_halo.read_halo(['pos', 'vel', 'mass', 'pid', 'pot'], halo='MW', ptype='dm')

    print("done loading data")
    # Load COM 
    tag = re.sub(r"_\d+\.hdf5$", "", snapname)
    nsnap = int(re.search(r'_(\d+)\.hdf5$', snapname).group(1))
    bulge_com = np.loadtxt(f"{outpath}/{sim}/{tag}_nba_bulge_pot.txt")[nsnap,0:3]
    halo_com = np.loadtxt(f"{outpath}/{sim}/{tag}_nba_halo_pot.txt")[nsnap,0:3]

    # Compute density profile
    mwhalo_rcom = nba.com.CenterHalo(halo_data)
    mwhalo_rcom.recenter(halo_com, np.array([0,0,0]))    

    mwbulge_rcom = nba.com.CenterHalo(bulge_data)
    mwbulge_rcom.recenter(bulge_com, np.array([0,0,0]))    

    new_halo_com, _ = mwhalo_rcom.min_potential(rcut=5)
    new_bulge_com, _ = mwbulge_rcom.min_potential(rcut=2)
    
    #print('-> new halo center {}'.format(new_halo_com))
    #print('-> new bulge center {}'.format(new_bulge_com))
    assert np.linalg.norm(new_halo_com) < 0.5
    assert np.linalg.norm(new_bulge_com) < 0.3
    
    hbins = np.linspace(0, 300, 500)
    bbins = np.linspace(0, 20, 100)

    GC21_halo_profile = nba.structure.Profiles(halo_data['pos'], edges=hbins)
    GC21_bulge_profile = nba.structure.Profiles(bulge_data['pos'], edges=bbins)

    Rhalo, Dhalo = GC21_halo_profile.density(mass=halo_data['mass'])
    Rbulge, Dbulge = GC21_bulge_profile.density(mass=bulge_data['mass'])
    nsnap_f = "{:03d}".format(nsnap) 
    #print("done computing profile")
    #p.savetxt(f"{outpath}/{tag}_halo_density_profile_{nsnap_f}.txt", Dhalo)
    #p.savetxt(f"{outpath}/{tag}_bulge_density_profile_{nsnap_f}.txt".format(nsnap), Dbulge)

    return nsnap_f, Rhalo, Dhalo, Rbulge, Dbulge

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python compute_density_profile.py SNAPNAME SNAPSHOT_DIR OUTPATH")
        sys.exit(1)
    
    
    
    snapname = sys.argv[1]
    snapshot_dir = sys.argv[2]
    outpath = sys.argv[3]
    process_snapshot(snapname, snapshot_dir, outpath)
    snaplist_file, snapshot_dir, outpath = sys.argv[1], sys.argv[2], sys.argv[3]

    nsnap = int(re.search(r'_(\d+)\.hdf5$', snapname).group(1))
    if os.path.isfile(f"{outpath}/{sim}_density_profiles_{nsnap}.npz") == False:
        all_dhalo = []
        all_dbulge = []
        all_rhalo = []
        all_rbulge = []
        
        snapname = snapname.strip()
        nsnap, Rhalo, Dhalo, Rbulge, Dbulge = process_snapshot(snapname, snapshot_dir, outpath)
        all_dhalo.append(Dhalo)
        all_dbulge.append(Dbulge)
        all_rhalo.append(Rhalo)
        all_rbulge.append(Rbulge)

        # Convert to arrays and save one file each
        all_dhalo = np.array(all_dhalo)
        all_rhalo = np.array(all_rhalo)
        all_dbulge = np.array(all_dbulge)
        all_rbulge = np.array(all_rbulge)

        np.savez(f"{outpath}/{sim}_density_profiles_{nsnap}.npz",
                 Rh = all_rhalo,
                 Rb = all_rbulge,
                 Dh = all_dhalo,
                 Db = all_dbulge, tag=nsnap)
