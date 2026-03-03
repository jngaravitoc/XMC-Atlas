import os
import sys
import agama
from ios_nbody_sims import LoadSim

agama.setUnits(mass=1e10, length=1, velocity=1)  # Msol, kpc, km/s

def createAgamaPotential(potential_file, xyz):
	pot_agama = agama.Potential(potential_file)
	print(pot_agama.force(xyz)


if __name__ == "__main__":
    file_name = sys.argv[1]
    cfg = load_config(config_file)
    snapname = cfg["snapname"]
    snapshot_dir = cfg["snapshot_dir"]
    orbit = cfg["orbit"]

    
    halo_data = LoadSim(snapshot_dir, snapname, expansion_center)
    mw_halo_particles = halo_data.load_halo(halo='MW', quantities=['pos', 'vel', 'pot', 'mass'], npart=npart)
    # Load center
    pos_center, vel_center, nsnap = load_center(snapname, expansion_center)
    mwhalo_recenter = nba.com.CenterHalo(mw_halo_particles)
    mwhalo_recenter.recenter(pos_center, [0,0,0])    

    mw_disk_particles = halo_data.load_mw_disk(quantities=['pos', 'vel', 'pot', 'mass'])
    mw_disk_recenter = nba.com.CenterHalo(mw_disk_particles)
    mw_disk_recenter.recenter(pos_center, [0,0,0])    

    npart_rand = 100_000
    pos = mw_halo_particles['pos']
    distance = np.linalg.norm(pos, axis=1)
    rcut = np.where(distance<500)
    random_part = np.random.randint(0, len(rcut), npart_rand)
    pos_rand = pos[rcut][random_part]
    print(pos_rand.shape())

    #createAgamaPotential(file_name, xyz)
