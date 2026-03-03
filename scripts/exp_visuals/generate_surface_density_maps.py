#start
#generate xy and xz surface density maps for the lmc disk
#parallelizable

import time
import sys
import pyEXP
import matplotlib.pyplot as plt
import plots_init as pi
import lmc_disk_exp_copilot as lc
import numpy as np
import snap_analysis as sna
import warnings
from scipy.integrate import simpson
import multiprocessing

warnings.filterwarnings("ignore", category = RuntimeWarning)

start_time = time.time()

# Specifying the basis configuration

disk_config = """
---
id: cylinder
parameters:
  acyl: 2.38                        # exponential disk scale length, Martin's suggestion
  hcyl: 0.34                       # exponential disk scale height
  nmaxfid: 64                      # maximum radial order for spherical basis
  lmaxfid: 64                      # maximum harmonic order for spherical basis
  mmax: 16                          # maximum azimuthal order of cylindrical basis
  nmax: 48                          # maximum radial order of cylindrical basis, Mike's suggestion
  ncylodd: 24                       # vertically anti-symmetric basis functions, Martin's suggestion - nmax/2
  ncylnx: 256                      # grid points in radial direction
  ncylny: 128                      # grid points in vertical direction
  rnum: 200                        # radial quadrature knots for Gram matrix
  pnum: 0                          # azimuthal quadrature knots for Gram matrix
  tnum: 80                         # latitudinal quadrature knots for Gram matrix
  vflag: 0                         # verbose output flag
  logr: false                      # logarithmically spaced radial grid
  cachename: /xdisk/gbesla/himansh/lmc_disk_exp/basis/lmc01/sim/lmc01.cache.run0    # name of the basis cache file
...
"""

disk_basis = pyEXP.basis.Basis.factory(disk_config)
sys.stdout.write("Basis check completed." + "\n")

# Extracting snapshot list
snap_list = sna.get_snaps('/xdisk/gbesla/group/lmcsmc12/sim/output/')

# Extracting coefficients
coefs = pyEXP.coefs.Coefs.factory('/xdisk/gbesla/himansh/lmc_disk_exp/coeffs/lmc01/lmc_smc/lmcsmc_coefs')
snap_sampler = 20 #Coefficients were computed for every 20th snapshot

# Visualizing the fields
# We will use the pointmesh functionality in pyEXP - https://github.com/EXP-code/pyEXP-examples/blob/main/Tutorials/Introduction/Part2-Analysis.ipynb

coef_sampler = 100 #Sample every 10th coefficient for field generation 
times = coefs.Times()[::coef_sampler] #extracting the times 
snap_samples = np.arange(len(snap_list))[::int(snap_sampler*coef_sampler)] #sampling the snapshots at the same interval

pmin  = [-15, -15, -5]
pmax  = [15, 15, 5]
grid  = [150, 150, 50]

x = np.linspace(pmin[0], pmax[0], grid[0])
y = np.linspace(pmin[1], pmax[1], grid[1])
z = np.linspace(pmin[2], pmax[2], grid[2])

points_x, points_y, points_z = np.meshgrid(x, y, z)
coordinates = np.stack([points_x.flatten(), points_y.flatten(), points_z.flatten()], axis = 1)

fields = pyEXP.field.FieldGenerator(times, coordinates)

sys.stdout.write("Starting field generation..." + "\n")
field_points = fields.points(disk_basis, coefs)
sys.stdout.write("Field generation completed." + "\n")

cm = plt.cm.Blues.reversed()
cm.set_bad(color = 'black')

for i in range(len(times)):
    
    dens_exp = field_points[times[i]]['dens'].reshape(grid[0], grid[1], grid[2])
    sigma_xy_exp = simpson(dens_exp, z).T
    sigma_xz_exp = simpson(dens_exp, y, axis = 1)
    
    # xy surface density map
    fig = plt.figure(figsize = (27, 6))

    ax0 = fig.add_subplot(1, 3, 1)
    pc = ax0.pcolormesh(x, y, np.log10(sigma_xy_exp).T, cmap = cm, vmin = 5.5, vmax = 8.5)
    cbar = fig.colorbar(mappable = pc, ax = ax0, label = r'$\log(\Sigma_{\ast} \:\: [M_\odot \rm{kpc}^{-2}])$')
    cbar.ax.tick_params(labelsize = 15)
    ax0.set_xlabel('x [kpc]')
    ax0.set_ylabel('y [kpc]')
    ax0.set_title('EXP Density, t = ' + str(np.round(times[i], 2)) + ' Myr')
    ax0.set_xlim(pmin[0], pmax[0])
    ax0.set_ylim(pmin[1], pmax[1])
    ax0.tick_params(which = 'both', color = 'white')
    ax0.spines['top'].set_color('white')
    ax0.spines['bottom'].set_color('white')
    ax0.spines['left'].set_color('white')
    ax0.spines['right'].set_color('white')
    pi.aesthetic(ax0)

    ax1 = fig.add_subplot(1, 3, 2)
    sigma_xy = lc.get_lmcsmc_xy_surface_density(snap_list[snap_samples[i]], xlim = [pmin[0], pmax[0]], 
                                                ylim = [pmin[1], pmax[1]], bins = grid[0]+1, z_bounds = np.array([pmin[2], pmax[2]]))
    pc = ax1.pcolormesh(x, y, np.log10(sigma_xy).T, cmap = cm, vmin = 5.5, vmax = 8.5)
    cbar = fig.colorbar(mappable = pc, ax = ax1, label = r'$\log(\Sigma_{\ast} \:\: [M_\odot \rm{kpc}^{-2}])$')
    cbar.ax.tick_params(labelsize = 15)
    ax1.set_xlabel('x [kpc]')
    ax1.set_ylabel('y [kpc]')
    ax1.set_title('Simulated Density, t = ' + str(np.round(times[i], 2)) + ' Myr')
    ax1.set_xlim(pmin[0], pmax[0])
    ax1.set_ylim(pmin[1], pmax[1])
    ax1.tick_params(which = 'both', color = 'white')
    ax1.spines['top'].set_color('white')
    ax1.spines['bottom'].set_color('white')
    ax1.spines['left'].set_color('white')
    ax1.spines['right'].set_color('white')
    pi.aesthetic(ax1)

    ax2 = fig.add_subplot(1, 3, 3)
    rel_res = (sigma_xy_exp - sigma_xy)/sigma_xy #(model - truth)/truth
    pc = ax2.pcolormesh(x, y, rel_res.T, cmap = plt.cm.vanimo, vmin = -1, vmax = 1)
    cbar = fig.colorbar(mappable = pc, ax = ax2)
    cbar.ax.tick_params(labelsize = 15)
    ax2.set_xlabel('x [kpc]')
    ax2.set_ylabel('y [kpc]')
    ax2.set_title('Residual, t = ' + str(np.round(times[i], 2)) + ' Myr')
    ax2.set_xlim(pmin[0], pmax[0])
    ax2.set_ylim(pmin[1], pmax[1])
    ax2.tick_params(which = 'both', color = 'white')
    ax2.spines['top'].set_color('white')
    ax2.spines['bottom'].set_color('white')
    ax2.spines['left'].set_color('white')
    ax2.spines['right'].set_color('white')
    pi.aesthetic(ax2)
    
    plt.savefig('./xy_maps/xy_map_' + str(snap_samples[i]) + '.jpg')
    plt.close()
    
    #xz surface density map
    fig = plt.figure(figsize = (27, 6))

    ax0 = fig.add_subplot(1, 3, 1)
    pc = ax0.pcolormesh(x, z, np.log10(sigma_xz_exp).T, cmap = cm, vmin = 6, vmax = 9.5)
    cbar = fig.colorbar(mappable = pc, ax = ax0, label = r'$\log(\Sigma_{\ast} \:\: [M_\odot \rm{kpc}^{-2}])$')
    cbar.ax.tick_params(labelsize = 15)
    ax0.set_xlabel('x [kpc]')
    ax0.set_ylabel('z [kpc]')
    ax0.set_title('EXP Density, t = ' + str(np.round(times[i], 2)) + ' Myr')
    ax0.set_xlim(pmin[0], pmax[0])
    ax0.set_ylim(pmin[2], pmax[2])
    ax0.tick_params(which = 'both', color = 'white')
    ax0.spines['top'].set_color('white')
    ax0.spines['bottom'].set_color('white')
    ax0.spines['left'].set_color('white')
    ax0.spines['right'].set_color('white')
    pi.aesthetic(ax0)

    ax1 = fig.add_subplot(1, 3, 2)
    sigma_xz = lc.get_lmcsmc_xz_surface_density(snap_list[snap_samples[i]], xlim = [pmin[0], pmax[0]], zlim = [pmin[2], pmax[2]], 
                                                y_bounds = [pmin[1], pmax[1]], x_bins = grid[0]+1, z_bins = grid[2]+1)
    pc = ax1.pcolormesh(x, z, np.log10(sigma_xz).T, cmap = cm, vmin = 6, vmax = 9.5)
    cbar = fig.colorbar(mappable = pc, ax = ax1, label = r'$\log(\Sigma_{\ast} \:\: [M_\odot \rm{kpc}^{-2}])$')
    cbar.ax.tick_params(labelsize = 15)
    ax1.set_xlabel('x [kpc]')
    ax1.set_ylabel('z [kpc]')
    ax1.set_title('Simulated Density, t = ' + str(np.round(times[i], 2)) + ' Myr')
    ax1.set_xlim(pmin[0], pmax[0])
    ax1.set_ylim(pmin[2], pmax[2])
    ax1.tick_params(which = 'both', color = 'white')
    ax1.spines['top'].set_color('white')
    ax1.spines['bottom'].set_color('white')
    ax1.spines['left'].set_color('white')
    ax1.spines['right'].set_color('white')
    pi.aesthetic(ax1)

    ax2 = fig.add_subplot(1, 3, 3)
    rel_res = (sigma_xz_exp - sigma_xz)/sigma_xz #(model - truth)/truth
    pc = ax2.pcolormesh(x, z, rel_res.T, cmap = plt.cm.vanimo, vmin = -1, vmax = 1)
    cbar = fig.colorbar(mappable = pc, ax = ax2)
    cbar.ax.tick_params(labelsize = 15)
    ax2.set_xlabel('x [kpc]')
    ax2.set_ylabel('z [kpc]')
    ax2.set_title('Residual, t = ' + str(np.round(times[i], 2)) + ' Myr')
    ax2.set_xlim(pmin[0], pmax[0])
    ax2.set_ylim(pmin[2], pmax[2])
    ax2.tick_params(which = 'both', color = 'white')
    ax2.spines['top'].set_color('white')
    ax2.spines['bottom'].set_color('white')
    ax2.spines['left'].set_color('white')
    ax2.spines['right'].set_color('white')
    pi.aesthetic(ax2)
    
    plt.savefig('./xz_maps/xz_map_' + str(snap_samples[i]) + '.jpg')
    plt.close()
    
    sys.stdout.write("Generating map " + str(i) + "\n")
    
end_time = time.time()
run_time = str(np.round((end_time - start_time), 0))

sys.stdout.write("Done ! Run time was: " + run_time + "\n")
    
#end