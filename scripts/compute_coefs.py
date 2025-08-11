#!/usr/bin/env python3.8
"""

"""

from argparse import ArgumentParser
import sys
import numpy as np
import pyEXP

import nba.ios.io_snaps as ios
import gadget_to_ascii as g2a
from allvars import BFEParameters
#from EXP_utils import make_config, make_a_BFE
#from EXP_external_sims import compute_coefs
#sys.path.append('/mnt/home/nico/ceph/codes/EXP_scripts/')

if __name__ == "__main__":
    paramfile = sys.argv[1]
    params = BFEParameters(paramfile)
    #if params.suite == 'GC21':
    print("-> Loading {} simulation data".param.suite)

    for i in range(params.initial_snapshot, params.final_snapshot):
        # *********************** Loading data: **************************
        print(in_path+snapname+"_{:03d}".format(i), n_halo_part, i)

        halo = ios.load_halo(
            in_path+snapname+"_{:03d}".format(i), 
            N_halo_part=[n_halo_part-1, 15000000], 
            com_frame=0, galaxy=0, snapformat=3, 
            q=['pos', 'vel', 'mass', 'pot', 'pid'], com_method='diskpot')
        
        print(np.shape(halo['pos']))
        """
        # Truncates halo:
        if rcut_halo > 0:
            pos_halo_tr, vel_halo_tr, mass_tr, ids_tr = g2a.truncate_halo(halo['pos'], halo['vel'], halo['mass'], halo['pid'], rcut_halo)
            del halo
        else:
            pos_halo_tr = halo['pos']
            vel_halo_tr = halo['vel']
            mass_tr = halo['mass']
            ids_tr = halo['pid']
            del halo
        # Sampling halo

        if npart_sample > 0:
            print("Samlling halo particles: \n")
            pos_halo_tr, vel_halo_tr, mass_tr, ids_tr = g2a.sample_halo(pos_halo_tr, vel_halo_tr,
                                                                        mass_tr, npart_sample, ids_tr)
        
        
                
        #*************************  Compute BFE: ***************************** 


        halo_config =  make_config("sphereSL", 4000, 0.01, rcut_halo, lmax, nmax, rs, modelname="SLGrid.empirical.halo.isolate.mwlmc5", cachename=".slgrid_sph_cache_halo_mwlmc5")
        basis = pyEXP.basis.Basis.factory(halo_config)

        if i==init_snap:
            basis, coefs = make_a_BFE(pos_halo_tr, mass_tr/np.sum(mass_tr), basis_id='sphereSL', time=i*0.02, numr=500, rmin=0.01, rmax=rcut_halo, lmax=lmax, nmax=nmax, scale=rs, modelname="SLGrid.empirical.halo.isolate." + coef_filename, cachename=".slgrid_sph_cache_" + coef_filename, add_coef=False, coef_file= coef_filename + '.h5', empirical=False)
            #compute_coefs(halo_config, mass_tr, pos_halo_tr, i*0.02, out_name, compname, add_coef = False)
        else:
            basis, coefs = make_a_BFE(pos_halo_tr, mass_tr/np.sum(mass_tr), basis_id='sphereSL', time=i*0.02, numr=500, rmin=0.01, rmax=rcut_halo, lmax=lmax, nmax=nmax, scale=rs, modelname="SLGrid.empirical.halo.isolate." + coef_filename, cachename=".slgrid_sph_cache_" + coef_filename, add_coef=True, coef_file= coef_filename + '.h5', empirical=False)
            #compute_coefs(halo_config, mass_tr, pos_halo_tr, i*0.02, out_name, compname, add_coef = True)
   
       """
            # *****  Find bound particles   *****
            #if ((SatBFE == 1) & (SatBoundParticles == 1)):
            #    out_log.write("* Computing satellite bound particles!\n")
                
                # *** Compute satellite bound paticles ***
                #armadillo = lmcb.find_bound_particles(
                #        pos_sat_em, vel_sat_em, mass_sat_em, ids_sat_em, 
                #        sat_rs, nmax_sat, lmax_sat, ncores,
                #        npart_sample = 100000)
                
            #    armadillo = lmcb.find_bound_particles(
            #            pos_sat_em, vel_sat_em, mass_sat_em, ids_sat_em, 
            #            sat_rs, nmax_sat, lmax_sat, args.n_cores, npart_sample=1000000)

                # npart_sample sets the number of particles to compute the
                # potential in each cpu more than 100000 usually generate memory
                # errors

                # removing old variables
                #del(pos_sat_em)
                #del(vel_sat_em)
                

                #out_log.write('Done: Computing satellite bound particles! \n')
                #pos_bound = armadillo[0]
                #vel_bound = armadillo[1]
                #ids_bound = armadillo[2]
                #pos_unbound = armadillo[3]
                #vel_unbound = armadillo[4]
                #ids_unbound = armadillo[5]
                #rs_opt = armadillo[6]
                # mass arrays of bound and unbound particles
                #_part_bound = len(ids_bound)
                #N_part_unbound = len(ids_unbound)
                #mass_bound_array = np.ones(N_part_bound)*mass_sat_em[0]
                #mass_unbound_array = np.ones(N_part_unbound)*mass_sat_em[0]
                #out_log.write("Satellite particle mass {} \n".format(mass_sat_em[0]))    
                # Mass bound fractions
                #Mass_bound = (N_part_bound/len(ids_sat_em))*np.sum(mass_sat_em)
                #Mass_unbound = (N_part_unbound/len(ids_sat_em))*np.sum(mass_sat_em)
                #Mass_fraction = N_part_bound/len(ids_sat_em)
                #out_log.write("Satellite bound mass fraction {} \n".format(Mass_fraction))
                #out_log.write("Satellite bound mass {} \n".format(Mass_bound))
                #out_log.write("Satellite unbound mass {} \n".format(Mass_unbound))



            
                
