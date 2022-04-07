#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 00:14:12 2022

Main file to calculate results for publication

Pteropod IBM ECOMOD

Individual-based modelling of shelled pteropods. Urs Hofmann Elizondo and
Meike Vogt, 2022

@author: ursho
"""
from dataclasses import asdict
import csv
import datetime
import importlib
import os
import sys

import numpy as np

sys.path.insert(1,"/net/kryo/work/ursho/PhD/Projects/Pteropod_IBM/Shelled_Pteropod_IBM/")
import spIBM
import project_funcs

MODULE_PATH = '/home/ursho/PhD/Projects/Pteropods/My_parcels/Parcels_master_copy/parcels/parcels/__init__.py'
MODULE_NAME = "parcels"
spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

from parcels import ParticleSet

def main():
    # =========================================================================
    # Read in user input from terminal
    # =========================================================================
    year, version, control, config_file = spIBM.parse_inputs()

    # =========================================================================
    # Read YAML file with all parameter values and fill in the control and
    # version field
    # =========================================================================
    My_config = spIBM.read_config_files(config_file)
    My_config.control = control
    My_config.version = version

    # =========================================================================
    # Read environment
    # =========================================================================
    fieldset = spIBM.read_environment(My_config, year, My_config.control)

    # =========================================================================
    # Calculate initial idealized population
    # =========================================================================
    if not os.path.exists(My_config.output_dir_initialization):
        os.makedirs(My_config.output_dir_initialization)
        np.random.seed(seed=My_config.seed)

    NUMBER_OF_INDIVIDUALS = 15000
    START_GENERATION = 0
    NUMBER_OF_ATTRIBUTES = 17

    my_pteropods = \
        spIBM.define_initial_population(
                number_of_individuals=NUMBER_OF_INDIVIDUALS,
                start_generation=START_GENERATION,
                number_of_attributes=NUMBER_OF_ATTRIBUTES)

    # =========================================================================
    #                            Alternative
    #
    # my_attributes = {0: np.arange(number_of_individuals),
    #       1: start_generation, 2: 0, 3: 0.15, 4: 0, 5: 1,
    #       6: 0, 7: np.random.uniform(low=-1,high=1, size=(number_of_individuals)),
    #       8: 0, 9: -1, 10: -1, 11: -1, 12: 0, 13: 0, 14: 0, 15: 0, 16: 1}
    #my_pteropods =
    #    spIBM.define_initial_population_dynamic(
    #        number_of_individuals=number_of_individuals,
    #        number_of_attributes=number_of_attributes,
    #        dictionary_of_values=my_attributes)
    # =========================================================================

    if My_config.flag_calculate_initial_population:
        spIBM.run_ibm_idealized(My_config, my_pteropods, start_gen=0,
                                time=5000, length_t=None, save_population=True,
                                save_abundance=True)

    # =========================================================================
    # Determine starting day given the abundances calculated above
    # This part requires external validation data (e.g. from MAREDAT)
    # =========================================================================
    REF_DATA_FILE ="/home/ursho/PhD/Projects/Pteropod_IBM/Data/MarEDat20120203Pteropods.nc"
    daily_abundance_maredat, std_abundance_maredat = \
        project_funcs.get_daily_maredat_obs(ref_data=REF_DATA_FILE)

#    directory_mort = My_config.directory_mort
#    similarity_file = My_config.similarity_file
    output_dir = My_config.output_dir_initialization
    gen0_file = My_config.gen0_file
    gen1_file = My_config.gen1_file

    My_config.start_day = \
        spIBM.determine_starting_day(output_dir, gen0_file, gen1_file,
                                     daily_abundance_maredat,
                                     std_abundance_maredat,
                                     start=None)

    # =========================================================================
    # Read initial idealized population at the start day
    # =========================================================================
    initial_population = \
        np.genfromtxt(output_dir + \
                      '/Pteropods_Day_{}.csv'.format(int(My_config.start_day)),
                      delimiter=',')
    num_init = initial_population.shape[0]

    # =========================================================================
    # Get the initial random positions (only calculate once for the first year)
    # =========================================================================
    grid_file = My_config.mesh_file
    outfile = My_config.output_dir_initialization + \
              My_config.initial_positions_file

    np.random.seed(seed=My_config.version*5)

    #Ideally this is done once for the very first year, then only read from
    #file later on
    if My_config.flag_calculate_initial_positions:
        latlon_list = \
            project_funcs.get_initial_positions(num=num_init,
                                                grid_file=grid_file,
                                                outfile=outfile)

    latlon_list = np.genfromtxt(outfile, delimiter=',')

    # =========================================================================
    # Initialize particles and kernel
    # =========================================================================
    pclass = spIBM.PteropodParticle
    pset_ptero = spIBM.initialize_particles(fieldset,pclass,
                                            initial_population, latlon_list)
    kernel = pset_ptero.Kernel(spIBM.pteropod_kernel)


    # =========================================================================
    # Run physics only initialization, and reset times
    # =========================================================================
    if My_config.flag_run_physics_only:
        pset_ptero = spIBM.run_physics_only(My_config, pset_ptero, fieldset,
                                            kernel, year, total_runtime=3,
                                            dt=1.0, outputdt=1.0)

    #always read from file. On the first year calculate the value and then read
    #from file
    my_file = My_config.output_dir_physics + \
              My_config.physics_only_file.format(My_config.version)
    pset_ptero = ParticleSet.from_particlefile(fieldset=fieldset,
                                               pclass=pclass,
                                               filename=my_file,
                                               lonlatdepth_dtype=np.float32)


    #Dynamic version of reset_particle_attributes
    reset_dictionary = {'time': 0.0,
                        'MyID': initial_population[:,0],
                        'generation': initial_population[:,1],
                        'stage': initial_population[:,2],
                        'shell_size': initial_population[:,3],
                        'days_of_growth': initial_population[:,4],
                        'survival': initial_population[:,5],
                        'num_spawning_event': initial_population[:,6],
                        'ERR': initial_population[:,7],
                        'spawned': initial_population[:,8],
                        'Parent_ID': initial_population[:,9],
                        'Parent_shell_size': initial_population[:,10],
                        'damage': initial_population[:,14]}

    pset_ptero = spIBM.reset_particle_attributes(pset_ptero,
                                                 initial_population,
                                                 reset_dictionary)

    # =========================================================================
    # Run coupled model
    # =========================================================================
    print('Starting simulation...')
    next_ID = max(initial_population[:,0])+1
    print('Shape initial:', initial_population.shape)
    print('Next ID is {}'.format(next_ID))

    oldest_life_stage = initial_population[:,2] == 3
    largest_individual = initial_population[:,3] == max(np.unique(initial_population[:,3]))
    mask_current_gen = oldest_life_stage + largest_individual
    current_gen = np.nanmax(initial_population[mask_current_gen,1])

    #define time for which the model should work
    d0 = datetime.date(year,1,1)
    d1 = datetime.date(year,12,31)
    time_mat = np.empty((3,(d1-d0).days))
    for i in range(time_mat.shape[1]):
        time_mat[0,i] = (d0+datetime.timedelta(days=i)).year
        time_mat[1,i] = (d0+datetime.timedelta(days=i)).day
        time_mat[2,i] = i

    pset_ptero.run_ibm_coupled(My_config, pset_ptero, fieldset, pclass,
                               kernel, time_mat, next_ID, current_gen,
                               length_t=None)

    # =========================================================================
    # Save model parameters used for the year
    # =========================================================================
    Parameters_dict = asdict(My_config)
    with open(My_config.output_tables+'/Parameters_{}.csv'.format(year), 'w') as f:
        w = csv.DictWriter(f, Parameters_dict.keys())
        w.writeheader()
        w.writerow(Parameters_dict)

    sys.exit()

'''
Main Function
'''
if __name__ in "__main__":

    main()
