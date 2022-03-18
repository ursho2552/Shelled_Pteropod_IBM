#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 16:47:18 2022

@author: ursho
"""
from parcels import ErrorCode
import spIBM.population_module as population_module
import spIBM.parcels_module as parcels_module
import spIBM.coupler_module as coupler_module
from pathlib import Path
from tqdm import tqdm, trange
import os
import numpy as np
import datetime

    

def run_IBM_idealized(Config_param,my_pteropods,start_gen=0,time=5000,L_t=None,save_population=False,save_abundance=False):
    """This function runs the shelled pteropod IBM using idealized conditions and without advection and DVM.
        
    Keyword arguments: 
    Config_param -- dataclass object containing the paths for idealized environmental conditions, and model parameters
    my_pteropods -- initial pteropod population
    start_gen -- generation of the initial pteropod population (default: 0)
    time -- number of days to run the IBM (default: 5000)
    L_t -- size of the pteropod (in mm) as a function of time (default: None). For the default value we use the growth function from Wang et al. 2017
    save_population -- boolean flag indicating if the pteropod population of each simulation day should be saved (default: False)
    save_abundance -- boolean flag indicating if the abundance time series should be saved at the end of the simulation (default: False)
    """
    sst_file = Config_param.dir_env+Config_param.sst_file
    food_file = Config_param.dir_env+Config_param.food_file

    assert Path(sst_file).is_file(), 'File containing idealized Temperature does not exist'
    assert Path(food_file).is_file(), 'File containing idealized Chlorophyll does not exist'

    daily_sst = np.genfromtxt(sst_file,delimiter=',')
    daily_food = np.genfromtxt(food_file,delimiter=',')

    assert daily_sst.shape[0] == 365, 'Temperature data has the wrong size. Should be 365 days'
    assert daily_food.shape[0] == 365, 'Chlorophyll data has the wrong size. Should be 365 days'

    
    stage_0 = np.full((9, time), 0.0)
    stage_1 = np.full((9, time), 0.0)
    
    number_of_individuals = my_pteropods.shape[0]
    current_gen = start_gen
    next_ID = number_of_individuals

    if start_gen == 0:
        stage_0[0,0] = number_of_individuals
        stage_0[4,0] = 1.0
    elif start_gen == 1:
        stage_1[0,0] = number_of_individuals
        stage_1[4,0] = 1.0

    current_gen = start_gen
    next_ID = number_of_individuals
    
    #spring generation
    rate_g0_0 = Config_param.rateG00
    rate_g0_1 = Config_param.rateG01
    rate_g0_2 = Config_param.rateG02
    rate_g0_3 = Config_param.rateG03

    #overwintering generation
    rate_g1_0 = Config_param.rateG10
    rate_g1_1 = Config_param.rateG11
    rate_g1_2 = Config_param.rateG12
    rate_g1_3 = Config_param.rateG13
    
    mynumeggs = Config_param.num_eggs
    delta_ERR = Config_param.delta_ERR
    day_start = Config_param.day_start_initial_eggs
    T0 = Config_param.T0
    Ks = Config_param.Ks

    if L_t is None:
        L_t = population_module.calculate_growth_fct()
    tbar = trange(1,time, leave=True)
    for i in tbar:
        #define optimal conditions
        my_pteropods[:,13] = 4
        T = daily_sst[(day_start+i)%365]
        F = daily_food[(day_start+i)%365]
        #mortality
        die_list,my_pteropods = population_module.mortality(my_pteropods,rate_g0_0,rate_g0_1,rate_g0_2,rate_g0_3,
                           rate_g1_0,rate_g1_1,rate_g1_2,rate_g1_3)

        #growth
        my_pteropods = population_module.shell_growth(my_pteropods,L_t,Arag=4,T=T,F=F,T0=T0,Ks=Ks)
        #development
        my_pteropods = population_module.development(my_pteropods,L_t)
        #spawning events
        my_pteropods,next_ID,current_gen = population_module.spawning(my_pteropods,current_gen,next_ID,num_eggs=mynumeggs,delta_ERR=delta_ERR)
        #set food sum to any number that is not zero (Otherwise the particle is recognized as "beached" and removed
        my_pteropods[:,16] = 1
        #accounting
        stage_0[0,i] = np.argwhere((my_pteropods[:,2] == 0) & (my_pteropods[:,1]%2 == 0) & (my_pteropods[:,5] == 1)).shape[0]
        stage_0[1,i] = np.argwhere((my_pteropods[:,2] == 1) & (my_pteropods[:,1]%2 == 0) & (my_pteropods[:,5] == 1)).shape[0]
        stage_0[2,i] = np.argwhere((my_pteropods[:,2] == 2) & (my_pteropods[:,1]%2 == 0) & (my_pteropods[:,5] == 1)).shape[0]
        stage_0[3,i] = np.argwhere((my_pteropods[:,2] == 3) & (my_pteropods[:,1]%2 == 0) & (my_pteropods[:,5] == 1)).shape[0]
        stage_0[4:8,i] = stage_0[:4,i]/np.sum(stage_0[:4,i],axis=0) if np.sum(stage_0[:4,i],axis=0)  > 0 else 0
        tmp = my_pteropods[np.argwhere(my_pteropods[:,1]%2 == 0),3]
        if tmp.size:
            stage_0[8,i] = np.nanmedian(tmp)

        stage_1[0,i] = np.argwhere((my_pteropods[:,2] == 0) & (my_pteropods[:,1]%2 == 1) & (my_pteropods[:,5] == 1)).shape[0]
        stage_1[1,i] = np.argwhere((my_pteropods[:,2] == 1) & (my_pteropods[:,1]%2 == 1) & (my_pteropods[:,5] == 1)).shape[0]
        stage_1[2,i] = np.argwhere((my_pteropods[:,2] == 2) & (my_pteropods[:,1]%2 == 1) & (my_pteropods[:,5] == 1)).shape[0]
        stage_1[3,i] = np.argwhere((my_pteropods[:,2] == 3) & (my_pteropods[:,1]%2 == 1) & (my_pteropods[:,5] == 1)).shape[0]
        stage_1[4:8,i] = stage_1[:4,i]/np.sum(stage_1[:4,i],axis=0) if np.sum(stage_1[:4,i],axis=0) > 0 else 0
        tmp = my_pteropods[np.argwhere(my_pteropods[:,1]%2 == 1),3]        
        if tmp.size:
            stage_1[8,i] = np.nanmedian(tmp)
        my_pteropods[:,12] = i
        
        individuals = np.sum(stage_0[:4,i],axis=0)+np.sum(stage_1[:4,i],axis=0)
        tbar.set_description(f'{individuals} Individuals')
        tbar.refresh
#        print('Done timestep {} or day {} with {} individuals'.format(i,(i+day_start)%365,individuals))
        if save_population:
            if not os.path.exists(Config_param.output_dir_initialization):
                os.makedirs(Config_param.output_dir_initialization)

            np.savetxt(Config_param.output_dir_initialization+Config_param.out_ptero_file.format(i), my_pteropods, delimiter=',')
            
    if save_abundance:
        if not os.path.exists(Config_param.output_dir_initialization):
                os.makedirs(Config_param.output_dir_initialization)
                
        np.savetxt(Config_param.output_dir_initialization+Config_param.gen0_file, stage_0, delimiter=',')    
        np.savetxt(Config_param.output_dir_initialization+Config_param.gen1_file, stage_1, delimiter=',')

        

        return


def run_IBM_coupled(Config_param, pset, fieldset, pclass, kernels, time_mat,next_ID, current_gen, L_t=None):
    """This function runs the shelled pteropod IBM using modeled/observed environmental conditions and with a defined kernel for movement and interation with the environment.
        
    Keyword arguments: 
    Config_param -- dataclass object containing the paths for idealized environmental conditions, and model parameters
    pset -- Ocean Parcels particle object containing the initial population with initialized attributes
    fieldset -- Ocean Parcels fieldset object defining the environmental conditions
    pclass -- Ocean Parcels particle class
    kernels -- Ocean Parcels kernel. Defines how the particels move and interact with the environment
    time_mat -- array containing the year on the first row, the day on the second row, and the number of days after the beginning of the simulation period
    next_ID -- the unique identifier for the next ID
    current_gen -- identifier for the current generation
    L_t -- size of the pteropod (in mm) as a function of time (default: None). For the default value we use the growth function from Wang et al. 2017
    """

    #spring generation
    rate_g0_0 = Config_param.rateG00
    rate_g0_1 = Config_param.rateG01
    rate_g0_2 = Config_param.rateG02
    rate_g0_3 = Config_param.rateG03

    #overwintering generation
    rate_g1_0 = Config_param.rateG10
    rate_g1_1 = Config_param.rateG11
    rate_g1_2 = Config_param.rateG12
    rate_g1_3 = Config_param.rateG13
    
    dir_env = Config_param.dir_env
    food_file = Config_param.food_file
    daily_food = np.genfromtxt(dir_env+food_file,delimiter=',')
    b = max(daily_food)
    a = min(daily_food)
    num_eggs = Config_param.num_eggs
    Ks = Config_param.Ks
    T0 = Config_param.T0
    
    if L_t is None:
        L_t = population_module.calculate_growth_fct()
    
    flag_init = 1
    day_counter = 0
    
    tbar = trange(time_mat.shape[1], leave=True)
    for i in tbar:

        year = np.squeeze(time_mat[0,i]).astype(int)
        
        if year > Config_param.start_year:
            day_counter = 1
            Config_param.start_year = year
        else:
            day_counter += 1
        output_dir = Config_param.output_dir_simulation + 'year_{}_V_{}/'.format(year,Config_param.version)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filename_day = output_dir+"JitPtero_Day_{}".format(day_counter)
        if flag_init == 1:
            tbar.set_description('Initializing for the first time')
            tbar.refresh
            pset = coupler_module.prepare_particles(pset,fieldset,year)
            flag_init = 0
        
        tbar.set_description(f'Day {day_counter}: Advection')
        tbar.refresh
        
        pset.execute(kernels,runtime=datetime.timedelta(days=1),dt=datetime.timedelta(hours=1.0),\
                output_file=pset.ParticleFile(name=filename_day, outputdt=datetime.timedelta(hours=1.0)),\
                verbose_progress=False,recovery={ErrorCode.ErrorThroughSurface: parcels_module.ReturnToSurface,ErrorCode.ErrorOutOfBounds: parcels_module.PushToWater})
        
        tbar.set_description(f'Day {day_counter}: Mortality')
        tbar.refresh
        my_data = coupler_module.convert_to_mat(pset)
        
        day_vec = np.array([year,Config_param.version,day_counter,Config_param.control]).astype(int)
        outfile_mort = Config_param.outfile_mort
        
        die_list,my_data = population_module.mortality(my_data,rate_g0_0,rate_g0_1,rate_g0_2,rate_g0_3,rate_g1_0,rate_g1_1,rate_g1_2,rate_g1_3,day=day_vec,outfile=outfile_mort)
        tbar.set_description(f'Day {day_counter}: Deletion')
        tbar.refresh
        coupler_module.get_dead_particles(pset,die_list)
        
        
        tbar.set_description(f'Day {day_counter}: Growth')
        tbar.refresh
        if my_data.shape[0] < 1:
            print('All pteropods are dead')
            break
        
        mean_food = np.squeeze(my_data[:,16])
        
        F_scaled = a + (mean_food - 0.05)*(b-a) / (0.9 - 0.05)
        outfile_growth = Config_param.outfile_growth
        
        my_data = population_module.shell_growth(my_data,L_t,Arag=np.squeeze(my_data[:,13]),
                               T=np.squeeze(my_data[:,15]),
                               F=F_scaled,T0=T0,Ks=Ks,day=day_vec,outfile=outfile_growth)
        
        tbar.set_description(f'Day {day_counter}: Development')
        tbar.refresh
        my_data = population_module.development(my_data,L_t)
        
        #spawning events
        tbar.set_description(f'Day {day_counter}: Spawning')
        tbar.refresh
        my_data,next_ID,current_gen = population_module.spawning(my_data,current_gen,next_ID,num_eggs=num_eggs)
        
        #update pset_ptero
        tbar.set_description(f'Day {day_counter}: Updating')
        tbar.refresh
        pset = coupler_module.update_particleset(my_data,pset,fieldset,pclass,year,i,die_list)
        
        #save my_Data as csv file
        tbar.set_description(f'Day {day_counter}: Saving')
        tbar.refresh
        output_dir_table =  Config_param.output_tables + 'year_{}_V_{}'.format(year,Config_param.version)
        
        if not os.path.exists(output_dir_table):
            os.makedirs(output_dir_table)
        filename_day_table = output_dir_table+"/JitPtero_Day_{}.csv".format(day_counter)
        np.savetxt(filename_day_table, my_data, delimiter=',')
        
        tbar.set_description(f'Day {day_counter}: Done')
        tbar.refresh
            
    return


def run_physics_only(Config_param, pset, fieldset, kernel, year, total_runtime=3, dt=1.0, outputdt=None):
    """This function runs the movement and interaction with the environment (kernel) of particles without the mortality, growth, development and spawing functions.
    The function return an Ocean Parcels particle object with adapted attributes
    
        
    Keyword arguments: 
    Config_param -- dataclass object containing the paths for idealized environmental conditions, and model parameters
    pset -- Ocean Parcels particle object containing the initial population with initialized attributes
    fieldset -- Ocean Parcels fieldset object defining the environmental conditions
    kernels -- Ocean Parcels kernel. Defines how the particels move and interact with the environment
    year -- year of the simulation
    total_runtime -- number of days to run the model using only physics and without population dynamics (default: 3 days)
    dt -- sub-time-step to run the kernel (default: 1 hour)
    outputdt -- time-step at which the physics only run is saved (default: None, and uses dt)
    """
    
    assert dt <= 24, "The sub time-step should be smaller than the model time-step (1 day)"
    if outputdt is None:
        outputdt = dt
    
    if not os.path.exists(Config_param.output_dir_physics):
        os.makedirs(Config_param.output_dir_physics)
    
    for i in tqdm(range(total_runtime), desc='Physics only progress'):
        
        filename_day = Config_param.output_dir_physics+Config_param.physics_only_file.format(i)
        
        pset = coupler_module.prepare_particles(pset,fieldset,year)

        outfile = None if outputdt==False else pset.ParticleFile(name=filename_day, outputdt=datetime.timedelta(hours=outputdt))

        
        pset.execute(kernel,runtime=datetime.timedelta(days=1.0),dt=datetime.timedelta(hours=dt),\
                       recovery={ErrorCode.ErrorThroughSurface: parcels_module.ReturnToSurface,ErrorCode.ErrorOutOfBounds: parcels_module.PushToWater},\
                       verbose_progress=False,output_file=outfile)
        
    return pset

