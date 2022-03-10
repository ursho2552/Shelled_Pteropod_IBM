#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 16:50:58 2022
Scripts used to read the user input parameters for the shelled pteropod IBM, and to initialize the model run
@author: ursho
"""

from parcels import FieldSet, Field, ParticleSet
import numpy as np
import xarray as xr
from dataclasses import dataclass
import yaml
import argparse
import scipy.stats


@dataclass
class Config_parameters():
    """Class to define paths and model parameters used in the model
    
    Keyword arguments:
    None

    """ 
    
    data_path: str
    mesh_file: str

    directory_mort: str
    similarity_file: str

    output_dir_initialization: str
    gen0_file: str
    gen1_file: str
    out_ptero_file: str
    initial_positions_file: str

    reference_abundance_data: str

    output_dir_physics: str
    physics_only_file: str

    output_dir_simulation_scratch: str
    outfile_mort_scratch: str
    outfile_growth_scratch: str
    output_tables_scratch: str

    output_dir_simulation: str
    outfile_mort: str
    outfile_growth: str
    output_tables: str

    dir_env: str
    sst_file: str
    food_file: str

    velocity_file: str
    aragonite_file: str
    extreme_file: str
    oxygen_file: str
    temperature_file: str
    chlorophyll_file: str
    depth_file: str
    unbeach_file: str
    mask_file: str
    
    velocity_U_variable_name: str
    velocity_V_variable_name: str
    velocity_W_variable_name: str
    aragonite_variable_name: str
    extreme_variable_name: str
    temperature_variable_name: str
    oxygen_variable_name: str
    chlorophyll_variable_name: str
    depth_variable_name: str
    mask_variable_name: str
    unbeach_lat_variable_name: str
    unbeach_lon_variable_name: str  
    
    lon_name: str
    lat_name: str
    depth_name: str
    time_name: str

    flag_calculate_initial_population: bool
    flag_calculate_initial_positions: bool
    flag_run_physics_only: bool
    
    #control and version should not be here
    control: int
    start_year: int
    version: int
    start_day: int
    num_init: int
    day_start_initial_eggs: int
    seed: int
    rateG00: float
    rateG01: float
    rateG02: float
    rateG03: float
    rateG10: float
    rateG11: float
    rateG12: float
    rateG13: float
    num_eggs: int
    delta_ERR: float
    Ks: float
    T0: float
    Tmax: float
    
def read_config_files(config_file,config_class=Config_parameters):
    """This function reads a configuration file, and fills in the attributes of the dataclass with the respective entries in the configuratio file
    
    Keyword arguments:
    config_file -- path to configuration file. Should be a yaml file
    config_class -- dataclass (default: Config_parameters dataclass defined above)
    """ 
    assert '.yaml' in config_file.lower(), "The configuration file should be a '.yaml' file"
    
    with open(config_file) as file:
        config_list = yaml.load(file, Loader=yaml.FullLoader)

    config = config_class(**config_list)

    return config

def parse_inputs():
    """This reads user input from the terminal to start running the shelled pteropod IBM
    
    """ 

    parser = argparse.ArgumentParser(description="Run shelled pteropod IBM")
    parser.add_argument("--year", required=True, type=int,
                        help="Year of the simulation. Year should coincide with name of file.")
    parser.add_argument("--version", required=True, type=int,
                        help="Version of the run. This integer is used to set the random seed.")
    parser.add_argument("--control", required=False, nargs='?',
                        const=0, default=0, type=int,
                        help="Determine which scenario is used (0: with extremes; 1: without extremes).")
    parser.add_argument("--config_file", required=False, nargs='?',
                        const="IBM_config_parameters.yaml",
                        default="IBM_config_parameters.yaml", type=str,
                        help="Yaml file containing the paths and parameters needed for the IBM.")

    args = parser.parse_args()



    return args.year, args.version, args.control, args.config_file




def read_environment(Config_param, year,control=0):
    """This function defines the environmental conditions for the coupled simulation.
    The function defines an Ocean Parcels fieldset with all environmental conditions needed for the shelled pteropod IBM to run
    
    Keyword arguments:
    Config_param -- dataclass containing all paths and parameters
    year -- year of the simulation. Should correspond to the names of the files
    control -- flag to determine if a 'control' (control=1) field for aragonite is used instead of the original one (control=0)
    
    """
    data_path = Config_param.data_path
    mesh_file = Config_param.mesh_file
    
    velocity_file = data_path+Config_param.velocity_file.format(year)
    if control == 1:
        aragonite_file = data_path+Config_param.extreme_file.format(year)
    else:
        aragonite_file = data_path+Config_param.aragonite_file.format(year)
        
    extreme_file = data_path+Config_param.extreme_file.format(year)
    
    oxygen_file = data_path+Config_param.oxygen_file.format(year)
    temperature_file = data_path+Config_param.temperature_file.format(year)
    chlorophyll_file = data_path+Config_param.chlorophyll_file.format(year)
    depth_file = data_path+Config_param.depth_file
    
    unbeach_file = data_path+Config_param.unbeach_file
    
    filenames = {'U': {'lon': mesh_file, 'lat': mesh_file, 'depth': depth_file, 'data': velocity_file},
                 'V': {'lon': mesh_file, 'lat': mesh_file, 'depth': depth_file, 'data': velocity_file},
                 'W': {'lon': mesh_file, 'lat': mesh_file, 'depth': depth_file, 'data': velocity_file},
                 'arag':{'lon': mesh_file, 'lat': mesh_file, 'depth': depth_file, 'data': aragonite_file},
                 'extremes':{'lon': mesh_file, 'lat': mesh_file, 'depth': depth_file, 'data': extreme_file},
                 'temp':{'lon': mesh_file, 'lat': mesh_file, 'depth': depth_file, 'data': temperature_file},
                 'O2':{'lon': mesh_file, 'lat': mesh_file, 'depth': depth_file, 'data': oxygen_file},
                 'Chl':{'lon': mesh_file, 'lat': mesh_file, 'depth': depth_file, 'data': chlorophyll_file},
                 'Mydepth':{'lon': mesh_file, 'lat': mesh_file, 'depth': depth_file, 'data': depth_file},
                 'mask': {'lon': mesh_file, 'lat': mesh_file, 'data': unbeach_file},
                 'unBeach_lat': {'lon': mesh_file, 'lat': mesh_file, 'data': unbeach_file},
                 'unBeach_lon': {'lon': mesh_file, 'lat': mesh_file, 'data': unbeach_file}}
    
    variables = {'U': Config_param.velocity_U_variable_name,
                 'V': Config_param.velocity_V_variable_name,
                 'W': Config_param.velocity_W_variable_name,
                 'arag': Config_param.aragonite_variable_name,
                 'extremes': Config_param.extreme_variable_name,
                 'temp': Config_param.temperature_variable_name,
                 'O2': Config_param.oxygen_variable_name,
                 'Chl': Config_param.chlorophyll_variable_name,
                 'Mydepth': Config_param.depth_variable_name,
                 'mask': Config_param.mask_variable_name,
                 'unBeach_lat': Config_param.unbeach_lat_variable_name,
                 'unBeach_lon': Config_param.unbeach_lon_variable_name} 
 
    lon_name = Config_param.lon_name
    lat_name = Config_param.lat_name
    depth_name = Config_param.depth_name
    time_name = Config_param.time_name
    
    dimensions = {'U': {'lon': lon_name, 'lat': lat_name, 'depth': depth_name, 'time': time_name},
                  'V': {'lon': lon_name, 'lat': lat_name, 'depth': depth_name, 'time': time_name},
                  'W': {'lon': lon_name, 'lat': lat_name, 'depth': depth_name, 'time': time_name},
                  'arag': {'lon': lon_name, 'lat': lat_name, 'depth': depth_name, 'time': time_name},
                  'extremes': {'lon': lon_name, 'lat': lat_name, 'depth': depth_name, 'time': time_name},
                  'temp': {'lon': lon_name, 'lat': lat_name, 'depth': depth_name, 'time': time_name},
                  'O2': {'lon': lon_name, 'lat': lat_name, 'depth': depth_name, 'time': time_name},
                  'Chl': {'lon': lon_name, 'lat': lat_name, 'depth': depth_name, 'time': time_name},
                  'Mydepth': {'lon': lon_name, 'lat': lat_name, 'depth': depth_name},
                  'mask': {'lon': lon_name, 'lat': lat_name},
                  'unBeach_lat': {'lon': lon_name, 'lat': lat_name},
                  'unBeach_lon': {'lon': lon_name, 'lat': lat_name}}
    
    fieldset = FieldSet.from_c_grid_dataset(filenames, variables, dimensions,allow_time_extrapolation=False)
    
    print('Adding lower and upper bounds...')
    fieldset.add_field(Field('bottom_depth', fieldset.Mydepth.depth[-1, :, :], lon=fieldset.Mydepth.grid.lon, lat=fieldset.Mydepth.grid.lat))
    fieldset.add_field(Field('top_depth', fieldset.Mydepth.depth[0, :, :], lon=fieldset.Mydepth.grid.lon, lat=fieldset.Mydepth.grid.lat))

    return fieldset


def define_initial_population(number_of_individuals, start_generation, number_of_attributes=17):
    """This function defines a starting population of eggs
    
    Keyword arguments:
    number_of_individuals -- number of indidivuals in the starting population
    start_generation -- start of the generation
    number_of_attributes -- number of attributes to characterize each individual
    
    """
    
    initial_population = np.random.rand(number_of_individuals, number_of_attributes)
    #ID
    initial_population[:,0] = np.arange(number_of_individuals)
    #generation
    initial_population[:,1] = start_generation
    #stage
    initial_population[:,2] = 0
    #shell_size
    initial_population[:,3] = 0.15
    #days_of_growth
    initial_population[:,4] = 0
    #survive
    initial_population[:,5] = 1
    #num_spawning_events
    initial_population[:,6] = 0
    #ERR
    initial_population[:,7] = np.random.uniform(low=-1, high=1, size=(number_of_individuals))
    #spawned
    initial_population[:,8] = 0
    #Parent_ID
    initial_population[:,9] = -1
    #Parent_shell_size
    initial_population[:,10] = -1
    #time_of_birth
    initial_population[:,11] = -1
    #current_time
    initial_population[:,12] = 0
    #current_time
    initial_population[:,13] = 0
    #accumulated damage
    initial_population[:,14] = 0
    #average temp
    initial_population[:,15] = 0
    #average food
    initial_population[:,16] = 1


    return initial_population



def define_initial_population_dynamic(number_of_individuals, number_of_attributes, dictionary_of_values):
    """This function defines a starting population with attributes defined in a dictionary
    
    Keyword arguments:
    number_of_individuals -- number of indidivuals in the starting population
    number_of_attributes -- number of attributes to characterize each individual
    dictionary_of_values -- dictionary containing values or functions for each attribute
    """
    
    assert len(dictionary_of_values) == number_of_attributes, "The disctionary must contain values for each attribute"
    assert all([k for k in dictionary_of_values.keys()] == np.arange(number_of_attributes)), "The dictionary keys should be the indeces for the columns (attributes) given as integers"
    

    initial_population = np.random.rand(number_of_individuals, number_of_attributes)
    
    for key in dictionary_of_values.keys():
    
        initial_population[:,key] = dictionary_of_values[key]
    
    return initial_population


def determine_starting_day(output_dir,gen0_file,gen1_file,observations,observations_std,start=None):
    """This function determines the starting day given daily simulated abundances and observations and range in observed abundances
    
    Keyword arguments:
    output_dir -- directory of files with modeled abundances
    gen0_file -- file with abundances for the first generation
    gen1_file -- file with abundances for the second generation
    observations -- daily abundance observations
    observations_std -- daily abundance ranges (here the standard deviation is used as an example)
    start -- first day in the modeled abundance that should be considered in the comparison
    """

    stage_0 = np.genfromtxt(output_dir+gen0_file,delimiter=',')
    stage_1 = np.genfromtxt(output_dir+gen1_file,delimiter=',')
    stage_0 = np.nan_to_num(stage_0)
    stage_1 = np.nan_to_num(stage_1)

    cycle1 = stage_1[1:4,:]
    cycle0 = stage_0[1:4,:]
    data = np.sum(cycle1,axis=0)+np.sum(cycle0,axis=0)

    print('Matching to observations...')
    best_mean,best_rolling,start_day,max_Pearson,max_Spearman,min_rmse,outside_range = match_to_observations(data,observations,observations_std,start=None)
    

    
    print('The following metrics were found:')
    print(start_day,max_Spearman,max_Pearson,min_rmse,outside_range)

    print('Start day is: {}'.format(start_day))

    return start_day


def read_attributes_from_file(filename_day_essential,fieldset,pclass):
    """This function reads in the attributes of particels stored as xarray.
    The function is very specific to the project, and should be adapted if the project changes, or a dynamic implementation is needed
    
    Keyword arguments:
    filename_day_essential -- path to file containing all essential information for the particles
    fieldset -- Ocean Parcesl fieldset defining the environmental conditions
    pclass -- Ocean Parcels particle class
    """
    
    ds = xr.open_dataset(filename_day_essential)

    time = ds.time[:,-1].values
    lat = ds.lat[:,-1].values
    lon = ds.lon[:,-1].values
    z = ds.z[:,-1].values
    temp = ds.temp[:,-1].values
    temp_sum = ds.temp_sum[:,-1].values
    food = ds.food[:,-1].values
    food_sum = ds.food_sum[:,-1].values
    oxygen = ds.oxygen[:,-1].values
    oxygen_sum = ds.oxygen_sum[:,-1].values
    arag_exposure = ds.arag[:,-1].values
    arag_exposure_sum = ds.arag_sum[:,-1].values
    damage = ds.damage[:,-1].values
    generation = ds.generation[:,-1].values
    stage = ds.stage[:,-1].values
    survive = ds.survive[:,-1].values
    num_spawning_event = ds.num_spawning_event[:,-1].values
    shell_size = ds.shell_size[:,-1].values

    days_of_growth = ds.days_of_growth[:,-1].values
    ERR = ds.ERR[:,-1].values
    spawned = ds.spawned[:,-1].values
    MyID = ds.MyID[:,-1].values
    Parent_ID = ds.Parent_ID[:,-1].values
    Parent_shell_size = ds.Parent_shell_size[:,-1].values

    extreme = ds.extreme[:,-1].values
    extreme_arag = ds.extreme_arag[:,-1].values

    

    max_ID = np.max(MyID)+1
    current_gen = np.nanmax(generation[np.squeeze(np.argwhere((stage==3) | (shell_size == max(np.unique(shell_size))))).astype(int)])


    pset = ParticleSet(fieldset=fieldset, pclass=pclass,\
                            time=time,\
                            lat=lat,\
                            lon=lon,\
                            depth=z,\
                            temp=temp,\
                            temp_sum=temp_sum,\
                            food=food,\
                            food_sum=food_sum,\
                            oxygen=oxygen,\
                            oxygen_sum=oxygen_sum,\
                            arag_exposure=arag_exposure,\
                            arag_exposure_sum=arag_exposure_sum,\
                            damage=damage,\
                            generation=generation,\
                            stage=stage,\
                            survive=survive,\
                            num_spawning_event=num_spawning_event,\
                            shell_size=shell_size,\
                            days_of_growth=days_of_growth,\
                            ERR=ERR,\
                            spawned=spawned,\
                            MyID=MyID,\
                            Parent_ID=Parent_ID,\
                            Parent_shell_size=Parent_shell_size,\
                            extreme=extreme,\
                            extreme_arag=extreme_arag,\
                            step_counter=extreme_arag*0 + 24,\
                            lonlatdepth_dtype=np.float32)

    return pset, max_ID, current_gen


def reset_particle_attributes(pset,initial_population,dictionary):
    """This function resets the attributes of a particle set to those provided in a matrix.

    
    Keyword arguments:
    pset -- Ocean Parcels particleset
    initial_population -- array containing the values 
    dictionary -- dictionary containing the attributes of pset to change (key) and values to change
    """

    for key in dictionary.keys():
        pset.particle_data[key][:] = dictionary[key]
    
#    pset.particle_data['time'][:] = 0.0
#    pset.particle_data['stage'][:] = initial_population[:,2]
#    pset.particle_data['survive'][:] = initial_population[:,5]
#    pset.particle_data['num_spawning_event'][:] = initial_population[:,6]
#    pset.particle_data['generation'][:] = initial_population[:,1]
#    pset.particle_data['shell_size'][:] = initial_population[:,3]
#    pset.particle_data['days_of_growth'][:] = initial_population[:,4]
#    pset.particle_data['ERR'][:] = initial_population[:,7]
#    pset.particle_data['spawned'][:] = initial_population[:,8]
#    pset.particle_data['Parent_ID'][:] = initial_population[:,9]
#    pset.particle_data['Parent_shell_size'][:] = initial_population[:,10]
#    pset.particle_data['MyID'][:] = initial_population[:,0]
#    pset.particle_data['damage'][:] = initial_population[:,14]
        
    return pset

def initialize_particles(fieldset, pclass, initial_population, locations):
    """This function initializes a particle set. Function is specific to the project, and should be adapted for other projects

    Keyword arguments:
    fieldset -- Ocean Parcels particleset
    pclass -- Ocean Parcels particle class
    initial_population -- Initial values for particel attributes
    locations -- locations of pteorpods, lons in first column, lats in second column, depth in the third column for each pteropod
    
    """
    assert locations.shape[0] == initial_population.shape[0], "The number of entries in the initial population and the locationas is not the same"
    
    depths = locations[:,2].astype(np.float32)
    lats = locations[:,1].astype(np.float32)
    lons = locations[:,0].astype(np.float32)


    pset = ParticleSet(fieldset=fieldset, pclass=pclass, lon=lons,\
                   lat=lats,depth=depths,\
                   time=0.0,stage=initial_population[:,2],survive=initial_population[:,5],num_spawning_event=initial_population[:,6],\
                   generation=initial_population[:,1], shell_size=initial_population[:,3], days_of_growth=initial_population[:,4],\
                   ERR=initial_population[:,7], spawned=initial_population[:,8], Parent_ID=initial_population[:,9], Parent_shell_size=initial_population[:,10],\
                   MyID=initial_population[:,0],lonlatdepth_dtype=np.float32)
    
    return pset




def moving_average(data_set, w=30):
    assert w > 0
    assert w%2 == 0
    
    N = data_set.shape[0]
    moving_av = np.ones((N,))*np.nan
    
    before_after = int(w/2)
    start_pos = int(w/2)
    end_pos = int(data_set.shape[0]-before_after)
    for i in range(start_pos,end_pos):
        moving_av[i] = np.nanmean(data_set[i-before_after:i+before_after])
    
        
    return moving_av

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


    
    
def match_to_observations(data,observations,observations_std,start=None):
    """This function calculates the optimal pattern match up between the data from the model and obsevations data.
    The similarity is calculated based on the Pearson, Spearman correlation coefficients, the Manhattan distance, and
    the range of modeled abundances outside of the range of observed abundances. Function returns the mean, rolling mean, the start day, and similarity metrics
    
    Keyword arguments:
    data -- modeled abudances
    observations -- observed abundances
    observations_std -- range of observed abundances
    start -- first day to consider in data for the comparison (default: None, the comparison is only done for the last third of data)
    """
    
    assert len(observations) == len(observations_std), "The observations and the observations_std should have the same size"
    
    if start is None:
        start = int(data.shape[0]*2/3)
    
    
    std = abs(observations-observations_std)
    min_daily_abundance_unit = (observations - std)/max(observations)
    max_daily_abundance_unit = (observations + std)/max(observations)

    daily_abundance_unit = observations/max(observations)
    max_Pearson = 0
    max_Spearman = 0
    min_start = start
    min_L1 = 100000000
    min_outside_range = 100000000
    sum_factors = 0
    corrected_min_start = min_start
    best_mean = np.zeros((365,))
    best_rolling = np.zeros((365,))
    
    for i in range(400):
        my_data = data[min_start+i:]
        mean_data = np.zeros((365,))
        counter = np.zeros((365,))
        if min_start+i+365*3 < data.shape[0]:
            for j in range(my_data.shape[0]):
                
                ii = j%365
                mean_data[ii] += my_data[j]
                counter[ii] += 1
        mean_data = mean_data/counter/np.nanmax(mean_data)
        #calculate rolling mean
        rolling_mean = moving_average(np.hstack((mean_data,mean_data,mean_data)),30)[365:365*2]
        rolling_mean_unit = rolling_mean/max(rolling_mean)
        
        PearsonR = scipy.stats.pearsonr(rolling_mean_unit,daily_abundance_unit)[0]
        SpearmanR = scipy.stats.spearmanr(rolling_mean_unit,daily_abundance_unit)[0]
        
        L1 = np.sum(abs(rolling_mean_unit-daily_abundance_unit)) 
        outside_range = len(np.argwhere((rolling_mean_unit > max_daily_abundance_unit) | (rolling_mean_unit < min_daily_abundance_unit)))/len(daily_abundance_unit)
        
        
        if sum_factors <  1-(L1/365) + PearsonR + SpearmanR - outside_range:
            min_L1 = L1
            min_outside_range = outside_range
            corrected_min_start = min_start+i
            max_Pearson = PearsonR
            max_Spearman = SpearmanR
            sum_factors =  1-(L1/365) + PearsonR + SpearmanR - outside_range
            best_mean = mean_data.copy()
            best_rolling = rolling_mean.copy()
            
    return [best_mean,best_rolling,corrected_min_start,max_Pearson,max_Spearman,min_L1,min_outside_range]

