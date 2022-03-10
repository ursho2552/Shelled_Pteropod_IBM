#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 15:38:12 2022

@author: ursho
"""

from parcels import JITParticle, Variable 
import numpy as np

class PteropodParticle(JITParticle):
    """Definition of the attributes and additional characteristics of the pteropod particle.
    
    Keyword arguments:
    JITParticle -- Ocean Parcels particle class

    """
    
    #Class of particles with attributes needed to keep track of DVM, stage, position
    #velocity, and exposure
    
    # =========================================================================
    # Attributes for DVM
    # =========================================================================
    
    #time at which the particle arrives at the surface
    arrival = Variable('arrival',dtype=np.int64, initial=0)
    #time at which the particle departs from the depth
    departure_from_depth = Variable('departure_from_depth',dtype=np.int64, initial=0)
    #time at which the particle departs from the sruface
    departure = Variable('departure',dtype=np.int64, initial=0)
    #vertical velocities
    up = Variable('up',dtype=np.float32, initial=0.0)
    down = Variable('down',dtype=np.float32, initial=0.0)
    down_wings = Variable('down_wings',dtype=np.float32, initial=0.0)
    max_depth = Variable('max_depth',dtype=np.float32, initial=0.0)
    min_depth = Variable('min_depth',dtype=np.float32, initial=0.0)
    next_max_depth = Variable('next_max_depth',dtype=np.float32, initial=0.0)
    flag_down = Variable('flag_down',dtype=np.int32, initial=0)
    flag_up = Variable('flag_up',dtype=np.int32, initial=0)
    productivity = Variable('productivity', dtype=np.float32, initial=0.0)
    O2_level = Variable('O2_level', dtype=np.float32, initial=0.0)
    chl_ascent = Variable('chl_ascent',dtype=np.float32, initial=-9999.9)
    # =========================================================================
    # Attributes for environment
    # =========================================================================
    temp = Variable('temp',dtype=np.float32, initial=np.nan)
    temp_sum = Variable('temp_sum',dtype=np.float32,initial=0.0)
    food = Variable('food',dtype=np.float32, initial=np.nan)
    food_sum = Variable('food_sum',dtype=np.float32,initial=0.0)
    oxygen = Variable('oxygen',dtype=np.float32, initial=np.nan)
    oxygen_sum = Variable('oxygen_sum',dtype=np.float32,initial=0.0)
    
    arag_exposure = Variable('arag_exposure',dtype=np.float32, initial=np.nan)
    arag_exposure_sum = Variable('arag_exposure_sum',dtype=np.float32,initial=0.0)
    step_counter = Variable('step_counter',dtype=np.int32,initial=0)
    damage = Variable('damage',dtype=np.float32,initial=0.0)
#    death = Variable('death',dtype=np.int32,initial=0)
    # =========================================================================
    # Attributes for pteropod development
    # =========================================================================
    generation = Variable('generation',dtype=np.float32, initial=np.nan)
    stage = Variable('stage',dtype=np.int32, initial=0)
    survive = Variable('survive',dtype=np.int32,initial=1)
    num_spawning_event = Variable('num_spawning_event',dtype=np.int32,initial=0)
    shell_size = Variable('shell_size',dtype=np.float32,initial=0.15)
    days_of_growth = Variable('days_of_growth',dtype=np.float32,initial=0.0)
    ERR = Variable('ERR',dtype=np.float32,initial=0.0)
    spawned = Variable('spawned',dtype=np.int32,initial=0)
    
    # =========================================================================
    # Attributes to determine spawning location
    # =========================================================================
    chl_max = Variable('chl_max',dtype=np.float32, initial=-100.0)
    depth_chl_max = Variable('depth_chl_max',dtype=np.float32, initial=0.0)
    lon_chl_max = Variable('lon_chl_max',dtype=np.float32, initial=0.0)
    lat_chl_max = Variable('lat_chl_max',dtype=np.float32, initial=0.0)
    
    # =========================================================================
    # Attributes to keep track of pteropods with tables
    # =========================================================================
    MyID = Variable('MyID',dtype=np.int32, initial=-1)
    Parent_ID = Variable('Parent_ID',dtype=np.int32,initial=-1)
    Parent_shell_size = Variable('Parent_shell_size',dtype=np.float32,initial=0.0)
    myxi = Variable('myxi',dtype=np.int32,initial=-1)
    myyi = Variable('myyi',dtype=np.int32,initial=-1)
    myzi = Variable('myzi',dtype=np.int32,initial=-1)
    
    reseed_lat = Variable('reseed_lat',dtype=np.float32, initial=0.0)
    reseed_lon = Variable('reseed_lon',dtype=np.float32, initial=0.0)
    reseed_depth = Variable('reseed_depth',dtype=np.float32, initial=0.0)
    reseed_flag = Variable('reseed_flag',dtype=np.int32, initial=0)
    
    prev_lat = Variable('prev_lat',dtype=np.float32, initial=0.0)
    prev_lon = Variable('prev_lon',dtype=np.float32, initial=0.0)
    prev_depth = Variable('prev_depth',dtype=np.float32, initial=0.0)
    
    flag_position = Variable('flag_position',dtype=np.int32, initial=0)
    extreme = Variable('extreme',dtype=np.float32, initial=0.0)
    extreme_arag = Variable('extreme_arag',dtype=np.float32, initial=0.0)
    
    
def ReturnToSurface(particle, fieldset, time):
    """This function handels "through surface errors" by pushing particles back to below the surface
    
    Keyword arguments:
    particle -- Ocean Parcels particle
    fieldset -- Ocean Parcels fieldset
    time -- Simulation time

    """ 

    #read position in mask
    flag_rho = fieldset.mask[0,0,particle.lat,particle.lon]
    if flag_rho < 1:
        # push back to nearest water pixel
        print('Particle beached ', flag_rho)
        unU = fieldset.unBeach_lon[0,0,particle.lat,particle.lon]
        unV = fieldset.unBeach_lat[0,0,particle.lat,particle.lon]
        
        particle.lat = particle.lat + unV
        particle.lon = particle.lon + unU
    else:
        print('Too high')
        print(particle.MyID)
        
        print(particle.depth)
        
        print(fieldset.top_depth[0, 0, particle.lat, particle.lon])
        particle.depth = fieldset.top_depth[0, 0, particle.lat, particle.lon] + 0.1
        print(particle.depth)


def PushToWater(particle, fieldset, time):
    """This function handels "through boundary errors" by pushing particles back to the ocean if beached
    
    Keyword arguments:
    particle -- Ocean Parcels particle
    fieldset -- Ocean Parcels fieldset
    time -- Simulation time

    """ 
    
    #read position in mask
    flag_rho = fieldset.mask[0,0,particle.lat,particle.lon]
    if flag_rho < 1:
        # push back to nearest water pixel
        print('Particle beached ', flag_rho)
        unU = fieldset.unBeach_lon[0,0,particle.lat,particle.lon]
        unV = fieldset.unBeach_lat[0,0,particle.lat,particle.lon]
        
        particle.lat = particle.lat + unV
        particle.lon = particle.lon + unU
    else:
        #particle is too deep
        print('Particle is too deep')
        if particle.reseed_flag == 1:
            #reseeding took place and was not succesfull (for some strange unknown reason...)
            #revert back
            print('Reseeding problem...')
            print(particle.depth)
            print(particle.prev_lat)
            particle.lat = particle.prev_lat
            print(particle.prev_lon)
            particle.lon = particle.prev_lon
            print(particle.prev_depth)
            particle.depth = particle.prev_depth
            print(particle.depth)
        
        else:
            #no reseeding took place just some other error
            print('Other problem...')
            print(fieldset.bottom_depth[0, 0, particle.lat, particle.lon])
            print(fieldset.top_depth[0, 0, particle.lat, particle.lon])
            particle.depth = fieldset.bottom_depth[0, 0, particle.lat, particle.lon] - 0.1
            
            
    
    
def pteropod_kernel(particle, fieldset, time): 
    """This function defines the movement and interaction of pteropds with the environment
    (In the future, this function should be separated into advection, DVM, interaction)
    
    Keyword arguments:
    particle -- Ocean Parcels particle
    fieldset -- Ocean Parcels fieldset
    time -- Simulation time

    """ 
    # ================================================================================================
    # Advection of particles using fourth-order Runge-Kutta integration including vertical velocity.
    # Function needs to be converted to Kernel object before execution
    # ================================================================================================
    particle.flag_position = 0
    particle.prev_lat = particle.lat
    particle.prev_lon = particle.lon
    particle.prev_depth = particle.depth
    
    #Ensure particle is on a water pixel
    #read position in mask
    flag_rho = fieldset.mask[0,0,particle.lat,particle.lon]
    if flag_rho < 1:
        # push back to nearest water pixel
        unU = fieldset.unBeach_lon[0,0,particle.lat,particle.lon]
        unV = fieldset.unBeach_lat[0,0,particle.lat,particle.lon]
        
        particle.lat = particle.lat + unV
        particle.lon = particle.lon + unU
    
 
    particle.flag_position += 1 
    
    if particle.depth  > fieldset.bottom_depth[0, 0, particle.lat, particle.lon]:
        particle.depth  = fieldset.bottom_depth[0, 0, particle.lat, particle.lon] - 0.1
    elif particle.depth  < fieldset.top_depth[0, 0, particle.lat, particle.lon]:
        particle.depth = fieldset.top_depth[0, 0, particle.lat, particle.lon] + 0.1
    particle.flag_position += 1    
    (u1, v1, w1) = fieldset.UVW[particle.time, particle.depth, particle.lat, particle.lon]
    particle.flag_position += 1
    lon1 = particle.lon + u1*.5*particle.dt
    lat1 = particle.lat + v1*.5*particle.dt
    if particle.depth + w1 * .5 * particle.dt > fieldset.bottom_depth[0, 0, lat1, lon1]:
        dep1 = fieldset.bottom_depth[0, 0, lat1,lon1] - 0.1
    elif particle.depth + w1 * .5 * particle.dt < fieldset.top_depth[0, 0, lat1, lon1]:
        dep1 = fieldset.top_depth[0, 0, lat1, lon1] + 0.1    
    else:
        dep1 = particle.depth + w1 * .5 * particle.dt
    particle.flag_position += 1
    (u2, v2, w2) = fieldset.UVW[particle.time + .5 * particle.dt, dep1, lat1, lon1]
    particle.flag_position += 1
    lon2 = particle.lon + u2*.5*particle.dt
    lat2 = particle.lat + v2*.5*particle.dt
    particle.flag_position += 1
    
    if particle.depth + w2 * .5 * particle.dt > fieldset.bottom_depth[0, 0, lat2, lon2]:
        dep2 = fieldset.bottom_depth[0, 0, lat2, lon2] - 0.1
    elif particle.depth + w2 * .5 * particle.dt < fieldset.top_depth[0, 0,lat2, lon2]:
        dep2 = fieldset.top_depth[0, 0, lat2, lon2] + 0.1
    else:
        dep2 = particle.depth + w2 * .5 * particle.dt
    (u3, v3, w3) = fieldset.UVW[particle.time + .5 * particle.dt, dep2, lat2, lon2]
    particle.flag_position += 1
    
    lon3 = particle.lon + u3*particle.dt
    lat3 = particle.lat + v3*particle.dt
    
    if particle.depth + w3 * particle.dt > fieldset.bottom_depth[0,  0, lat3, lon3]:
        dep3 = fieldset.bottom_depth[0,  0, lat3, lon3] - 0.1
    elif particle.depth + w3 * particle.dt < fieldset.top_depth[0,  0, lat3, lon3]:
        dep3 = fieldset.top_depth[0,  0, lat3, lon3] + 0.1    
    else:
        dep3 = particle.depth + w3 * particle.dt
    (u4, v4, w4) = fieldset.UVW[particle.time + particle.dt, dep3, lat3, lon3]
    particle.flag_position += 1
    
    particle.lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * particle.dt
    particle.lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * particle.dt
    particle.flag_position += 1
    if particle.depth + (w1 + 2*w2 + 2*w3 + w4) / 6. * particle.dt > fieldset.bottom_depth[0, 0, particle.lat, particle.lon]:
        particle.depth = fieldset.bottom_depth[0, 0, particle.lat, particle.lon] - 0.1
    elif particle.depth + (w1 + 2*w2 + 2*w3 + w4) / 6. * particle.dt < fieldset.top_depth[0, 0, particle.lat, particle.lon]:
        particle.depth = fieldset.top_depth[0, 0, particle.lat, particle.lon] + 0.1    
    else:
        particle.depth += (w1 + 2*w2 + 2*w3 + w4) / 6. * particle.dt
    particle.flag_position += 1
    day = 0
    if particle.shell_size < 1.31:
        day = 1
    elif particle.shell_size >= 1.31: #Three weeks for the development of parapodia
        max_depth = fieldset.bottom_depth[0, 0, particle.lat, particle.lon] - 0.1
        min_depth = fieldset.top_depth[0, 0, particle.lat, particle.lon] + 0.1
        if max_depth < particle.max_depth:
            particle.max_depth = max_depth
        particle.min_depth = min_depth   

        #use particle time to determine if they are going up or down based on particle.departure
        if particle.time <= particle.departure and particle.time >= particle.departure_from_depth:
            day = 1
        #priority is for the particle to get to the surface
            particle.flag_down = 0
            #test current chl level
            current_chl = fieldset.Chl[particle.time, particle.depth, particle.lat, particle.lon]
            if current_chl < 0:
                current_chl = 0
            #compare to previous chl level
            if current_chl < particle.chl_ascent:
                #stop ascending
                particle.flag_up = 1
            else:
                particle.chl_ascent = current_chl
                particle.flag_up = 0
                
            if particle.flag_up == 0:
                if particle.depth > particle.min_depth + particle.up*particle.dt:    
                    particle.depth -= particle.up*particle.dt
                #if too fast, then it goes straight to the top    
                else:
                    particle.depth = particle.min_depth
                    particle.flag_up = 1
            #then if already at the surface, then the particle drifts around        
            elif particle.flag_up == 1:
                if particle.depth < particle.min_depth:
                    particle.depth = particle.min_depth
                    particle.flag_up = 1
                if particle.depth > particle.min_depth:
                    particle.flag_up = 0
            
        #going down:
        elif particle.time < particle.departure_from_depth or particle.time > particle.departure:
        #priority is to stay around the desired max_depth
            day = 0
            particle.flag_up = 0
            particle.chl_ascent = -9999
            
            if particle.flag_down == 0:
                #if not fast enough then stepwise 
                particle.O2_level = fieldset.O2[particle.time, particle.depth, particle.lat, particle.lon]
                particle.productivity = fieldset.Chl[particle.time, particle.depth, particle.lat, particle.lon]
                if particle.depth  < particle.max_depth - particle.down*particle.dt and particle.productivity > 0.0 and particle.O2_level > 60:
                    particle.depth += particle.down*particle.dt
                #if too fast, then it goes straight to the bottom     
                elif particle.depth >=  particle.max_depth - particle.down*particle.dt and particle.productivity > 0.0 and particle.O2_level > 60:
                    particle.depth = particle.max_depth
                    particle.flag_down = 1
                    
                
                particle.productivity = fieldset.Chl[particle.time, particle.depth, particle.lat, particle.lon]
                particle.O2_level = fieldset.O2[particle.time, particle.depth, particle.lat, particle.lon]
                
                if particle.productivity <= 0 or particle.O2_level <= 60:
                    particle.flag_down = 1

            #then if already at the desired depth, then the particle drifts around   
            elif particle.flag_down == 1:
                
                particle.productivity = fieldset.Chl[particle.time, particle.depth, particle.lat, particle.lon]
                particle.O2_level = fieldset.O2[particle.time, particle.depth, particle.lat, particle.lon]
                #swim up again to avoid deoxygenation
                if particle.O2_level <= 60:
                    if particle.depth > particle.min_depth + particle.up*particle.dt:    
                        particle.depth -= particle.up*particle.dt
                #if too fast, then it goes straight to the top    
                    else:
                        particle.depth = particle.min_depth
 
                #check if avoidance was successful
                    particle.O2_level = fieldset.O2[particle.time, particle.depth, particle.lat, particle.lon]
                    
                if particle.O2_level > 60:
                    particle.flag_down = 0
                    
                    
                if particle.depth > particle.max_depth:
                    particle.depth = particle.max_depth
                    particle.flag_down = 0
    
    
    #Sample temperature
    particle.flag_position += 1
    particle.temp = fieldset.temp[particle.time, particle.depth, particle.lat, particle.lon]
    particle.temp_sum = particle.temp_sum + particle.temp
    particle.flag_position += 1
    #Sample food
    particle.food = fieldset.Chl[particle.time, particle.depth, particle.lat, particle.lon]
    if particle.food  < 0 or day == 0: 
        particle.food = 0
    particle.food_sum = particle.food_sum + particle.food
    particle.flag_position += 1
    
    #Sample aragonite
    particle.extreme = fieldset.extremes[particle.time, particle.depth, particle.lat, particle.lon]
    if particle.extreme > 0:
        particle.extreme_arag = fieldset.arag[particle.time, particle.depth, particle.lat, particle.lon] + particle.extreme_arag
    
    particle.arag_exposure = fieldset.arag[particle.time, particle.depth, particle.lat, particle.lon]
    particle.arag_exposure_sum = particle.arag_exposure_sum + particle.arag_exposure
    particle.flag_position += 1
    #sample oxygen 
    particle.oxygen = fieldset.O2[particle.time, particle.depth, particle.lat, particle.lon]
    particle.oxygen_sum = particle.oxygen_sum + particle.oxygen
    particle.flag_position += 1
    particle.step_counter = particle.step_counter + 1
    
    #sample indeces
    particle.myxi = particle.xi
    particle.myyi = particle.yi
    particle.myzi = particle.zi
    
    particle.flag_position += 1
    current_chl = fieldset.Chl[particle.time, particle.depth, particle.lat, particle.lon]
    if current_chl >= particle.chl_max:
        particle.chl_max = current_chl
        particle.depth_chl_max = particle.depth
        particle.lon_chl_max = particle.lon
        particle.lat_chl_max = particle.lat     
    particle.flag_position += 1