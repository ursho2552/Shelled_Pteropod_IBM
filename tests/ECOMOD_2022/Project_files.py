#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 09:42:42 2022
Scripts used for the Shelled pteropod IBM project ECOMOD
@author: ursho
"""

import numpy as np
import xarray as xr
from scipy import interpolate
import os

def get_initial_positions(num=1500,grid_file="/nfs/kryo/work/fdesmet/roms/input/pactcs30/pactcs30_grd.nc",outfile='output_initialization/Initial_positions.csv'):
    """This function determines locations in the domain that are away from the shore, and within a given
    range of lat and lon. The grid file needs a field with the lat and lon values at rho, as well as the land/ocean mask.
    The list of values is saved in a csv file
    
    Parameters:
    num (int): Number of points that need to be found in the region
    grid_file (string): path to netCDF file with the grid mask of the region
    outfile (string): name of file to save list
    
    
    Returns:
    list(int): List of possible locations as lon/lat/depth(=5m)
    

    UHE 01/10/2020
    """    
    #get the data for lon and lat in the region
    directory_ROMS_regions = "/nfs/kryo/work/fdesmet/roms/output/pactcs30/hc003_daily_pactcs30/avg/"
    DS_ebus = xr.open_dataset(directory_ROMS_regions+"Subdomains_pactcs30_bnds.nc")
    #get the data for lon and lat in the region
    DS_mask = xr.open_dataset(grid_file)
    
    lat_list = np.reshape(DS_mask.lat_rho.values,(-1,1))
    lon_list = np.reshape(DS_mask.lon_rho.values,(-1,1))
    mask_vals = DS_mask.mask_rho.values.copy()

    mask_vals_orig = DS_mask.mask_rho.values.copy()
    mask_vals_orig_u = DS_mask.mask_u.values.copy()
    mask_vals_orig_v = DS_mask.mask_v.values.copy()
    
    mask_vals_orig[:,:-1] = mask_vals_orig[:,:-1] + mask_vals_orig_u
    mask_vals_orig[:,1:] = mask_vals_orig[:,1:] + mask_vals_orig_u
    mask_vals_orig[:-1,:] = mask_vals_orig[:-1,:] + mask_vals_orig_v
    mask_vals_orig[1:,:] = mask_vals_orig[1:,:] + mask_vals_orig_v
    
    ebus_mask = np.reshape(DS_ebus.ROMS_regions_mask.values[:,:,-1],(-1,1))
    
    
    for i in range(4):
        mask_vals[:,1:] = mask_vals[:,1:]  + mask_vals[:,:-1]
        mask_vals[:,:-1] = mask_vals[:,:-1] + mask_vals[:,1:]
        mask_vals[:-1,:] = mask_vals[:-1,:] + mask_vals[1:,:] 
        mask_vals[1:,:] = mask_vals[1:,:] + mask_vals[:-1,:]
        
        mask_vals[:,1:] = mask_vals[:,1:]  + mask_vals[:,:-1]
        mask_vals[:,:-1] = mask_vals[:,:-1] + mask_vals[:,1:]
        mask_vals[:-1,:] = mask_vals[:-1,:] + mask_vals[1:,:] 
        mask_vals[1:,:] = mask_vals[1:,:] + mask_vals[:-1,:]
    

    mask_new = np.reshape(mask_vals,(-1,1))
    mask_vals_orig = np.reshape(mask_vals_orig,(-1,1))

    
    print('Searching for positions...')

    ind_tmp = np.argwhere((ebus_mask == 1)&(mask_new==max(mask_new))&(mask_vals_orig==5))
      
    choice = np.random.choice(ind_tmp[:,0],num,replace=True)
    
    list_latlon = np.empty((num,3))
    list_latlon[:,0] = lon_list[choice,0]
    list_latlon[:,1] = lat_list[choice,0]
    list_latlon[:,2] =  5.0
    
    np.savetxt(outfile, list_latlon, delimiter=',')

    return list_latlon


def get_daily_maredat_obs(min_lat=120,max_lat=151,ref_data="/cluster/home/ursho/kryo_work/ursho/PhD/Projects/Pteropod_IBM/Data/MarEDat20120203Pteropods.nc"):
    """This function interpolates monthly MAREDAT abundances into daily observations for a comparison with modeled abundances.
    The reference data needs to be in monthly means (12 entries). Data has to be in daily data. 
    The similarity is calculated using the Dynamic Time Warping (DTW) approach.
    
    Parameters:
    data (array): Array containing the daily output
    start (int): start index from which the time series of data is taken
    end (int): maximum index of the time series of data
    ref_data (string): Path to netcdf file with monthly means
    
    Returns:
    min_start (int): index at which the pattern matchup is best between data and reference data
    
    UHE 01/10/2020
    """ 
    
    assert type(min_lat) == int
    assert type(max_lat) == int

    DS_ptero = xr.open_dataset(ref_data,decode_times=False)
    
    abundance = np.empty([14])
    abundance_std = np.empty([14])
    for i in range(12):
        abundance[i+1] = np.nansum(DS_ptero.ABUNDANCE[i,:,min_lat:max_lat,:].values) 
        abundance_std[i+1] = np.nansum(DS_ptero.STDEV_ABUND[i,0:11,min_lat:max_lat,:].values)
 
    abundance[0] = abundance[12]
    abundance[13] = abundance[1]
    
    abundance_std[0] = abundance_std[12]
    abundance_std[13] = abundance_std[1]
    
    x = np.array([-15.5,15.5,45,74.5,105,135.5,166,196.5,227.5,258,288.5,319,349.5,380.5])
    abundance_interp = interpolate.interp1d(x, abundance)
    xnew = np.arange(0, 365,1)
    daily_abundance = abundance_interp(xnew)
    
    abundance_std_interp = interpolate.interp1d(x, abundance_std)
    daily_abundance_std = abundance_std_interp(xnew)
    
    return daily_abundance,daily_abundance_std

def Get_rates_from_file(directory_mort,similarity_file,threshold):
    """This function finds a set of mortality rates with a ratio of spring to fall abundance
    that approaches the value threshold.
    
    Parameters:
    rate_file (string): Path to file containing the possible sets of mortality rates
    ratio_file (string): Path to file conaining the mean ratio between spring and fall abundances for all possible sets of mortality rates
    threshold (float): Optimal ratio between spring and fall abundances

    
    Returns:
    np.squeeze(possible_rates[b,:]) (array): Array containing the set of mortality rates that leads to a simulation where the ratio best matches the threshold
    
    UHE 01/10/2020
    """ 

    first_flag = 0
    for i in range(threshold):
        if os.path.exists(directory_mort+similarity_file.format(i)):
            if first_flag == 0:
                first_i = i
                first_flag = 1
                
            possible_sim = np.genfromtxt(directory_mort+similarity_file.format(i),delimiter=',')
            
            if possible_sim.shape[0] > 0:
            
                if i == first_i:
                    similarity_mat = possible_sim
                else:
                    similarity_mat = np.vstack((similarity_mat,possible_sim))
    
    optimum = 1- similarity_mat[:,11]/365 + similarity_mat[:,9] + similarity_mat[:,10] - similarity_mat[:,12]
    
    b = np.argwhere(optimum == max(optimum))[0]
    
    return np.squeeze(similarity_mat[b[0],0:8])
