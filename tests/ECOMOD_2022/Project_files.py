#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 09:42:42 2022
Scripts used for the Shelled pteropod IBM project ECOMOD
@author: ursho
"""

import glob
import numpy as np
import xarray as xr
from scipy import interpolate

def get_initial_positions(num=1500,
                          grid_file="/nfs/kryo/work/fdesmet/roms/input/pactcs30/pactcs30_grd.nc",
                          outfile='output_initialization/Initial_positions.csv'):
    """This function determines locations in the domain that are away from the 
    shore, and within a given range of lat and lon. The grid file needs a field
    with the lat and lon values at rho, as well as the land/ocean mask.
    The list of values is saved in a csv file. Returns a list of possible 
    locations as lon|lat|depth (=5 m)
    
    Keyword arguments:
    num -- number of points that need to be found in the region
    grid_file - path to netCDF file with the grid mask of the region
    outfile -- name of file to save list
    """    
    #get the data for lon and lat in the region
    directory_ROMS_regions = "/nfs/kryo/work/fdesmet/roms/output/pactcs30/hc003_daily_pactcs30/avg/"
    DS_ebus = xr.open_dataset(directory_ROMS_regions + "Subdomains_pactcs30_bnds.nc")
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

    ind_tmp = np.argwhere((ebus_mask == 1) & (mask_new==max(mask_new)) & (mask_vals_orig==5))
      
    choice = np.random.choice(ind_tmp[:,0],num,replace=True)
    
    list_latlon = np.empty((num,3))
    list_latlon[:,0] = lon_list[choice,0]
    list_latlon[:,1] = lat_list[choice,0]
    list_latlon[:,2] =  5.0
    
    np.savetxt(outfile, list_latlon, delimiter=',')

    return list_latlon


def get_daily_maredat_obs(
        min_lat=120, max_lat=151, 
        ref_data="/cluster/home/ursho/kryo_work/ursho/PhD/Projects/Pteropod_IBM/Data/MarEDat20120203Pteropods.nc"):
    """This function interpolates monthly MAREDAT abundances into daily 
    observations for a comparison with modeled abundances. The reference data 
    needs to be in monthly means (12 entries) by depth, by latitude, by 
    longitude. Function returns the interpolated daily observations and 
    standard deviation for the given latitude range.
    
    Keyword arguments:
    min_lat -- minimum latitude to consider
    max_lat -- maximum latitude to consider
    ref_data -- path to netcdf file with monthly means
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

def get_mortality_rates_from_file(directory_mort,similarity_file):
    """This function finds the set of daily mortalities that lead to the best
    fit between model output and observed abundance.
    
    Keyword arguments:
    directory_mort -- path to file containing the possible sets of mortality rates
    similarity_file -- file name containing the sets of mortality rates and 
        goodness of fit metrics
    """ 
    
    for i, file in enumerate(glob.glob(directory_mort+similarity_file)):
        
        possible_sim = np.genfromtxt(file,delimiter=',')
        similarity_mat = possible_sim if i==0 else np.vstack((similarity_mat,possible_sim))
    
    optimum = 1- similarity_mat[:,11]/365 + similarity_mat[:,9] + similarity_mat[:,10] - similarity_mat[:,12]
    
    b = np.argwhere(optimum == max(optimum))[0]
    
    return np.squeeze(similarity_mat[b[0],0:8])
