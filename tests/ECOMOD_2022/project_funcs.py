#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 09:42:42 2022
Scripts used for the Shelled pteropod IBM project ECOMOD
@author: ursho
"""

import glob
import logging
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


    for _ in range(4):
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

    logging.info('Searching for positions...')

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

    assert isinstance(min_lat,int)
    assert isinstance(max_lat,int)

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

    optimum = 1 - \
        similarity_mat[:,11]/365 + similarity_mat[:,9] + similarity_mat[:,10] - similarity_mat[:,12]

    b = np.argwhere(optimum == max(optimum))[0]

    return np.squeeze(similarity_mat[b[0],0:8])


def calculate_timings(L_t_spring,day_start,daily_sst,daily_npp,T0=14.5,Ks=3,dt=20):

    rates_all = [(L_t_spring[i] - L_t_spring[i-1])/L_t_spring[i-1] for i in range(1,len(L_t_spring))]
    rates_all.insert(0,rates_all[0])
    rates_all.append(rates_all[-1])

    new_rate = list()
    length_q2 = list()
    
    T0 = T0
    Ks = Ks
    day_start = day_start
    length_q2.append(0.15)
    
    spring_t = np.arange(day_start, day_start + 329)%365
    spring_T = daily_sst[spring_t]
    spring_F = daily_npp[spring_t]
    rate_ref = rates_all[0]
    
    for i in np.arange(0,len(spring_t)):
        
        Ks_s = Ks 
        tmp = rate_ref*1.3**((spring_T[i] - T0)/10)*spring_F[i]/(Ks_s + spring_F[i])
        new_rate.append(tmp)
        length_q2.append(length_q2[-1]*(1+new_rate[-1]))
        dist = abs(L_t_spring-length_q2[-1])
        pos_idx = np.argwhere(dist == min(dist))[0][0]
        rate_ref = rates_all[pos_idx]

    maturity_time = np.argwhere(length_q2 >= L_t_spring[90])[0]
    end_mat = maturity_time+dt
    
    spring_L = length_q2
    new_rate = list()
    length_q2 = list()
    
    length_q2.append(0.15)
    
    winter_t = np.arange(day_start + end_mat, day_start + end_mat + 500)%365
    winter_T = daily_sst[winter_t]
    winter_F = daily_npp[winter_t]

    rate_ref = rates_all[0]
    
    for i in np.arange(0,len(winter_t)):
        
        Ks_s = Ks 
        new_rate.append(rate_ref*1.3**((winter_T[i] - T0)/10)*winter_F[i]/(Ks_s + winter_F[i]))
        length_q2.append(length_q2[-1]*(1 + new_rate[-1]))
        dist = abs(L_t_spring - length_q2[-1])
        pos_idx = np.argwhere(dist == min(dist))[0][0]
        rate_ref = rates_all[pos_idx]

    maturity_time_w = np.argwhere(length_q2 >=L_t_spring[90])[0]
    winter_L = length_q2
    
    return[maturity_time+maturity_time_w,maturity_time,maturity_time_w,spring_L,winter_L]