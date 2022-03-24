#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 14:48:46 2022
Functions used to calculate the mortality, growth, development, and spawning of the modeled pteropods
@author: ursho
"""

import numpy as np
import scipy
import os

def calculate_growth_fct():
    """Calculate the shell size of pteropods as a function of their age.
    The formula was taken from Wang et al., 2017. Returns the growth rate in (mm) as a function of time after birth

    Keyword arguments:
    None
    """

    WP = 0.1
    K = 5.07
    L_inf = 4.53
    C = 0.4
    ts = WP-0.5
    t0 = 120/365

    t = np.arange(121,121+300)/365

    S_t = (C*K)/(2*np.pi) * np.sin(2*np.pi*(t-ts))
    S_t0 = (C*K)/(2*np.pi) * np.sin(2*np.pi*(t0-ts))
    L_t_spring = L_inf*(1-np.exp(-(K*(t-t0) + S_t - S_t0)))
    L_t_spring += 0.15-L_t_spring[0]

    return L_t_spring



def get_number_individuals(indeces,stage,generation,rate_g0_0,rate_g0_1,rate_g0_2,rate_g0_3,rate_g1_0,rate_g1_1,rate_g1_2,rate_g1_3):
    """This function uses beta survival (b) rates to calculate the fraction of individual that survive after one time step. Returns the number of individuals
    that would die on position 0, and the mortality rate (1-rate) on position 1. UHE 25/09/2020
    The function used is taken from Bednarsek et al., 2016:
        N_{t+1} = N_{t}*exp(-b*t),
        rate = exp(-b*t) with t = 1 day.

    Keyword arguments:
    indeces -- list of indeces of a specific stage in the population
    stage -- integer identifier for the life stage
    generation -- integer identifier for the generation, 0 for spring, 1 for winter
    rate_gX_Y -- beta survival rates for each stage and generation
    """
    if generation == 0:
        if stage == 0:
            rate = np.exp(-rate_g0_0)
        elif stage == 1:
            rate = np.exp(-rate_g0_1)
        elif stage == 2:
            rate = np.exp(-rate_g0_2)
        elif stage == 3:
            rate = np.exp(-rate_g0_3)

    elif generation == 1:
        if stage == 0:
            rate = np.exp(-rate_g1_0)
        if stage == 1:
            rate = np.exp(-rate_g1_1)
        elif stage == 2:
            rate = np.exp(-rate_g1_2)
        elif stage == 3:
            rate = np.exp(-rate_g1_3)

    return [int(np.around(indeces.size*(1-rate))),1-rate]


def mortality(pteropod_list,rate_g0_0=0.211,rate_g0_1=0.09,rate_g0_2=0.09,rate_g0_3=0.09,
            rate_g1_0=0.142,rate_g1_1=0.01,rate_g1_2=0.01,rate_g1_3=0.01,day=None,outfile='/cluster/scratch/ursho/output_simulation_extremes/mortalities/'):
    """Select subsets of the individuals found in pteropod list according to their stage and generation.
    These are then used in 'get_number_individuals' function to calculate the mortality rates. Returns list of indeces of individuals that die, and the
    pteropod array with updated values for survival. UHE 25/09/2020

    Keyword arguments:
    pteorpod_list -- array containing all state variables characterizing the pteropods
    rate_gX_Y -- beta mortality rates for each stage and generation
    day -- list containing year, version, day and control for saving causes for mortalitites (background, old age, spawning,...; default None)
    outfile -- output directory

    """
    num_dead_nat = 0
    num_dead_dis = 0
    num_dead_old = 0

    all_rands = np.array([])
    all_delta_rates = np.array([])
    all_rates = np.array([])
    all_tmps = np.array([])
    for gen in range(2):
        for i in range(4):
            tmp = np.squeeze(np.argwhere((pteropod_list[:,2] == i) & (pteropod_list[:,1]%2 == gen)))
            if tmp.size == 1:
                tmp = np.array([tmp])

            if tmp.size > 0:
                num_ind,rate = get_number_individuals(tmp,i,gen,rate_g0_0,rate_g0_1,rate_g0_2,rate_g0_3,
                                                 rate_g1_0,rate_g1_1,rate_g1_2,rate_g1_3)

                if num_ind > 0:
#                    ind = np.random.choice(tmp,size=int(num_ind),replace=False)
                    rands = np.random.random(tmp.size)
                    #change rate according to their aragonite exposure (Lischka et al., 2011)
                    coeff = -6.34
                    arag0 = 1.5

                    #list of aragonite
                    aragt = pteropod_list[tmp,13]
                    #no change if above threshold
                    aragt[aragt > arag0] = arag0

                    delta_rate = coeff * (aragt-arag0)/100

                    #check mortality due to dissolution
                    pteropod_list[tmp[np.squeeze(rands < rate)],5] = 0
                    #count number of dead pteorpods (natural death)
                    num_dead_nat = num_dead_nat + np.sum(rands < rate)

                    #UHE 25/09/2020: Every single individual is treated the same
                    pteropod_list[tmp[np.squeeze(rands < rate+delta_rate)],5] = 0
                    #count increase in mortality (dissolition death)
                    num_dead_dis = num_dead_dis + np.sum((rands >= rate)&(rands < rate+delta_rate))

                    all_rands = np.hstack((all_rands,np.squeeze(rands)))
                    all_delta_rates = np.hstack((all_delta_rates,np.squeeze(delta_rate)))
                    all_rates = np.hstack((all_rates,np.squeeze(delta_rate*0+rate)))
                    all_tmps = np.hstack((all_tmps,np.squeeze(tmp)))

    #remove pteropods that are too old, that already spawned, or that are beached (given as 0 food)
    pteropod_list[np.squeeze(np.argwhere(pteropod_list[:,4] > 300)),5] = 0
    pteropod_list[np.squeeze(np.argwhere(pteropod_list[:,6] >= 1)),5] = 0


    #count mortality due to age
    num_dead_old = np.sum(pteropod_list[:,5]==0) - num_dead_dis - num_dead_nat

    pteropod_list[np.squeeze(np.argwhere(pteropod_list[:,16] <= 0)),5] = 0
    num_dead_beaching = np.sum(pteropod_list[:,5]==0) - num_dead_dis - num_dead_nat - num_dead_old
#    print('Death beaching: {}'.format(num_dead_beaching))
    num_dead_old = np.sum(pteropod_list[:,5]==0) - num_dead_dis - num_dead_nat

    #get overall list of dead particles, needed for parcels objects
    dead_particles = np.squeeze(np.argwhere(pteropod_list[:,5] == 0)).astype(int)

    num_ptero = np.squeeze(np.argwhere(pteropod_list[:,5] == 1)).astype(int)
    if num_ptero.size == 1:
        num_ptero = np.array(num_ptero)
        new_pteropod_list = np.reshape(pteropod_list[num_ptero,:].copy(),(1,17))
    else:
        new_pteropod_list = pteropod_list[num_ptero,:].copy()
    #ensure that the shape is correct

    #==========================================================
    # Save rands, rate, dealta_rate, index, UHE 02/06/2021
    #==========================================================
    if day is not None:
        assert len(day) == 4,"The argument 'day' should have the lenght 4 with the structure (year,version,day,control)"
        #save variables as csv file
        outfile_mort = outfile+'year_{}_V_{}_control_{}/'.format(day[0],day[1],day[3])
        if not os.path.exists(outfile_mort):
            os.makedirs(outfile_mort)

        array_save = np.vstack((all_tmps,all_rands,all_delta_rates,all_delta_rates*0+all_rates))
        np.savetxt(outfile_mort+'Mortality_Day_{}.csv'.format(int(day[2])), array_save, delimiter=',')

        outfile_mort = outfile+'year_{}_V_{}_control_{}/'.format(day[0],day[1],day[3])
        if not os.path.exists(outfile_mort):
                os.makedirs(outfile_mort)
        array_save = np.array([num_dead_nat,num_dead_dis,num_dead_old])
#        print(array_save.shape)
#        print(array_save)
        np.savetxt(outfile_mort+'Mortality_type_Day_{}.csv'.format(int(day[2])), array_save, delimiter=',')

    return dead_particles,new_pteropod_list


def mortality_DW(pteropod_list,rate_g0_0=0.211,rate_g0_1=0.09,rate_g0_2=0.09,rate_g0_3=0.09,
            rate_g1_0=0.142,rate_g1_1=0.01,rate_g1_2=0.01,rate_g1_3=0.01,longevity=390,base1=5.26,
                 exp1=-0.25,base2=5.26,exp2=-0.25,day=None,outfile='/cluster/scratch/ursho/output_simulation_extremes/mortalities/'):
    """Select subsets of the individuals found in pteropod list according to their stage and generation.
    These are then used in 'get_number_individuals' function to calculate the mortality rates. Returns list of indeces of individuals that die, and the
    pteropod array with updated values for survival. UHE 25/09/2020

    Keyword arguments:
    pteorpod_list -- array containing all state variables characterizing the pteropods
    rate_gX_Y -- beta mortality rates for each stage and generation
    longevity -- maximum longevity
    baseX -- base of equation used to calculate the dry weight specific mortality for the X generation
    expX -- exponent of equation used to calculate the dry weight specific mortality for the X generation
    day -- list containing year, version, day and control for saving causes for mortalitites (background, old age, spawning,...; default None)
    outfile -- output directory

    """
    num_dead_nat = 0
    num_dead_dis = 0
    num_dead_old = 0

    all_rands = np.array([])
    all_delta_rates = np.array([])
    all_rates = np.array([])
    all_tmps = np.array([])
    for gen in range(2):
#         for i in range(4):
        tmp = np.squeeze(np.argwhere(pteropod_list[:,1]%2 == gen))
        if tmp.size == 1:
            tmp = np.array([tmp])

        if tmp.size > 0:

            num_ind,rate_eggs = get_number_individuals(tmp,0,gen,rate_g0_0,rate_g0_1,rate_g0_2,rate_g0_3,
                                                 rate_g1_0,rate_g1_1,rate_g1_2,rate_g1_3)

            flag_eggs = np.squeeze(np.argwhere(pteropod_list[tmp,2] == 0))

            DW = (0.137*pteropod_list[tmp,3]**1.5005)/1000
            if gen == 0:
                rate = base1/1000 * DW**(exp1)

            else:
                rate = base2/1000 * DW**(exp2)

            if sum(flag_eggs) > 0:
                rate[flag_eggs] = rate_eggs
            num_ind = 1

            if num_ind > 0:
#                    ind = np.random.choice(tmp,size=int(num_ind),replace=False)
                rands = np.random.random(tmp.size)

                #change rate according to their aragonite exposure (Lischka et al., 2011)
                coeff = -6.34
                arag0 = 1.5

                #list of aragonite
                aragt = pteropod_list[tmp,13]
                #no change if above threshold
                aragt[aragt > arag0] = arag0

                delta_rate = coeff * (aragt-arag0)/100

                #check mortality due to dissolution
                pteropod_list[tmp[np.squeeze(rands < rate)],5] = 0
                #count number of dead pteorpods (natural death)
                num_dead_nat = num_dead_nat + np.sum(rands < rate)

                pteropod_list[tmp[np.squeeze(rands < rate+delta_rate)],5] = 0
                #count increase in mortality (dissolition death)
                num_dead_dis = num_dead_dis + np.sum((rands >= rate)&(rands < rate+delta_rate))

                all_rands = np.hstack((all_rands,np.squeeze(rands)))
                all_delta_rates = np.hstack((all_delta_rates,np.squeeze(delta_rate)))
                all_rates = np.hstack((all_rates,np.squeeze(delta_rate*0+rate)))
                all_tmps = np.hstack((all_tmps,np.squeeze(tmp)))

    #remove pteropods that are too old, that already spawned, or that are beached (given as 0 food)
    pteropod_list[np.squeeze(np.argwhere(pteropod_list[:,4] > longevity)),5] = 0
    pteropod_list[np.squeeze(np.argwhere(pteropod_list[:,6] >= 1)),5] = 0


    #count mortality due to age
    num_dead_old = np.sum(pteropod_list[:,5]==0) - num_dead_dis - num_dead_nat

    pteropod_list[np.squeeze(np.argwhere(pteropod_list[:,16] <= 0)),5] = 0
    num_dead_beaching = np.sum(pteropod_list[:,5]==0) - num_dead_dis - num_dead_nat - num_dead_old
#    print('Death beaching: {}'.format(num_dead_beaching))
    num_dead_old = np.sum(pteropod_list[:,5]==0) - num_dead_dis - num_dead_nat

    #get overall list of dead particles, needed for parcels objects
    dead_particles = np.squeeze(np.argwhere(pteropod_list[:,5] == 0)).astype(int)

    num_ptero = np.squeeze(np.argwhere(pteropod_list[:,5] == 1)).astype(int)
    if num_ptero.size == 1:
        num_ptero = np.array(num_ptero)
        new_pteropod_list = np.reshape(pteropod_list[num_ptero,:].copy(),(1,17))
    else:
        new_pteropod_list = pteropod_list[num_ptero,:].copy()
    #ensure that the shape is correct

    #==========================================================
    # Save rands, rate, dealta_rate, index, UHE 02/06/2021
    #==========================================================
    if day is not None:
        #save variables as csv file
        outfile_mort = outfile+'year_{}_V_{}_control_{}/'.format(day[0],day[1],day[3])
        if not os.path.exists(outfile_mort):
            os.makedirs(outfile_mort)

        array_save = np.vstack((all_tmps,all_rands,all_delta_rates,all_delta_rates*0+all_rates))
        np.savetxt(outfile_mort+'Mortality_Day_{}.csv'.format(int(day[2])), array_save, delimiter=',')

        outfile_mort = outfile+'year_{}_V_{}_control_{}/'.format(day[0],day[1],day[3])
        if not os.path.exists(outfile_mort):
                os.makedirs(outfile_mort)
        array_save = np.array([num_dead_nat,num_dead_dis,num_dead_old])
#        print(array_save.shape)
#        print(array_save)
        np.savetxt(outfile_mort+'Mortality_type_Day_{}.csv'.format(int(day[2])), array_save, delimiter=',')

    return dead_particles,new_pteropod_list





def calculate_shell_carbonate(L):
    """This function calculates the calcium carbonate content in the shell of a pteropod
    The function is taken from Bednarsek et al., Deep Sea Research Part 2 59: 105-116 (2012)
    Returns the calcium carbonate content in mg CaCO3 on position 0 and the dry weight in mg DW on position 1. UHE 25/09/2020

    Keyword arguments:
    L -- Shell size in mm

    """
    DW = (0.137*L**1.5005)
    Shell_calc = DW*0.25*0.27*8.33

    return Shell_calc,DW

def calculate_dissolution_calcification(L,damage,delta_L,Arag,gain_flag=0):
    """This function calculates the loss and gain of CaCO3 given the current size, growth function, and exposure to aragonite saturation states.
    The function first determines the dissolution, and compares it to the calcium carbonate that could be produced under non-corrosive conditions
    to calculate the net gain/loss of CaCO3. The pteropod can then either repair the damage as much as possible or grow. Returns the new shell size in mm
    after dissolution/lack of accretion on position 0, and the accumulated damage in mg CaCO3 on position 1. UHE 13/01/2022


    Keyword arguments:
    L -- Shell size in mm
    damage -- Current accumulated damage in mg CaCO3
    delta_L -- Current increase in size under idealized conditions in mm
    Arag -- Aragonite saturation state experienced by pteropod
    gain_flag -- Identifier to determine if additional calcifiction should be considered

    """
    #ensure the input is an array even if scalars are given as input
    Ln = np.asarray([L]) if np.isscalar(L) or L.ndim == 0 else np.asarray(L)
    damagen = np.asarray([damage]).astype(np.float64) if np.isscalar(damage) or damage.ndim == 0 else np.asarray(damage).astype(np.float64)
    delta_Ln = np.asarray([delta_L]) if np.isscalar(delta_L) or delta_L.ndim == 0  else np.asarray(delta_L)
    Aragn = np.asarray([Arag]) if np.isscalar(Arag) or Arag.ndim == 0  else np.asarray(Arag)

    Shell_calc,DW = calculate_shell_carbonate(Ln)

    loss = 65.76 * np.exp(-4.7606*Aragn)*Shell_calc/100

    zero = np.zeros(L.size)
    L_new = Ln.copy()
    Shell_new = Ln.copy()*0.0

    #Calculate calcification at size L
    if gain_flag == 1:
        WW = DW/(0.28*1000) #in g
        Q = (0.57*np.log(Aragn)+0.25)/1000000  #in mol/(g ww h)
        Molar_mass = 100.0869 #in g/mol
        f_day = 24 #hours per day
        gain = Q*WW*Molar_mass*f_day*1000
        loss = np.amax([loss-gain,zero],axis=0)

    L_pot = Ln+delta_Ln
    Shell_calc_new,DW = calculate_shell_carbonate(L_pot)

    net = Shell_calc_new - Shell_calc - loss

    flag_smaller_damage = net > damagen
    if np.sum(flag_smaller_damage) > 0:
        Shell_new[flag_smaller_damage] = Shell_calc[flag_smaller_damage] + net[flag_smaller_damage] - damagen[flag_smaller_damage]
        L_new[flag_smaller_damage] = (Shell_new[flag_smaller_damage]/(0.25*0.27*8.33*0.137))**(1/float(1.5005))
    damagen = np.max([zero,damagen-net],axis=0)

    return L_new,damagen


def shell_growth(pteropod_list,growth_fct_gen0,Arag=4,T=16,F=7,T0=14.5,Ks=4.8,Tmax=31,Tmin=0.6,day=None,outfile='/cluster/scratch/ursho/output_simulation_extremes/Growth/'): #Ks=4.8
    """This function determines the net shell growth given the aragonite saturation state, current size, and generation. Returns array
    containing updated attributes characterizing the pteropods. UHE 25/09/2020

    Keyword arguments:
    pteorpod_list -- Array containing all state variables characterizing the pteropods
    growth_fct_gen0 -- Shell size as function of time for spring (X=0) and winter (X=1) generation
    Arag -- Aragonite saturation state experiences by each pteropod on one day
    T -- Temperature. Default value was set to 16 to simulate optimal conditions
    F -- Food/Phytoplankton carbon available. Default value 7 was chosen to simulate optimal conditions
    T0 -- Refernce temperature for the growth rate. Default value set to 14.5 according to Wang et al. 2017
    Ks -- Food/Phytoplankton carbon half-saturation constant. The default value is set to 2.6
    Tmax -- Maximum temperature for growth
    Tmin -- Minimum temperature for growth
    day -- list containing year, version, day and control for saving all growth rates for each day (default None)
    outfile -- output directory

    """
    #If dissolution is turned off, then create array with experienced aragonite saturation states
    #that are too high (4) to not have an effect on shell growth

    if np.ndim(Arag) == 0: #changed
        #only happens if there is no input
        Arag = pteropod_list[:,5].copy()*Arag

    list_days_of_growth = np.arange(pteropod_list.shape[0])
    #increase shell size according to temp and food, UHE 17/03/2021
    if list_days_of_growth.size > 0:

        #get the growth rate as fraction of size increase in each time step
        growth_rate = [(growth_fct_gen0[i]-growth_fct_gen0[i-1])/growth_fct_gen0[i-1] for i in range(1,len(growth_fct_gen0))]
        #repeat the first at the beginning and the last one at the end
        growth_rate.insert(0,growth_rate[0])
        growth_rate.append(growth_rate[-1])
        #convert to numpy array
        growth_rate = np.array(growth_rate)

        #current length
        L = pteropod_list[list_days_of_growth,3]
        #ensure the structure of L is correct if there is only one pteropod or multiple pteropods
        if L.shape[0] != 1:
            L = np.squeeze(L)
        #calculate distance to reference and find index with minimum distance
        pos_idx = np.array([np.squeeze(np.argwhere(abs(growth_fct_gen0-i) == abs(growth_fct_gen0-i).min())) for i in L])

        food_effect = F/(Ks+F)

        rate = growth_rate[pos_idx]*1.3**((T-T0)/10) * food_effect
        #Add thresholds of max and min temp
        rate[(T>Tmax)|(T<Tmin)] = 0
        delta_L = rate*L

        damage = np.squeeze(pteropod_list[list_days_of_growth,14])
        pteropod_list[list_days_of_growth,3],pteropod_list[list_days_of_growth,14]  = calculate_dissolution_calcification(L,damage,delta_L,Arag[list_days_of_growth])

        #==========================================================
        # Save rate, delta_L, T, F, damage, UHE 02/06/2021
        #==========================================================
        if day is not None:
            assert len(day) == 4, "The argument 'day' should contain year, version, day, and control"
            #save variables as csv file
            array_save = np.array([rate,delta_L,damage,T,F])
            outfile_growth = outfile+'year_{}_V_{}_control_{}/'.format(day[0],day[1],day[3])
            if not os.path.exists(outfile_growth):
                os.makedirs(outfile_growth)
            np.savetxt(outfile_growth+'Growth_Day_{}.csv'.format(int(day[2])), array_save, delimiter=',')

    return pteropod_list

def development(pteropod_list,growth_fct_gen0):
    """This function determines the life stage depending on the size of the pteropods.
    And increases the growth time by one day. Returns array containing updated attributes
    characterizing the pteropods. UHE 25/09/2020

    Keyword arguments:
    pteropod_list -- Array containing all state variables characterizing the pteropods
    growth_fct_gen0 -- Shell size as function of time

    """
    #adapt stages using thresholds
    pteropod_list[np.squeeze(np.where(pteropod_list[:,3] >= growth_fct_gen0[6])).astype(int),2] = 1
    pteropod_list[np.squeeze(np.where(pteropod_list[:,3] >= growth_fct_gen0[30])).astype(int),2] = 2
    pteropod_list[np.squeeze(np.where(pteropod_list[:,3] >= growth_fct_gen0[90])).astype(int),2] = 3

    #increase days of growth if they survive
    pteropod_list[:,4] = pteropod_list[:,4] + 1

    return pteropod_list

def spawning(pteropod_list, current_generation,next_ID,num_eggs=500,delta_ERR=20):
    """This function subsets the adult pteropods of a given generation, and determines which pteropods are ready to spawn eggs.
    Returns the array containing the updated attributes characterizing the pteropods, the next largest ID, and the current generation spawning. UHE 25/09/2020

    Keyword arguments:
    pteropod_list -- Array containing all state variables characterizing the pteropods
    current_generation -- Identifier of the current generation that will spawn the next generation
    next_ID -- The largest ID + 1 out of the entire population
    num_eggs -- Number of eggs spawned per adult of a single spawning event
    delta_ERR -- increase in the Egg Release Readiness (ERR) index per day as 1/delta_ERR

    """

    #add to the ERR if adults have repaired the damage
    pteropod_list[np.squeeze(np.argwhere(pteropod_list[:,2] == 3)).astype(int),7] += 1/(delta_ERR/2)

    available_generations = pteropod_list[np.argwhere(pteropod_list[:,2] == 3),1]
    if len(np.unique(available_generations)) >= 1  and  max(np.unique(available_generations)) !=  current_generation:
        current_generation = max(np.unique(available_generations))

    #get number of adults in the current generation that can produce eggs
    for idx in range(1):
    #get number of adults in the current generation that can produce eggs
        adults_ind = np.squeeze(np.argwhere((pteropod_list[:,2] == 3) &
                                    (pteropod_list[:,1]%2 == current_generation%2) &
                                    (pteropod_list[:,7] >= 1.0) &
                                    (pteropod_list[:,6] == idx))).astype(int)


        #get the total number of new eggs(particles)
        if adults_ind.size > 0:
            #for each entry in adult, create egg more entries
            #get the generation, Parent_ID, Shell_size, time_of_birth
            generation = np.squeeze(pteropod_list[adults_ind,1])
            Parent_ID = np.squeeze(pteropod_list[adults_ind,0])
            Parent_shell_size = np.squeeze(pteropod_list[adults_ind,3])
            time_birth = np.squeeze(pteropod_list[adults_ind,12])

            eggs = np.random.rand(adults_ind.size, 17)
            #ID
            eggs[:,0] = -1
            #generation
            eggs[:,1] = generation+1
            #stage
            eggs[:,2] = 0
            #shell_size
            eggs[:,3] = 0.15
            #days_of_growth
            eggs[:,4] = 0
            #survive
            eggs[:,5] = 1
            #num. spawning events
            eggs[:,6] = 0
            #ERR distribution around -1 and std 0.1
            eggs[:,7] = np.random.normal(-1,0.1,adults_ind.size)
            #spawned
            eggs[:,8] = 0
            #Parent_ID
            eggs[:,9] = Parent_ID
            #Parent_shell_size
            eggs[:,10] = Parent_shell_size
            #time_birth
            eggs[:,11] = -1
            #current_time
            eggs[:,12] = time_birth
            #aragonite, but used as parent index in matrix
            eggs[:,13] = adults_ind
            #damage accumulated
            eggs[:,14] = 0
            #temperature
            eggs[:,15] = 0
            #food
            eggs[:,16] = 0

            egg_list = np.repeat(eggs, repeats=int(num_eggs/(idx+1)), axis=0)
            egg_list[:,0] = np.arange(next_ID,next_ID+egg_list.shape[0])

            pteropod_list = np.concatenate((pteropod_list,egg_list))
            next_ID = max(pteropod_list[:,0])+1
            pteropod_list[adults_ind,6] = 1

    return pteropod_list, next_ID, current_generation

def spawning_gradual(pteropod_list, current_generation,next_ID,max_eggs=500,max_size=3.962,num_eggs_per_size=None,sizes_per_egg=None):
    """This function subsets the adult pteropods of a given generation, and determines which pteropods are ready to spawn eggs.
    The egg production is linked to the size of the pteropods
    Returns the array containing the updated attributes characterizing the pteropods, the next largest ID, and the current generation spawning. UHE 10/02/2022

    Keyword arguments:
    pteropod_list -- Array containing all state variables characterizing the pteropods
    current_generation -- Identifier of the current generation that will spawn the next generation
    next_ID -- The largest ID + 1 out of the entire population
    max_eggs -- Maximum number of eggs per adult throughout their life time
    max_size -- Maximum size of pteropods (proposed by Reviewer)
    num_eggs_per_size -- list containing the number of eggs produced at a given size
    sizes_per_eggs -- list of sizes at which a defined number of eggs is produced

    """

    #add a maximum size as proposed by Reviewer
    pteropod_list[np.squeeze(np.argwhere(pteropod_list[:,3] > max_size )).astype(int),7] = 500

    #define number of eggs
    if (num_eggs_per_size is None) or (sizes_per_egg is None):
        print("The keyword 'num_eggs_per_size' and 'sizes_per_egg' have to be declared, otherwise a default value is used.")
        num_eggs_per_size = np.array([0,1,2,4,8,16,32,64,128,256])
        sizes_per_egg = np.linspace(3.64,3.962,10)
    else:
        assert len(num_eggs_per_size) == len(sizes_per_egg), "The size of 'num_eggs_per_size' must be equal to the size of 'sizes_per_egg'"

    interp_num_eggs = scipy.interpolate.interp1d(sizes_per_egg,num_eggs_per_size)

    available_generations = pteropod_list[np.argwhere(pteropod_list[:,2] == 3),1]
    if len(np.unique(available_generations)) >= 1  and  max(np.unique(available_generations)) !=  current_generation:
        current_generation = max(np.unique(available_generations))

    #get number of adults in the current generation that can produce eggs
#     for idx in range(1):
    #get number of adults in the current generation that can produce eggs
    adults_ind = np.squeeze(np.argwhere((pteropod_list[:,2] == 3) &
                                (pteropod_list[:,1]%2 == current_generation%2) &
                                (pteropod_list[:,7] < max_eggs) &
                                (pteropod_list[:,6] == 0))).astype(int)

    if adults_ind.size == 1:
        adults_ind = np.array([adults_ind])
    #get the total number of new eggs(particles)
    if adults_ind.size > 0:
        for ind in adults_ind:
            #for each entry in adult, create egg more entries
            #get the generation, Parent_ID, Shell_size, time_of_birth
            generation = np.squeeze(pteropod_list[ind,1])
            Parent_ID = np.squeeze(pteropod_list[ind,0])
            Parent_shell_size = np.squeeze(pteropod_list[ind,3])
            time_birth = np.squeeze(pteropod_list[ind,12])

            eggs = np.random.rand(1, 17)
            #ID
            eggs[:,0] = -1
            #generation
            eggs[:,1] = generation+1
            #stage
            eggs[:,2] = 0
            #shell_size
            eggs[:,3] = 0.15
            #days_of_growth
            eggs[:,4] = 0
            #survive
            eggs[:,5] = 1
            #num. spawning events/eggs
            eggs[:,6] = 0
            #ERR
            eggs[:,7] = 0
            #spawned
            eggs[:,8] = 0
            #Parent_ID
            eggs[:,9] = Parent_ID
            #Parent_shell_size
            eggs[:,10] = Parent_shell_size
            #time_birth
            eggs[:,11] = -1
            #current_time
            eggs[:,12] = time_birth
            #aragonite, but used as parent index in matrix
            eggs[:,13] = ind
            #damage accumulated
            eggs[:,14] = 0
            #temperature
            eggs[:,15] = 0
            #food
            eggs[:,16] = 0
            repeats = int(np.round(interp_num_eggs(Parent_shell_size)))

            egg_list = np.repeat(eggs, repeats=repeats, axis=0)
            egg_list[:,0] = np.arange(next_ID,next_ID+egg_list.shape[0])

            pteropod_list = np.concatenate((pteropod_list,egg_list))
            next_ID = max(pteropod_list[:,0])+1
            pteropod_list[ind,7] += repeats
            if pteropod_list[ind,7] >= max_eggs:
                pteropod_list[ind,6] = 1

    return pteropod_list, next_ID, current_generation


