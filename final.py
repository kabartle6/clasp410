#!/usr/bin/env python3

'''
Final Project

The functions, forcing, transient temp and glacier melt, calculate the radiative forcing, 
transient temperature change, and change in glacier length caused by increasing CO2 
concentrations. Run this script to create the plots for my final project.
'''


# Import libraries
import numpy as np
import matplotlib.pyplot as plt


def forcing(c, c0):
    '''
    This function calculates forcing due to change in CO2 concentration from
    c0 to c.

    Paramters
    ---------
    c : float or numpy array
        Concentration of CO2 in ppm
    c0 : float
        Initial value for concrentration of CO2 (i.e. value at which forcing = 0)

    Returns
    -------
    Q : float or numpy array
        Calcalted forcing from CO2 changing from a concentration of c0 to c
    '''
    Q = 5.35*np.log(c/c0)
    return Q


def transient_temp(forcing, years):
    '''
    This function calculates transient temperature change since 1870 given a numpy 
    array of radiative forcing and a numpy array of years.

    Paramters
    ---------
    forcing : numpy array
        Radiative forcing for each year in years array
    years : numpy array
        Transient temperature change is calculated for each year in this array

    Returns
    -------
    t_list
    '''


    time = (years-1870)*(365*60*60*24) # Time array in sec
    # Define constants
    cw = 4218 # Ocean heat capacity
    rho = 1000 # Ocean density
    dz = 70 # Mixing depth
    ce = cw*rho*dz # Effective heat capacity
    lambdaR = 1.0 # Climate sensitivty
    tao = lambdaR*ce # Climate response time
    midpoint_sum = 0 # Used for integration
    T_list = np.zeros(len(forcing)-1) # Transient temperature array

    # Calculate T' for each t by integrating transient temperature equation using midpoint method
    for i in range(0,len(forcing)-1):
        midpoint = ((forcing[i] + forcing[i+1])/(ce*2))
        midpoint_sum += np.exp((time[i]+0.5*365*60*60*24)/tao)*midpoint*365*60*60*24
        T = np.exp(-(time[i+1])/tao)*(midpoint_sum)
        T_list[i] = T

    # Return T' for all t
    return T_list 


def glacier_melt(T_list, slope):
    '''
    Calculate the change in terminus position of a glacier due to transient 
    temperature change given the slope of the glacier.

    Parameters
    ----------
    T_list : numpy array
        Change in temperature from 1870 in degrees C
    slope : float
        Slope of glacier in km/km

    Returns
    -------
    melt : numpy array
        Change in terminus position of glacier from 1870 in km
    '''
    gamma = -7 # Lapse rate in degrees C/km
    melt = 2/(slope*gamma) * T_list
    return melt


'''
Question 1
'''
c0 = 289 # CO2 concentration in 1870
year = np.arange(1870,2101,1) # Array of years spanned
# Calculate CO2 concentration for each year
co2 = 295 - 0.111*(year-1870) + 1.57*10**(-3)*(year-1870)**2 + 3.03*10**(-5)*(year-1870)**3
plt.figure()
plt.plot(year, co2)
plt.xlabel('Year')
plt.ylabel('CO2 (ppm)')
plt.savefig('co2_1.png')

# Calculate forcing
Q = forcing(co2,c0)
plt.figure()
plt.plot(year, Q)
plt.xlabel('Year')
plt.ylabel('CO2 Focring (W/m2)')
plt.savefig('focing1.png')

# Calculate transient temperature change
T = transient_temp(Q, year)
plt.figure()
plt.plot(year[:-1], T)
plt.xlabel('Year')
plt.ylabel('Transient Temperature Change (Degrees C)')
plt.savefig('temp1.png')

# Calculate change in terminus position
melt = glacier_melt(T, 0.5)
plt.figure()
plt.plot(year[:-1], melt)
plt.xlabel('Year')
plt.ylabel('Change in Glacier Terminus Position from 1870 (km)')
plt.savefig('melt1.png')


'''
Question 2
'''
# Calculate the change in glacier terminus position for various slopes
slope_list = np.arange(0.1, 1.1, 0.1)
melt_list = np.zeros(len(slope_list))
for i in range(len(slope_list)):
    melt = glacier_melt(T, slope_list[i])
    melt_list[i] = melt[2023-1870]

plt.figure()
plt.plot(slope_list, melt_list)
plt.xlabel('Glacier Slope')
plt.ylabel('Change in Glacier Terminus Position from 1870 to 2023')
plt.savefig('melt2.png')

'''
Question 3
'''
# CO2 if we stop emissions in 2023
co2[(2023-1870):] = co2[2023-1870]
plt.figure()
plt.plot(year, co2)
plt.xlabel('Year')
plt.ylabel('CO2 (ppm)')
plt.savefig('co2_3.png')

# Calculate forcing
Q = forcing(co2,c0)
plt.figure()
plt.plot(year, Q)
plt.xlabel('Year')
plt.ylabel('CO2 Focring (W/m2)')
plt.savefig('forcing3.png')

# Calculate transient temperature change
T = transient_temp(Q, year)
plt.figure()
plt.plot(year[:-1], T)
plt.xlabel('Year')
plt.ylabel('Transient Temperature Change (Degrees C)')
plt.savefig('temp3.png')

# Calculate change in terminus position
melt = glacier_melt(T, 0.5)
plt.figure()
plt.plot(year[:-1], melt)
plt.xlabel('Year')
plt.ylabel('Change in Glacier Terminus Position from 1870 (km)')
plt.savefig('melt3.png')


'''
Question 4
'''
# CO2 if concentrations increase linearly by 1 ppm/yr starting in 2023
co2[(2023-1870):] += np.arange(0, 78)
plt.figure()
plt.plot(year, co2)
plt.xlabel('Year')
plt.ylabel('CO2 (ppm)')
plt.savefig('co2_4.png')

# Calculate forcing
Q = forcing(co2,c0)
plt.figure()
plt.plot(year, Q)
plt.xlabel('Year')
plt.ylabel('CO2 Focring (W/m2)')
plt.savefig('forcing4.png')

# Calculate transient temperature change
T = transient_temp(Q, year)
plt.figure()
plt.plot(year[:-1], T)
plt.xlabel('Year')
plt.ylabel('Transient Temperature Change (Degrees C)')
plt.savefig('temp4.png')

# Calculate change in terminus position
melt = glacier_melt(T, 0.5)
plt.figure()
plt.plot(year[:-1], melt)
plt.xlabel('Year')
plt.ylabel('Change in Glacier Terminus Position from 1870 (km)')
plt.savefig('melt4.png')



# CO2 if concentrations increase linearly by 0.5 ppm/yr starting in 2023
co2[(2023-1870):] = co2[2023-1870]
co2[(2023-1870):] += np.arange(0, 39, 0.5)
plt.figure()
plt.plot(year, co2)
plt.xlabel('Year')
plt.ylabel('CO2 (ppm)')
plt.savefig('co2_4_2.png')

# Calculate forcing
Q = forcing(co2,c0)
plt.figure()
plt.plot(year, Q)
plt.xlabel('Year')
plt.ylabel('CO2 Focring (W/m2)')
plt.savefig('forcing4_2.png')

# Calculate transient temperature change
T = transient_temp(Q, year)
plt.figure()
plt.plot(year[:-1], T)
plt.xlabel('Year')
plt.ylabel('Transient Temperature Change (Degrees C)')
plt.savefig('temp4_2.png')

# Calculate change in terminus position
melt = glacier_melt(T, 0.5)
plt.figure()
plt.plot(year[:-1], melt)
plt.xlabel('Year')
plt.ylabel('Change in Glacier Terminus Position from 1870 (km)')
plt.savefig('melt4_2.png')