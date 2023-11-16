#!/usr/bin/env python3

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Read in data
c0 = 288
year = np.arange(1870,2101,1)
y = 4.05*(10**-13)*np.exp(0.1946*(year))
plt.plot(year,y)
plt.show()
# Calculate forcing
# Q = 5.35*np.log(df['co2']/c0)

def transient_temp(lambdaR, dz, forcing):
    '''
    This function calculates transient temperature change given lambaR, dz, 
    and an array of radiative forcing.
    '''

    time = (year-1870)*(365*60*60*24) # Time array in sec
    # Define constants
    cw = 4218 # Ocean heat capacity
    rho = 1000 # Ocean density
    ce = cw*rho*dz
    tao = lambdaR*ce
    midpoint_sum = 0
    T_list = []

    # Calculate T' for each t using midpoint method
    for i in range(0,len(forcing)-1):
        midpoint = ((forcing[i] + forcing[i+1])/(ce*2))
        midpoint_sum += np.exp((time[i]+0.5*365*60*60*24)/tao)*midpoint*365*60*60*24
        T = np.exp(-(time[i+1])/tao)*(midpoint_sum)
        T_list.append(T)

    # Return T' for all t
    return T_list 