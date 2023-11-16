#!/usr/bin/env python3

# Import libraries
import numpy as np
import matplotlib.pyplot as plt

radearth = 6357000. # Earth radius in meters.
lamb = 100. # Thermal diffusivity of atmosphere.
# mxlyr = 50. # depth of mixed layer (m)
# sigma = 5.67e-8 # Stefan-Boltzmann constant
# C = 4.2e6 # Heat capacity of water
# rho = 1020 # Density of sea-water

# Set initial temperature curve in Celcius:
T_warm = np.array([-47, -19, -11, 1, 9, 14, 19, 23, 25, 25,
                   23, 19, 14, 9, 1, -11, -19, -47])

def snowearth(npoints=18, dt=1, tstop=10000):
    '''
    Parameters
    ----------
    npoints : integer, default 18
        Number of latitude grid points
    '''