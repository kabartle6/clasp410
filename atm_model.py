#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

'''
Lab 3: Energy Balance Atmosphere Model

The function, atm(), is where the core algorithm to calculate the temperature
of the layers of an N-layer atmosphere is implemented. Validation of the 
calculations done in atm() is done in the function, validate(). vary_emiss(),
vary_N(), and altitude_plot() create the plots to answer question 3 of the lab.
venus() creates the plots to answer question 4 of the lab, and nuclear_winter() 
creates the plots to answer question 5
'''


stefan_boltzmann = 5.67*10**-8 # Define constant

def atm(nlayers, emiss, S0=1350, albedo=0.33, nuclear_winter=False):
    '''
    This function takes in a number of layers, emissivity, S0, and albedo
    value which it uses to calculate and return the temperature at the 
    surface and each layer of the N-layer atmosphere in an array.

    Parameters
    ----------
    nlayers : int
        The number of layers in the atmosphere
    emiss : float [0,1]
        The value of the emissivity of atmosphere.
    albedo: float [0,1], default=0.33
        The value of the albedo of the planet.
    nuclear_winter : bool, default=False
        If true, the first layer of the atmosphere will absorb all of the
        incoming solar flux instead of the surface.
    Returns
    -------
    temps : numpy array of temperatures
        The first element is the surface temperature in K, followed by the 
        temperature of each layer of the atmosphere in ascending order.
    '''
    # Create array of coefficients, an N+1xN+1 array:
    A = np.zeros([nlayers+1, nlayers+1])
    b = np.zeros(nlayers+1)
    # Populate based on our model:
    for i in range(nlayers+1):
        for j in range(nlayers+1):
            if (i==j and i==0):
                A[i, j] = -1
            elif (i==j): 
                A[i, j] = -2
            elif i==0:
                A[i, j] = (1-emiss)**(abs(i-j)-1)
            else:
                A[i, j] = emiss*(1-emiss)**(abs(i-j)-1)
    if not nuclear_winter: 
        b[0] = -S0/4*(1-albedo) # Solar flux absorbed by surface
    else:
        b[-1] = -S0/4*(1-albedo) # Solar flux absorbed by top layer of atmosphere
    Ainv = np.linalg.inv(A) # Get solution:
    fluxes = np.matmul(Ainv, b) # Note our use of matrix multiplication!
    temps = np.zeros(len(fluxes))
    temps[1:] = (fluxes[1:]/(emiss*stefan_boltzmann))**0.25 # Calculate atmosphere layer temperatures
    temps[0]  = (fluxes[0]/(stefan_boltzmann))**0.25 # Calculate surface temperature
    return temps

def validate():
    '''
    This fuction performs a validation of the values calculated by the model 
    by comparing them to values calculated by the model found here:
    https://singh.sci.monash.edu/models/Nlayer/N_layer.html
    for a 2 layer atmosphere with an emissivity of 1 and a 3 layer atmosphere
    with an emissivity of 0.5
    '''
    expected = [330.7, 298.8, 251.3] # Expected values from online model
    actual = np.round(atm(2,1),1) # Rounded values produced by model defined atm()
    if np.array_equal(actual, expected):
        print('Validation 1 Successful!')
    else:
        print('ERROR')
        return
    expected = [298.8, 270.0, 251.3, 227.1]
    actual = np.round(atm(3,0.5),1)
    if np.array_equal(actual, expected):
        print('Validation 2 Successful!')
    else:
        print('ERROR')
        return
    print('Validation Complete')
    return

def vary_emiss():
    '''
    This function calculates and plots the surface temperature of a planet 
    with a 1-layer atmosphere, S0=1350, albedo=0.33, and varying emissivity.
    '''
    emiss_list = np.arange(0,1.1,0.1) # List of varying emissivities
    Ts_list = np.zeros(len(emiss_list))
    for i in range(len(Ts_list)):
        Ts_list[i] = atm(1,emiss_list[i])[0] # Calculate surface T for each emissivity
    # Plot
    fig,ax = plt.subplots(1,1)
    ax.plot(emiss_list,Ts_list)
    ax.set_xlabel('Emissivity')
    ax.set_ylabel('Surface Temperature')
    ax.set_title('Surface Temperature vs Emissivity for 1-Layer Atmosphere')
    plt.savefig('VaryEmissivity.png') # Save figure


def vary_N():
    '''
    This function calculates and plots the surface temperature of a planet 
    with S0=1350, albedo=0.33, emissivity=0.55 and varying number of 
    atmospheric layers.
    '''
    N_list = np.arange(1,11,1) # List of varying N
    Ts_list = np.zeros(len(N_list))
    for i in range(len(Ts_list)):
        Ts_list[i] = atm(N_list[i], 0.255)[0] # Calculate surface T for each N
    # Plot
    fig,ax = plt.subplots(1,1)
    ax.plot(N_list,Ts_list)
    ax.set_xlabel('Number of Atmospheric Layers')
    ax.set_ylabel('Surface Temperature')
    ax.set_title('Surface Temperature vs Number of Layers in Atmosphere')
    plt.savefig('VaryN.png') # Save figure

def altitude_plot():
    '''
    This function calulates and plots of the temperature of each layer of
    a 5-layer atmosphere with S0=1350, albedo=0.33, emissivity=0.55.
    '''
    layers = np.arange(0,6,1) # List of atmospher layers
    Temps = atm(5,0.255) # Calculate T at each layer
    # Plot vertical T profile
    fig,ax = plt.subplots(1,1)
    ax.plot(Temps,layers)
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Layer of Atmosphere')
    ax.set_title('Temperature at Different Layers of 5-Layer Atmosphere')
    plt.savefig('AltitudePlot.png') # Save figure

def venus():
    '''
    This function calculates and plots the surface temperature of Venus 
    with S0=2600, albedo=0.33, emissivity=1 and varying number of 
    atmospheric layers.
    '''
    N_list = np.arange(1,31,1) # List of varying N
    Ts_list = np.zeros(len(N_list))
    for i in range(len(Ts_list)):
        Ts_list[i] = atm(N_list[i], 1, S0=2600,)[0] # Calculate surface T for each N
    # Plot
    fig,ax = plt.subplots(1,1)
    ax.plot(N_list,Ts_list)
    ax.set_xlabel('Number of Atmospheric Layers')
    ax.set_ylabel('Surface Temperature')
    ax.set_title('Surface Temperature of Venus vs Number of Layers in Atmosphere')
    plt.savefig('Venus.png') # Save figure

def nuclear_winter():
    '''
    This function calulates and plots of the temperature of each layer of
    a 5-layer atmosphere with S0=1350, albedo=0.33, emissivity=0.55  experiencing 
    nuclear winter such that all solar flux is absorbed at the top of the atmosphere.
    '''
    layers = np.arange(0,6,1) # List of atmospher layers
    Temps = atm(5, 0.5, nuclear_winter=True) # Calculate T at each layer
    # Plot vertical T profile
    fig,ax = plt.subplots(1,1)
    ax.plot(Temps,layers)
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Layer of Atmosphere')
    ax.set_title('Temperature at Different Layers of Nuclear Winter Atmosphere')
    plt.savefig('NuclearWinter.png') # Save figure
