#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

stefan_boltzmann = 5.67*10**-8

def atm(nlayers, emiss, S0=1350, albedo=0.33, nuclear_winter=False):
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
        b[0] = -S0/4*(1-albedo)
    else:
        b[-1] = -S0/4*(1-albedo)
    Ainv = np.linalg.inv(A) # Get solution:
    fluxes = np.matmul(Ainv, b) # Note our use of matrix multiplication!
    temps = np.zeros(len(fluxes))
    temps[1:] = (fluxes[1:]/(emiss*stefan_boltzmann))**0.25
    temps[0]  = (fluxes[0]/(stefan_boltzmann))**0.25
    return temps

def validate():
    expected = [330.7, 298.8, 251.3]
    actual = np.round(atm(2,1),1)
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
    emiss_list = np.arange(0,1.1,0.1)
    Ts_list = np.zeros(len(emiss_list))
    for i in range(len(Ts_list)):
        Ts_list[i] = atm(1,emiss_list[i])[0]
    plt.ion()
    fig,ax = plt.subplots(1,1)
    ax.plot(emiss_list,Ts_list)
    ax.set_xlabel('Emissivity')
    ax.set_ylabel('Surface Temperature')
    ax.set_title('Surface Temperature vs Emissivity for 1-Layer Atmosphere')


def vary_N():
    N_list = np.arange(1,11,1)
    Ts_list = np.zeros(len(N_list))
    for i in range(len(Ts_list)):
        Ts_list[i] = atm(N_list[i], 0.255)[0]
    plt.ion()
    fig,ax = plt.subplots(1,1)
    ax.plot(N_list,Ts_list)
    ax.set_xlabel('Number of Atmospheric Layers')
    ax.set_ylabel('Surface Temperature')
    ax.set_title('Surface Temperature vs Number of Layers in Atmosphere')

def altitude_plot():
    layers = np.arange(0,6,1)
    Temps = atm(5,0.255)
    plt.ion()
    fig,ax = plt.subplots(1,1)
    ax.plot(Temps,layers)
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Layer of Atmosphere')
    ax.set_title('Temperature at Different Layers of 5-Layer Atmosphere')

def venus():
    N_list = np.arange(1,31,1)
    Ts_list = np.zeros(len(N_list))
    for i in range(len(Ts_list)):
        Ts_list[i] = atm(N_list[i], 1, S0=2600,)[0]
    plt.ion()
    fig,ax = plt.subplots(1,1)
    ax.plot(N_list,Ts_list)
    ax.set_xlabel('Number of Atmospheric Layers')
    ax.set_ylabel('Surface Temperature')
    ax.set_title('Surface Temperature of Venus vs Number of Layers in Atmosphere')

def nuclear_winter():
    layers = np.arange(0,6,1)
    Temps = atm(5, 0.5, nuclear_winter=True)
    plt.ion()
    fig,ax = plt.subplots(1,1)
    ax.plot(Temps,layers)
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Layer of Atmosphere')
    ax.set_title('Temperature at Different Layers of Nuclear Winter Atmosphere')
