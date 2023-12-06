#!/usr/bin/env python3

'''
Lab 5: Snowball Earth Model

The function, snowearth(), is where the core algorithm to use an implicit solver for
calculating heat diffusion to find the temperature at different latitudes of Earth after
a given amount of time is implemented. example_plot() recreates the plot found in the
lab assignment. The functions, question2(), question3(), and question4() create the 
plots to answer questions 2, 3, and 4 of the lab respectively. All other functions were
provided by Dan :)
'''

# Import libraries
import numpy as np
import matplotlib.pyplot as plt

radearth = 6357000. # Earth radius in meters.
lamb = 100. # Thermal diffusivity of atmosphere.
mxlyr = 50. # depth of mixed layer (m)
sigma = 5.67e-8 # Stefan-Boltzmann constant
C = 4.2e6 # Heat capacity of water
rho = 1020 # Density of sea-water

# Set initial temperature curve in Celcius:
T_warm = np.array([-47, -19, -11, 1, 9, 14, 19, 23, 25, 25,
                   23, 19, 14, 9, 1, -11, -19, -47])

def temp_warm(lats_in):
    '''
    Create a temperature profile for modern day "warm" earth.

    Parameters
    ----------
    lats_in : Numpy array
        Array of latitudes in degrees where temperature is required.
        0 corresponds to the south pole, 180 to the north.

    Returns
    -------
    temp : Numpy array
        Temperature in Celcius.
    '''

    # Set initial temperature curve
    T_warm = np.array([-47, -19, -11, 1, 9, 14, 19, 23, 25, 25,
                       23, 19, 14, 9, 1, -11, -19, -47])

    # Get base grid:
    npoints = T_warm.size
    dlat = 180 / npoints  # Latitude spacing.
    lats = np.linspace(dlat/2., 180-dlat/2., npoints)  # Lat cell centers.

    # Fit a parabola to the above values
    coeffs = np.polyfit(lats, T_warm, 2)

    # Now, return fitting sampled at "lats".
    temp = coeffs[2] + coeffs[1]*lats_in + coeffs[0] * lats_in**2

    return temp

def insolation(S0, lats):
    '''
    Given a solar constant (`S0`), calculate average annual, longitude-averaged
    insolation values as a function of latitude.
    Insolation is returned at position `lats` in units of W/m^2.

    Parameters
    ----------
    S0 : float
        Solar constant (1370 for typical Earth conditions.)
    lats : Numpy array
        Latitudes to output insolation. Following the grid standards set in
        the diffusion program, polar angle is defined from the south pole.
        In other words, 0 is the south pole, 180 the north.

    Returns
    -------
    insolation : numpy array
        Insolation returned over the input latitudes.
    '''

    # Constants:
    max_tilt = 23.5   # tilt of earth in degrees

    # Create an array to hold insolation:
    insolation = np.zeros(lats.size)

    #  Daily rotation of earth reduces solar constant by distributing the sun
    #  energy all along a zonal band
    dlong = 0.01  # Use 1/100 of a degree in summing over latitudes
    angle = np.cos(np.pi/180. * np.arange(0, 360, dlong))
    angle[angle < 0] = 0
    total_solar = S0 * angle.sum()
    S0_avg = total_solar / (360/dlong)

    # Accumulate normalized insolation through a year.
    # Start with the spin axis tilt for every day in 1 year:
    tilt = [max_tilt * np.cos(2.0*np.pi*day/365) for day in range(365)]

    # Apply to each latitude zone:
    for i, lat in enumerate(lats):
        # Get solar zenith; do not let it go past 180. Convert to latitude.
        zen = lat - 90. + tilt
        zen[zen > 90] = 90
        # Use zenith angle to calculate insolation as function of latitude.
        insolation[i] = S0_avg * np.sum(np.cos(np.pi/180. * zen)) / 365.

    # Average over entire year; multiply by S0 amplitude:
    insolation = S0_avg * insolation / 365

    return insolation

def gen_grid(npoints=18):
    '''
    A covenience function for creating the grid. Creates a uniform latitudinal 
    grid from pole-to-pole.

    Parameters
    ----------
    npoints : The number of points on the grid.

    Returns
    -------
    dlat : float
        Latitude spacing in degrees.
    lats : Nupy array
        Latitudes in degrees.
    edge : numpy array
        Latitude bin edges in degrees.
    '''
    dlat = 180/npoints
    lats = np.linspace(dlat/2., 180-dlat/2., npoints)
    edge = np.linspace(0,180, npoints+1)

    return dlat, lats, edge

def snowearth(npoints=18, dt=1, tstop=10000, lamb = 100., S0=1370, emiss=1.0, dynamic=False,
              albedo_gnd=0.3, albedo_ice=0.6, t_init = None, flash_freeze=False,
              sphere=True, insolate=True, gamma=1.0):
    '''
    This function implements an implicit solver for calculating heat diffusion to find the 
    temperature at different latitudes of Earth after an amount of time determined by tstop
    and given initial conditions using the parameters outlined below.
    '''
    '''
    Parameters
    ----------
    npoints : integer, default 18
        Number of latitude grid points.
    dt : float, default 1
        Timestep in years.
    tstop : foat, default 10000
        Number of years to run model
    lamb : float, default 100
        Set diffusion coefficient.
    S0 : float, default 1370
        Solar constant.
    emiss : float, default 1
        Emissivity of the earth.
    dynamic : bool, default False
        Sets a dynamic albedo that changes between ice and no ice when set to True.
    albedo_gnd : float, default 0.3
        Albedo when ground is not covered in ice.
    albedo_ice : float, default 0.6
        Albedo of when ground is covered in ice.
    t_init : float or numpy array, default None
        Initial temperature condition at each latitude.
    flash_freeze : bool, default False
        Sets albedo to a constant value of 0.6 when True.
    sphere : bool, default True
        Include spherical correction.
    insolate : bool, default True
        Include radiative forcing term.
    gamma : float, default 1.0
        Insolation multiplier.
    

    Returns
    -------
    lats : numpy array
        An array of latitudes in degrees representing our solution grid.
    Temp : numpy array
        An array of temperatures on our grid at the end of the simulation
        in Celsius.
    '''

    # Create grid
    dlat, lats, edges = gen_grid(npoints)

    # Set delta-y:
    dy = np.pi * radearth * dlat/180

    # Set timestep in seconds
    dt_sec = dt * 365 * 24 * 3600

    # Set initial condition 
    Temp = np.ones(len(lats))
    if t_init is None:
        Temp = temp_warm(lats)
    else:
        Temp = Temp*t_init

    # Create tri-diag "A" matrix
    A = np.zeros((npoints, npoints))
    A[np.arange(npoints), np.arange(npoints)] = -2
    A[np.arange(npoints-1), np.arange(npoints-1)+1] = 1
    A[np.arange(npoints-1)+1, np.arange(npoints-1)] = 1

    # Apply zero-flux BCs
    A[0,0] = A[-1,-1] = -2
    A[0,1] = A[-1,-2] = 2
    A *= (1/dy)**2
    
    # Get matrix for advancing solution
    L = np.eye(npoints) - dt_sec * lamb * A
    Linv = np.linalg.inv(L)

    # Create matrix
    # Corner values for first order accurate Neumann boundary conditions.
    B = np.zeros((npoints, npoints))
    B[np.arange(npoints-1), np.arange(npoints-1)+1] = 1
    B[np.arange(npoints-1)+1, np.arange(npoints-1)] = -1
    B[0, :] = B[-1, :] = 0

    Axz = np.pi * ((radearth+50.0)**2 - radearth**2) * np.sin(np.pi/180.*lats)
    dAxz = np.matmul(B, Axz) / (Axz * 4 * dy**2)

    # Get total number of steps
    nsteps = int(tstop/dt)

    # Set insolation
    insol = gamma * insolation(S0, lats)

    # Set albedo
    if flash_freeze:
        albedo = np.ones(len(Temp)) * 0.6
    else:
        albedo = np.ones(len(Temp)) * 0.3

    # Solve!
    for i in range(nsteps):
        if dynamic:
            # Add dynamic albedo
            loc_ice = Temp <= -10
            albedo[loc_ice] = albedo_ice
            albedo[~loc_ice] = albedo_gnd

        if sphere:
            # Add spherical correction
            spherecorr = lamb*dt_sec*np.matmul(B, Temp)*dAxz
            Temp += spherecorr

        if insolate:
            # Add insolation term
            radiative = (1-albedo)*insol - emiss*sigma*(Temp+273)**4
            Temp += dt_sec * radiative / (rho*C*mxlyr)

        # Calculate temperature
        Temp = np.matmul(Linv, Temp) 

    return lats, Temp


def example_plot():
    '''
    This function creates a plot to validate the model solution against Figure 1 
    of the lab assignment document.
    '''
    # Basic Diffusion
    lats, temp1 = snowearth(sphere=False, insolate=False)
    # Diff + spherical
    lats, temp2 = snowearth(insolate=False)
    # Diff + spherical + radiative
    lats, temp3 = snowearth()
    # Initial Condition
    warm = temp_warm(lats)

    # Plot
    plt.figure()
    plt.plot(lats,warm, label='Initial Condition')
    plt.plot(lats,temp1, label='Basic Diffusion')
    plt.plot(lats,temp2, label='Diff + Spherical')
    plt.plot(lats,temp3, label='Diff + Spherical + Radiative')
    plt.xlabel('Latitude')
    plt.ylabel('Temperature (Degrees C)')
    plt.legend()
    plt.savefig('fig1.png')

def question2():
    '''
    This function creates the plots to answer question 3 of the lab.
    '''

    # Create array of lambdas and emissivities
    lambs = [0, 50, 100, 150]
    emisses = [0, 0.25, 0.5, 0.75, 1]

    # Calculate solution for each lamba value
    for i in range(len(lambs)):
        lats, temp = snowearth(lamb=lambs[i])
        plt.figure()
        plt.plot(lats, temp)
        plt.xlabel('Latitude')
        plt.ylabel('Temperature (Degrees C)')
        plt.title(f'Diffusivity = {lambs[i]}')
        plt.savefig(f'fig2_{i}.png')

    # Calculate solution for each emissivity value
    for i in range(len(emisses)):
        lats, temp = snowearth(emiss=emisses[i])
        plt.figure()
        plt.plot(lats, temp)
        plt.xlabel('Latitude')
        plt.ylabel('Temperature (Degrees C)')
        plt.title(f'Emissivity = {emisses[i]}')
        plt.savefig(f'fig3_{i}.png')
    
    # Reproduce warm-Earth 
    lats, temp = snowearth(lamb=55, emiss=0.72)
    # Initial condition
    warm = temp_warm(lats)

    # Plot
    plt.figure()
    plt.plot(lats, temp, label='Diffusivity = 55, Emissivity = 0.72')
    plt.plot(lats, warm, label='Initial Condition')
    plt.xlabel('Latitude')
    plt.ylabel('Temperature (Degrees C)')
    plt.title('Warm Earth Equilibrium')
    plt.legend()
    plt.savefig('fig4.png')


def question3():
    '''
    This function creates the plots to answer question 3 of the lab.
    '''
    # Hot Earth
    lats, temp1 = snowearth(lamb=55, emiss=0.72, t_init=60, dynamic=True)
    # Cold Earth
    lats, temp2 = snowearth(lamb=55, emiss=0.72, t_init=-60, dynamic=True)
    # Flash Freeze
    lats, temp3 = snowearth(lamb=55, emiss=0.72, flash_freeze=True)
    # Initial Condition
    warm = temp_warm(lats)

    # Plot
    plt.figure()
    plt.plot(lats, temp1, label='Warm Earth')
    plt.plot(lats, temp2, label='Cold Earth')
    plt.plot(lats, temp3, label='Flash Freeze')
    plt.plot(lats, warm, label='Initial Condition')
    plt.xlabel('Latitude')
    plt.ylabel('Temperature (Degrees C)')
    plt.legend()
    plt.savefig('fig5.png')


def question4():
    '''
    This function creates the plots to answer question 4 of the lab.
    '''
    # Create array of incrementing and decrementing gammas
    gammas = np.arange(0.4, 1.45, 0.05)
    gammas = np.append(gammas, np.arange(1.4, 0.35, -0.05))

    # Initialize array for average temperatures
    avg_temp = np.zeros(len(gammas))
    
    # Calculate inital average equilibrium temperature
    lats, temp = snowearth(lamb=55, emiss=0.72, gamma=gammas[0], t_init=-60, dynamic=True, npoints=50)
    avg_temp[0] = np.mean(temp)
    
    # Find average equilibrium temperature for all gammas
    for i in range(1,len(gammas)):
        lats, temp = snowearth(lamb=55, emiss=0.72, gamma=gammas[i], t_init=temp, dynamic=True, npoints=50)
        avg_temp[i] = np.mean(temp)
        
    # Plot
    plt.figure()
    plt.plot(gammas[0:int(len(gammas)/2)], avg_temp[0:int(len(gammas)/2)], label='Increasing Gammas')
    plt.plot(gammas[int(len(gammas)/2):len(gammas)], avg_temp[int(len(gammas)/2):len(gammas)], label='Decreasing Gammas')
    plt.xlabel('Gamma')
    plt.ylabel('Average Temperature')
    plt.legend()
    plt.savefig('fig6.png')
