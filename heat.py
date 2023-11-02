#!/usr/bin/env python3

'''
Lab 4: Heat Diffusion and Permafrost

The function, heat_solve(), is where the core algorithm to calculate heat diffusion
is implemented. The function, test(), validates heat_solve(). The function, 
permafrost() creates the plots to answer question 2 of the lab and warming() 
creates the plots to answer question 3 of the lab.

'''

import numpy as np
import matplotlib.pyplot as plt

def sample_init(x):
    '''Simple inital boundary conition function'''
    return 4*x - 4*x**2

# Kangerlussuaq average temperature:
t_kanger = np.array([-19.7, -21.0, -17., -8.4, 2.3, 8.4,
                     10.7, 8.5, 3.1, -6.0, -12.0, -16.9])

def temp_kanger(t): 
    '''
    For an array of times in days, return timeseries of temperature 
    for Kangerlussuaq, Greenland.
    '''
    t_amp = (t_kanger - t_kanger.mean()).max()
    return t_amp*np.sin(np.pi/180 * t - np.pi/2) + t_kanger.mean()

def warm_kanger1(t): 
    '''
    For an array of times in days, return timeseries of temperature 
    for Kangerlussuaq, Greenland with 0.5 degrees warming.
    '''
    t_amp = (t_kanger - t_kanger.mean()).max()
    return t_amp*np.sin(np.pi/180 * t - np.pi/2) + t_kanger.mean() + 0.5

def warm_kanger2(t): 
    '''
    For an array of times in days, return timeseries of temperature 
    for Kangerlussuaq, Greenland with 1 degrees warming.
    '''
    t_amp = (t_kanger - t_kanger.mean()).max()
    return t_amp*np.sin(np.pi/180 * t - np.pi/2) + t_kanger.mean() + 1

def warm_kanger3(t): 
    '''
    For an array of times in days, return timeseries of temperature 
    for Kangerlussuaq, Greenland with 3 degrees warming.
    '''
    t_amp = (t_kanger - t_kanger.mean()).max()
    return t_amp*np.sin(np.pi/180 * t - np.pi/2) + t_kanger.mean() + 3


def heat_solve(xmax=1.0, dx=0.2, tmax=0.2, dt=0.02, c2=1.0, init=sample_init, 
               top=0, bottom=0, Neumann=False):
    '''
    This function solves the 1D heat diffusion equation using a forward difference
    method given the parameters listed below.

    Parameters
    ----------
    xmax, tmax : float, default=1.0,0.2
        Total space and time heat diffusion is calculated for
    dt, dx : float, default=0.02,0.2
        Tima and space step.
    c2 : float
        Thermal diffusivity.
    init : scalar or function, default=sample_init
        Set initial condition. If a function, should take position
        as input and return temperature
    top, bottom : scalar or function, default=0,0
        Set boundary conditions. If a function, should take time
        as input and return temperature


    Returns
    -------
    x : numpy vector
        Array of position locations/x-grid
    t : numpy vector
        Array of times
    temp : numpy array

    '''

    # Check stability criterion
    if (dt > dx**2/(2*c2)):
        raise ValueError(f'Stability Criterion not met: dt={dt:6.2f}; dx={dx:06.2f}, c2={c2}')
    # set consatnts
    r = c2 * dt/dx**2

    # CReatet space and time grids
    x = np.arange(0,xmax+dx, dx)
    t = np.arange(0,tmax+dt, dt)
    # Save number of points
    M,N = x.size, t.size
    # Create temperature solution array
    temp = np.zeros([M, N])

    # Set initial condion
    if callable(init):
        temp[:,0] = init(x)
    else:
        temp[:,0] = init

    # Set boundary conditions
    if callable(top):
        temp[0,:] = top(t)
    else:
        temp[0,:] = top
    if callable(bottom):
        temp[-1,:] = bottom(t)
    else:
        temp[-1,:] = bottom

    # Solve!
    if Neumann: # Neumann solution
        for j in range(1, N):
            for i in range(1, M-1):
                temp[i,j] = (1-2*r)*temp[i,j-1] + r*(temp[i+1,j-1] + temp[i-1,j-1])
            temp[0,j] = temp[1,j]
            temp[-1,j] = temp[-2,j]

    else: # Dirichlet solution
        for j in range(1, N):
            temp[1:-1, j] = (1-2*r)*temp[1:-1,j-1] + r*(temp[2:,j-1] + temp[:-2,j-1])

    return x, t, temp


def test():
    '''
    This function performs a validation of heat_solve by comparing the solution
    to an example heat diffusion scenerio with the expected soltuion.

    Returns
    -------
    verify : numpy array
        Difference between expected and actual solution.
    '''
    solution = np.array([
    [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
    [0.640000, 0.480000, 0.400000, 0.320000, 0.260000, 0.210000, 0.170000, 0.137500, 0.111250, 0.090000, 0.072812],
    [0.960000, 0.800000, 0.640000, 0.520000, 0.420000, 0.340000, 0.275000, 0.222500, 0.180000, 0.145625, 0.117813],
    [0.960000, 0.800000, 0.640000, 0.520000, 0.420000, 0.340000, 0.275000, 0.222500, 0.180000, 0.145625, 0.117813],
    [0.640000, 0.480000, 0.400000, 0.320000, 0.260000, 0.210000, 0.170000, 0.137500, 0.111250, 0.090000, 0.072812],
    [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]])

    # Get solution using your solver:
    x, time, heat = heat_solve()

    max_error = (heat-solution).max()
    
    # Create a figure/axes object
    fig, axes = plt.subplots(1, 1)
    # Create a color map and add a color bar.
    map = axes.pcolor(time, x, heat, cmap='Purples_r')
    plt.colorbar(map, ax=axes, label='Temperature ($C$)')
    plt.title('Validating Solver 1D Heat Diffusivity')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Spacial Step')
    plt.savefig('test.png')
    axes.annotate('Max Error: {}'.format(max_error), xy=(0.2,-0.22), 
                xycoords='axes fraction', fontsize=9)
    
    return max_error

def boundaries():
    '''
    This function creates the plots for the HW assignment comparing 
    Neumann and Dirichlet boundary conditions
    '''
    # Dirichlet solution:
    x, time, heat = heat_solve(dx=0.02, dt=0.0002)

    # Create a figure/axes object
    fig, axes = plt.subplots(1, 1)
    # Create a color map and add a color bar.
    map = axes.pcolor(time, x, heat, cmap='plasma')
    plt.colorbar(map, ax=axes, label='Temperature ($C$)')
    plt.title('Heat Diffusion with Dirichlet Boundary Conditions')
    plt.xlabel('Time (s)')
    plt.ylabel('Distance (m)')
    plt.savefig('Dirichlet.png')

    # Neumann solution:
    x, time, heat = heat_solve(dx=0.02, dt=0.0002, Neumann=True)

    # Create a figure/axes object
    fig, axes = plt.subplots(1, 1)
    # Create a color map and add a color bar.
    map = axes.pcolor(time, x, heat, cmap='plasma')
    plt.colorbar(map, ax=axes, label='Temperature ($C$)')
    plt.title('Heat Diffusion with Neumann Boundary Conditions')
    plt.xlabel('Time (s)')
    plt.ylabel('Distance (m)')
    plt.savefig('Neumann.png')

def permafrost():
    '''
    This function creates the plots for Kangerlussuaq's ground temperature,
    answering question 2 of the lab assignment.
    '''
    # Get solution using your solver:
    x, time, heat = heat_solve(tmax=18250,xmax=100, dt=10, dx=1, c2=0.0216, bottom=5, top=temp_kanger, init=0) 
    # Create a figure/axes object
    fig, axes = plt.subplots(1, 1)
    # Create a color map and add a color bar.
    map = axes.pcolor(time/360, x, heat, cmap='seismic', vmin=-25, vmax=25)
    plt.colorbar(map, ax=axes, label='Temperature ($C$)')
    plt.gca().invert_yaxis()
    plt.xlabel('Time (Years)')
    plt.ylabel('Depth (m)')
    plt.title('Ground Temperature: Kangerlussuaq, Greenland')
    plt.savefig('HeatMap.png')
    
    # Set indexing for the final year of results:
    loc = int(-365) # Final 365 days of the result.
    # Extract the min values over the final year:
    winter = heat[:, loc:].min(axis=1)
    # Extract the max values over the final year:
    summer = heat[:, loc:].max(axis=1)
    # Create a temp profile plot:
    fig, ax2 = plt.subplots(1, 1, figsize=(10, 8))
    ax2.plot(winter, x, label='Winter')
    ax2.plot(summer, x, label='Summer', linestyle='--', color='red')
    plt.gca().invert_yaxis()
    plt.legend()
    plt.xlabel('Time (Years)')
    plt.ylabel('Depth (m)')
    plt.title('Ground Temperature: Kangerlussuaq, Greenland')
    plt.grid()
    plt.savefig('TempProfile.png')

def warming():
    '''
    This function creates the plots for Kangerlussuaq's ground temperature
    with 0.5, 1, and 3 degrees C of uniform temperature rise, answering 
    question 3 of the lab assignment.
    '''
    # Get solution using your solver for 0.5 degrees of warming:
    x, time, heat = heat_solve(tmax=18250,xmax=100, dt=10, dx=1, c2=0.0216, bottom=5, top=warm_kanger1, init=0) 
    # Create a figure/axes object
    fig, axes = plt.subplots(1, 1)
    # Create a color map and add a color bar.
    map = axes.pcolor(time/360, x, heat, cmap='seismic', vmin=-25, vmax=25)
    plt.colorbar(map, ax=axes, label='Temperature ($C$)')
    plt.gca().invert_yaxis()
    plt.xlabel('Time (Years)')
    plt.ylabel('Depth (m)')
    plt.title('Ground Temperature: Kangerlussuaq, Greenland')
    plt.savefig('HeatMap0.5.png')
    
    # Set indexing for the final year of results:
    loc = int(-365) # Final 365 days of the result.
    # Extract the min values over the final year:
    winter = heat[:, loc:].min(axis=1)
    # Extract the max values over the final year:
    summer = heat[:, loc:].max(axis=1)
    # Create a temp profile plot:
    fig, ax2 = plt.subplots(1, 1, figsize=(10, 8))
    ax2.plot(winter, x, label='Winter')
    ax2.plot(summer, x, label='Summer', linestyle='--', color='red')
    plt.gca().invert_yaxis()
    plt.legend()
    plt.xlabel('Time (Years)')
    plt.ylabel('Depth (m)')
    plt.title('Ground Temperature: Kangerlussuaq, Greenland')
    plt.grid()
    plt.savefig('TempProfile0.5.png')

    # Get solution using your solver for 1 degree of warming:
    x, time, heat = heat_solve(tmax=18250,xmax=100, dt=10, dx=1, c2=0.0216, bottom=5, top=warm_kanger2, init=0) 
    # Create a figure/axes object
    fig, axes = plt.subplots(1, 1)
    # Create a color map and add a color bar.
    map = axes.pcolor(time/360, x, heat, cmap='seismic', vmin=-25, vmax=25)
    plt.colorbar(map, ax=axes, label='Temperature ($C$)')
    plt.gca().invert_yaxis()
    plt.xlabel('Time (Years)')
    plt.ylabel('Depth (m)')
    plt.title('Ground Temperature: Kangerlussuaq, Greenland')
    plt.savefig('HeatMap1.png')
    
    # Set indexing for the final year of results:
    loc = int(-365) # Final 365 days of the result.
    # Extract the min values over the final year:
    winter = heat[:, loc:].min(axis=1)
    # Extract the max values over the final year:
    summer = heat[:, loc:].max(axis=1)
    # Create a temp profile plot:
    fig, ax2 = plt.subplots(1, 1, figsize=(10, 8))
    ax2.plot(winter, x, label='Winter')
    ax2.plot(summer, x, label='Summer', linestyle='--', color='red')
    plt.gca().invert_yaxis()
    plt.legend()
    plt.xlabel('Time (Years)')
    plt.ylabel('Depth (m)')
    plt.title('Ground Temperature: Kangerlussuaq, Greenland')
    plt.grid()
    plt.savefig('TempProfile1.png')

    # Get solution using your solver for 3 degrees of warming:
    x, time, heat = heat_solve(tmax=18250,xmax=100, dt=10, dx=1, c2=0.0216, bottom=5, top=warm_kanger3, init=0) 
    # Create a figure/axes object
    fig, axes = plt.subplots(1, 1)
    # Create a color map and add a color bar.
    map = axes.pcolor(time/360, x, heat, cmap='seismic', vmin=-25, vmax=25)
    plt.colorbar(map, ax=axes, label='Temperature ($C$)')
    plt.gca().invert_yaxis()
    plt.xlabel('Time (Years)')
    plt.ylabel('Depth (m)')
    plt.title('Ground Temperature: Kangerlussuaq, Greenland')
    plt.savefig('HeatMap3.png')
    
    # Set indexing for the final year of results:
    loc = int(-365) # Final 365 days of the result.
    # Extract the min values over the final year:
    winter = heat[:, loc:].min(axis=1)
    # Extract the max values over the final year:
    summer = heat[:, loc:].max(axis=1)
    # Create a temp profile plot:
    fig, ax2 = plt.subplots(1, 1, figsize=(10, 8))
    ax2.plot(winter, x, label='Winter')
    ax2.plot(summer, x, label='Summer', linestyle='--', color='red')
    plt.gca().invert_yaxis()
    plt.legend()
    plt.xlabel('Time (Years)')
    plt.ylabel('Depth (m)')
    plt.title('Ground Temperature: Kangerlussuaq, Greenland')
    plt.grid()
    plt.savefig('TempProfile3.png')   
    