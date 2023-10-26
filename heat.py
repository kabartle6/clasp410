#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def sample_init(x):
    '''Simple inital boundary conition function'''
    return 4*x - 4*x**2

def heat_solve(xmax=1.0, dx=0.2, tmax=0.2, dt=0.02, c2=1.0, init=sample_init, top=0, bottom=0, Neumann=False):
    '''
    dt, dx : float, default=0.02,0.2
        Tima and space step.
    c2 : float
        Thermal diffusivity.
    init : scalar or function function
        Set initial condition. If a function, should take position
        as input and return temperature
    Returns
    -------
    x : numpy vector
        Array of position locations/x-grid
    t : numpy vector
        Array of times
    temp : numpy array

    '''

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
    if Neumann:
        # temp[0,0] = temp[1,0]
        # temp[-1,0] = temp[-2,0]
        for j in range(1, N):
            for i in range(1, M-1):
                temp[i,j] = (1-2*r)*temp[i,j-1] + r*(temp[i+1,j-1] + temp[i-1,j-1])
            temp[0,j] = temp[1,j]
            temp[-1,j] = temp[-2,j]

    else:
        for j in range(1, N):
            temp[1:-1, j] = (1-2*r)*temp[1:-1,j-1] + r*(temp[2:,j-1] + temp[:-2,j-1])

    return x, t, temp

# Get solution using your solver:
x, time, heat = heat_solve() # Create a figure/axes object
fig, axes = plt.subplots(1, 1)
# Create a color map and add a color bar.
map = axes.pcolor(time, x, heat, cmap='seismic')
plt.colorbar(map, ax=axes, label='Temperature ($C$)')



def test():
    solution = np.array([
    [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
    [0.640000, 0.480000, 0.400000, 0.320000, 0.260000, 0.210000, 0.170000, 0.137500, 0.111250, 0.090000, 0.072812],
    [0.960000, 0.800000, 0.640000, 0.520000, 0.420000, 0.340000, 0.275000, 0.222500, 0.180000, 0.145625, 0.117813],
    [0.960000, 0.800000, 0.640000, 0.520000, 0.420000, 0.340000, 0.275000, 0.222500, 0.180000, 0.145625, 0.117813],
    [0.640000, 0.480000, 0.400000, 0.320000, 0.260000, 0.210000, 0.170000, 0.137500, 0.111250, 0.090000, 0.072812],
    [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]])

    # Get solution using your solver:
    x, time, heat = heat_solve()

    # Create a figure/axes object
    fig, axes = plt.subplots(1, 1)
    # Create a color map and add a color bar.
    map = axes.matshow(heat)
    plt.colorbar(map, ax=axes, label='Temperature ($C$)')
    plt.savefig('test.png')

    return heat-solution

def boundaries():
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