#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def sample_init(x):
    '''Simple inital boundary conition function'''
    return 4*x - 4*x**2

def heat_solve(xmax=1.0, dx=0.2, tmax=0.2, dt=0.2, c2=1.0, init=sample_init, top=0, bottom=0):
    '''
    dt, dx : float, default
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
    temps : numpy 
    '''

    # set consatnts
    r = c2 * dt/dx**2

    # CReatet space and time grids
    x = np.arange(0,xmax+dx, dx)
    t = np.arange(0,tmax+dt, dt)
    # Save number of points
    M,N = x.size(), t.size()

    # Create temperature solution array
    temp = np.zeros(x, t)

    # Set boundary conditions
    if callable(top):
        temp[0,:] = top(t)
    else:
        temp[0,:] = top
    if callable(bottom):
        temp[-1,:] = bottom(t)
    else:
        temp[-1,:] = bottom

    # Set initial condion
    if callable(init):
        temp[:,0] = init(x)
    else:
        temp[:,0] = init

    # Solve!
    for j in range(1, N):
        temp[1:-1, j] = (1-2*r)*temp[1:-1,j-1] + r*(temp[2:,j-1] + temp[:-2,j-1])

    return x, t, temp