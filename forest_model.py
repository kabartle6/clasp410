#!/usr/bin/env python3

'''
Lab 1: Wildfire Model
This file contains tools and scripts for completing Lab 1 for CLaSP 410.
To reproduce the plots shown in the lab report, run the function creat_model()
with the desired kwarg values.
'''

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

'''
This function takes in a set of parameters definig a model of a forest on fire and
progresses the model through time, plotting the grid at each timestep until there
is no more fire or until the maximum number of timesteps is reached. Returns the 
number of timesteps iterated and the percent of final grid that ends burnt.

Parameters
==========
nx : Integer > 0, Number of columns in grid

ny : Integer > 0, Number of rows in grid

nstep :  Integer > 0, Maximum nember of iterations

p_spread : Float [0,1], Probability that fire will spread to adjacent squares

p_bare :  Float [0,1], Probability that a square will start bare

p_start : Float [0,1], Probability that a square will start on fire
        If 0, only the center square will start on fire

Returns
=======
Number of timesteps iterated, percent of final grid that ends burnt

'''

def create_model(nx=3, ny=3, nstep=5, p_spread=1.0, p_bare=0.0, p_start=0.0):
    # Define forest grid
    forest = np.zeros((nstep,ny,nx), dtype=int) + 2

    # Set bare squares based on p_bare
    isbare = np.random.rand(ny,nx)
    isbare = isbare < p_bare
    forest[0,isbare] = 1

    if p_start == 0:
        forest[0,ny//2,nx//2] = 3 # Set center square to burning
    else:
        # Set burning squares based on p_start
        fire = np.random.rand(ny,nx)
        fire = fire < p_start
        forest[0,fire] = 3

    # Plot initial grid
    plt.ion()
    forest_cmap = ListedColormap(['tan', 'darkgreen', 'crimson'])
    fig, ax = plt.subplots(1,1)
    ax.pcolor(forest[0,:,:], cmap=forest_cmap, vmin=1,vmax=3)
    plt.title('Forest Status (iStep=0)')
    plt.xlabel('X (km)')
    plt.ylabel('Y (km)')

    time = nstep-1 # Number of timesteps

    # Loop through timesteps
    for k in range(1,nstep):
        forest[k,:,:] = forest[k-1,:,:] # Initialize current timestep
        for i in range(ny): # Loop through rows
            for j in range(nx): # Loop through cols
                if forest[k-1,j,i]==3: # If square was previously burning...
                    if i>0:
                        # Burn left
                        if forest[k-1,j,i-1] == 2 and np.random.rand() < p_spread:
                            forest[k,j,i-1] = 3 # Set square to burning if previously forest and random # < p_spread
                    if i<nx-1:
                        # Burn right
                        if forest[k-1,j,i+1] == 2 and np.random.rand() < p_spread:
                            forest[k,j,i+1] = 3
                    if j>0:
                        # Burn down
                        if forest[k-1,j-1,i] == 2  and np.random.rand() < p_spread:
                            forest[k,j-1,i] = 3
                    if j<ny-1:
                        # Burn up
                        if forest[k-1,j+1,i] == 2 and np.random.rand() < p_spread:
                            forest[k,j+1,i] = 3
                    forest[k,j,i] = 1 # Set square that was burning to bare
        # Plot grid
        forest_cmap = ListedColormap(['tan', 'darkgreen', 'crimson'])
        fig, ax = plt.subplots(1,1)
        ax.pcolor(forest[k,:,:], cmap=forest_cmap, vmin=1,vmax=3)
        plt.title(f'Forest Status (iStep={k})')
        plt.xlabel('X (km)')
        plt.ylabel('Y (km)')

        # Stop when no more squares are burning
        if 3 not in forest[k,:,:]:
            time = k # Update number of timesteps to reflect actual
            break
    percent_bare = (forest[time,:,:]==1).sum()/(nx*ny)*100 # Percent of final grid that is burnt
    return time+1, percent_bare



