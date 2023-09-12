#!/usr/bin/env python3

'''
Lab 1: Wildfire Model
'''

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Define variables
nx = 3
ny = 3
nstep = 5
p_burn = 1.0
p_bare = 0.5

# Define forest grid
forest = np.zeros((nstep,ny,nx), dtype=int) + 2

# Set bare squares based on p_bare
isbare = np.random.rand(ny,nx)
isbare = isbare < p_bare
forest[0,isbare] = 1

forest[0,ny//2,nx//2] = 3 # Set center square to burning

# Plot initial grid
forest_cmap = ListedColormap(['tan', 'darkgreen', 'crimson'])
fig, ax = plt.subplots(1,1)
ax.pcolor(forest[0,:,:], cmap=forest_cmap, vmin=1,vmax=3)
plt.title('Forest Status (iStep=0)')
plt.xlabel('X (km)')
plt.ylabel('Y (km)')

# Loop through timesteps
for k in range(1,nstep):
    forest[k,:,:] = forest[k-1,:,:]
    for i in range(nx):
        for j in range(ny):
            if forest[k-1,j,i]==3:
                if i>0:
                    # Burn left
                    if forest[k-1,j,i-1] == 2 and np.random.rand() < p_burn:
                        forest[k,j,i-1] = 3
                if i<nx-1:
                    # Burn right
                    if forest[k-1,j,i+1] == 2 and np.random.rand() < p_burn:
                        forest[k,j,i+1] = 3
                if j>0:
                    # Burn up
                    if forest[k-1,j-1,i] == 2  and np.random.rand() < p_burn:
                        forest[k,j-1,i] = 3
                if j<ny-1:
                    # Burn down
                    if forest[k-1,j+1,i] == 2 and np.random.rand() < p_burn:
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
        break



