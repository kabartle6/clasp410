#!/usr/bin/env python3

'''
Lab 1: Wildfire Model
This file contains tools and scripts for completing Lab 1 for CLaSP 410.
To reproduce the plots shown in the lab report, run forest_model.py. The 
code for each experiment is included in this file. Alternatively, run the 
function creat_model() with the desired kwarg values. 
'''

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

'''
This function takes in a set of parameters definig a model of a forest on fire and
progresses the model through time, plotting the grid at each timestep until there
is no more fire or until the maximum number of timesteps is reached. Returns the 
number of timesteps iterated and the percent of final grid that ends burnt. If the
model is a virus model, the function returns, the number of timesteps iterated, the
percent of final gird that is immune and the percent that is dead.

Parameters
==========
nx : Integer > 0, Number of columns in grid

ny : Integer > 0, Number of rows in grid

nstep :  Integer > 0, Maximum nember of iterations

p_spread : Float [0,1], Probability that fire will spread to adjacent squares

p_bare :  Float [0,1], Probability that a square will start bare

p_start : Float [0,1], Probability that a square will start on fire
        If 0, only the center square will start on fire

p_fatal : Float [0,1], Probability that infection will lead to death for virus model

Returns
=======
Number of timesteps iterated, percent of final grid that ends burnt, 
percent of final grid that ends dead

'''

def create_model(nx=3, ny=3, nstep=5, p_spread=1.0, p_bare=0.0, p_start=0.0, p_fatal=0.0):
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
    plt.ioff()
    forest_cmap = ListedColormap(['black','tan', 'darkgreen', 'crimson'])
    fig, ax = plt.subplots(1,1)
    ax.pcolor(forest[0,:,:], cmap=forest_cmap, vmin=0,vmax=3)
    plt.title('Forest Status (iStep=0)')
    plt.xlabel('X (km)')
    plt.ylabel('Y (km)')
    plt.savefig('Figs/Forest_Status_0.png')

    time = nstep-1 # Number of timesteps

    # Loop through timesteps
    for k in range(1,nstep):
        forest[k,:,:] = forest[k-1,:,:] # Initialize current timestep
        for i in range(ny): # Loop through rows
            for j in range(nx): # Loop through cols
                if forest[k-1,i,j]==3: # If square was previously burning...
                    if j>0:
                        # Burn left
                        if forest[k-1,i,j-1] == 2 and np.random.rand() < p_spread:
                            forest[k,i,j-1] = 3 # Set square to burning if previously forest and random # < p_spread
                    if j<nx-1:
                        # Burn right
                        if forest[k-1,i,j+1] == 2 and np.random.rand() < p_spread:
                            forest[k,i,j+1] = 3
                    if i>0:
                        # Burn down
                        if forest[k-1,i-1,j] == 2  and np.random.rand() < p_spread:
                            forest[k,i-1,j] = 3
                    if i<ny-1:
                        # Burn up
                        if forest[k-1,i+1,j] == 2 and np.random.rand() < p_spread:
                            forest[k,i+1,j] = 3
                    if np.random.rand() < p_fatal:
                        forest[k,i,j] = 0 # Set square that was infected (burning) to dead
                    else:
                        forest[k,i,j] = 1 # Set square that was burning to bare
        # Plot grid
        forest_cmap = ListedColormap(['black','tan', 'darkgreen', 'crimson'])
        fig, ax = plt.subplots(1,1)
        ax.pcolor(forest[k,:,:], cmap=forest_cmap, vmin=0,vmax=3)
        plt.title(f'Forest Status (iStep={k})')
        plt.xlabel('X (km)')
        plt.ylabel('Y (km)')
        plt.savefig(f'Figs/Forest_Status_{k}.png')
        plt.close()

        # Stop when no more squares are burning
        if 3 not in forest[k,:,:]:
            time = k # Update number of timesteps to reflect actual
            break
    percent_bare = (forest[time,:,:]==1).sum()/(nx*ny)*100 # Percent of final grid that is burnt
    percent_dead = (forest[time,:,:]==0).sum()/(nx*ny)*100 # Percent of final grid that is dead
    return time+1, percent_bare, percent_dead


'''
FOREST EXPERIMENT 1
===================
How does the spread of wildfire depend on the probability of spread of fire?
p_bare fixed at 0.0
'''

pspread_list = np.arange(0,1.1,0.1) # Vary probability of spread from 0 to 1
timesteps = []
bare_percents = []
for pspread in pspread_list: # Run model for each p_spread
    time, bare, dead = create_model(nx=100, ny=100, nstep=500, p_spread=pspread, p_start=0.001)
    timesteps.append(time) # Timesteps to no fire for each p_spread
    bare_percents.append(bare) # Percent of final grid bare for each p_spread

# Plot Probability Fire Will Spread vs Timesteps to No Fire
fig, ax = plt.subplots(1,1)
ax.plot(pspread_list,timesteps)
ax.set_title('Probability Fire Will Spread vs Timesteps to No Fire')
ax.set_xlabel('p_spread')
ax.set_ylabel('# of Timesteps')
fig.savefig('Forest_Experiment1_Time.png')

# Plot Probability Fire Will Spread vs Percent Bare of Final Grid
fig, ax = plt.subplots(1,1)
ax.plot(pspread_list,bare_percents)
ax.set_title('Probability Fire Will Spread vs Percent Bare of Final Grid')
ax.set_xlabel('p_spread')
ax.set_ylabel('Percent Bare (%)')
fig.savefig('Forest_Experiment1_Bare.png')
plt.close()

'''
FOREST EXPERIMENT 2
===================
How does the spread of wildfire depend on inital forest density?
p_spread fixed at 1.0
'''

pbare_list = np.arange(0,1.1,0.1) # Vary probability of starting bare from 0 to 1
timesteps = []
bare_percents = []
for pbare in pbare_list: # Run model for each p_bare
    time, bare, dead = create_model(nx=100, ny=100, nstep=500, p_bare=pbare, p_start=0.001)
    timesteps.append(time) # Timesteps to no fire for each p_bare
    bare_percents.append(bare) # Percent of final grid bare for each p_spread

# Plot Probability Square Will Start Bare vs Timesteps to No Fire
fig, ax = plt.subplots(1,1)
ax.plot(pbare_list,timesteps)
ax.set_title('Probability Square Will Start Bare vs Timesteps to No Fire')
ax.set_xlabel('p_bare')
ax.set_ylabel('# of Timesteps')
fig.savefig('Forest_Experiment2_Time.png')

# Plot Probability Square Will Start Bare vs Percent Bare of Final Grid
fig, ax = plt.subplots(1,1)
ax.plot(pbare_list,bare_percents)
ax.set_title('Probability Square Will Start Bare vs Percent Bare of Final Grid')
ax.set_xlabel('p_bare')
ax.set_ylabel('Percent Bare (%)')
fig.savefig('Forest_Experiment2_Bare.png')
plt.close()

'''
DISEASE EXPERIMENT 1
====================
How does disease mortality rate affect disease spread?
p_spread fixed at 1.0
p_immune fixed at 0.0
'''

pfatal_list = np.arange(0,1.1,0.1) # Vary mortality rate from 0 to 1
timesteps = []
immune_percents = []
dead_percents = []
for pfatal in pfatal_list: # Run model for each p_fatal
    time, immune, dead = create_model(nx=100, ny=100, nstep=500, p_start=0.001, p_fatal=pfatal)
    timesteps.append(time) # Timesteps to no disease for each p_fatal
    immune_percents.append(immune) # Percent of final grid immune for each p_fatal
    dead_percents.append(dead) # Percent of final grid dead for each p_fatal

# Plot Probability Disease is Fatal vs Timesteps to No Disease
fig, ax = plt.subplots(1,1)
ax.plot(pfatal_list,timesteps)
ax.set_title('Probability Disease is Fatal vs Timesteps to No Disease')
ax.set_xlabel('p_fatal')
ax.set_ylabel('# of Timesteps')
fig.savefig('Disease_Experiment1_Time.png')

# Plot Probability Disease is Fatal vs Percent Immune of Final Grid
fig, ax = plt.subplots(1,1)
ax.plot(pfatal_list,immune_percents)
ax.set_title('Probability Disease is Fatal vs Percent Immune of Final Grid')
ax.set_xlabel('p_fatal')
ax.set_ylabel('Percent Immune (%)')
fig.savefig('Disease_Experiment1_Immune.png')

# Plot Probability Disease is Fatal vs Percent Dead of Final Grid
fig, ax = plt.subplots(1,1)
ax.plot(pfatal_list,dead_percents)
ax.set_title('Probability Disease is Fatal vs Percent Dead of Final Grid')
ax.set_xlabel('p_fatal')
ax.set_ylabel('Percent Dead (%)')
fig.savefig('Disease_Experiment1_Dead.png')
plt.close()

'''
DISEASE EXPERIMENT 2
====================
How does early vaccine rate affect disease spread?
p_spread fixed at 1.0
p_fatal fixed at 0.5
'''

pimmune_list = np.arange(0,1.1,0.1) # Vary probability of initial immunity from 0 to 1
timesteps = []
immune_percents = []
dead_percents = []
for pimmune in pimmune_list: # Run model for each p_immune (same as p_bare)
    time, immune, dead = create_model(nx=100, ny=100, nstep=500, p_bare=pimmune, p_start=0.001, p_fatal=0.5)
    timesteps.append(time) # Timesteps to no disease for each p_immune
    immune_percents.append(immune) # Percent of final grid immune for each p_immune
    dead_percents.append(dead) # Percent of final grid dead for each p_immune

# Plot Probability Person Starts Immune vs Timesteps to No Disease
fig, ax = plt.subplots(1,1)
ax.plot(pimmune_list,timesteps)
ax.set_title('Probability Person Starts Immune vs Timesteps to No Disease')
ax.set_xlabel('p_immune')
ax.set_ylabel('# of Timesteps')
fig.savefig('Disease_Experiment2_Time.png')

# Plot Probability Person Starts Immune vs Percent Immune of Final Grid
fig, ax = plt.subplots(1,1)
ax.plot(pimmune_list,immune_percents)
ax.set_title('Probability Person Starts Immune vs Percent Immune of Final Grid')
ax.set_xlabel('p_immune')
ax.set_ylabel('Percent Immune (%)')
fig.savefig('Disease_Experiment2_Immune.png')

# Plot Probability Person Starts Immune vs Percent Dead of Final Grid
fig, ax = plt.subplots(1,1)
ax.plot(pimmune_list,dead_percents)
ax.set_title('Probability Person Starts Immune vs Percent Dead of Final Grid')
ax.set_xlabel('p_immune')
ax.set_ylabel('Percent Dead (%)')
fig.savefig('Disease_Experiment2_Dead.png')
plt.close()