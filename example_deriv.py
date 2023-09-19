#!/usr/bin/env python3

'''
This file explores different numerical appoximations for first derivatives
'''

import numpy as np
import matplotlib.pyplot as plt

def fwddiff(fx,dx=1):
    '''
    Take the forward difference first derivative of 'fx'.
    Final element uses a backwards differnce, i.e., same as last element
    '''

    # Create result array, fill with zeros
    dfdx = np.zeros(fx.size)
    dfdx[:-1] = (fx[1:] - fx[:-1]) / dx

    dfdx[-1] = dfdx[-2]

    return dfdx

# Make demo plot
dx = 0.5
x = np.arange(0,4*np.pi,dx)
fx = np.sin(x)
dfdx_sol = np.cos(x)
dfdx = fwddiff(fx,dx=dx)

# Plot
plt.ion()
plt.plot(x, dfdx_sol, label='Analytical Solution')
plt.plot(x, dfdx, label='Numerical Solution')
plt.legend()

dx_all = np.linspace(0.001,1,100)
err = []
for dx in dx_all:
    x = np.arange(0,4*np.pi,dx)
    fx = np.sin(x)
    dfdx_sol = np.cos(x)
    dfdx = fwddiff(fx,dx=dx)

    # Calc error
    err.append(np.max(np.abs(dfdx - dfdx_sol)))

fig,ax = plt.subplots(1,1)
ax.plot(dx_all,err)