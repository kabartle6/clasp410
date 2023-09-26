#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def dNdt_comp(t, N, a=1,b=2,c=1,d=3):
    dN1 = a*N[0]*(1 - N[0]) - b*N[0]*N[1]
    dN2 = c*N[1]*(1 - N[1]) - d*N[0]*N[1]
    return dN1, dN2

def dNdt_pred(t, N, a=1,b=2,c=1,d=3):
    dN1 = a*N[0]-b*N[0]*N[1]
    dN2 = -c*N[1]+d*N[0]*N[1]
    return dN1, dN2

def euler_solve(func, N1_init=.5, N2_init=.5, dT=.1, t_final=100.0): 
    '''
    <Your good docstring here>
    Parameters
    ----------
    func : function
        A python function that takes `time`, [`N1`, `N2`] as inputs and
        returns the time derivative of N1 and N2.
    N1_init : float
         <more good docstring here>
    '''
    time = np.arange(0,t_final,dT)
    N1 = np.zeros(len(time))
    N2 = np.zeros(len(time))
    N1[0] = N1_init
    N2[0] = N2_init
    for i in range(1, time.size):
        dN1, dN2 = func(i, [N1[i-1], N2[i-1]])
        N1[i] = N1[i-1] + dT*dN1
        N2[i] = N2[i-1] + dT*dN2
    return time, N1, N2

def solve_rk8(func, N1_init=.5, N2_init=.5, dT=10, t_final=100.0,
              a=1, b=2, c=1, d=3):
    '''
    Solve the Lotka-Volterra competition and predator/prey equations using
    Scipy's ODE class and the adaptive step 8th order solver.
    Parameters
    ----------
    func : function
        A python function that takes `time`, [`N1`, `N2`] as inputs and
        returns the time derivative of N1 and N2.
    N1_init, N2_init : float
        Initial conditions for `N1` and `N2`, ranging from (0,1]
    dT : float, default=10
        Largest timestep allowed in years.
    t_final : float, default=100
        Integrate until this value is reached, in years.
    a, b, c, d : float, default=1, 2, 1, 3
        Lotka-Volterra coefficient values
    Returns
    -------
    time : Numpy array
        Time elapsed in years.
    N1, N2 : Numpy arrays
        Normalized population density solutions.
    '''
    
    # Configure the initial value problem solver
    result = solve_ivp(func, [0, t_final], [N1_init, N2_init],
                       args=[a, b, c, d], method='DOP853', max_step=dT)
    
    # Perform the integration
    time, N1, N2 = result.t, result.y[0, :], result.y[1, :] # Return values to caller.
    return time, N1, N2
    

time, N1, N2 = euler_solve(dNdt_comp)
plt.ion()
fig,ax = plt.subplots(1,1)
ax.plot(time, N1, label='N1')
ax.plot(time, N2, label='N2')
plt.xlabel('Time')
plt.ylabel('Population Size')
plt.title('Population Size of Competing Species')
plt.legend()

time, N1, N2 = solve_rk8(dNdt_pred)
fig,ax = plt.subplots(1,1)
ax.plot(time, N1, label='Bunnies')
ax.plot(time, N2, label='Foxes')
plt.xlabel('Time')
plt.ylabel('Population Size')
plt.title('Population Size of Foxes and Bunnies')
plt.legend()
