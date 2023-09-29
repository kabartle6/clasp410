#!/usr/bin/env python3

'''
Lab 2: Population Control

The Lotka-Volterra competition and predator/prey coupled ODEs are solved 
using Euler's method (euler_solve()) and using RK8 (solve_rk8()). 
verification() is used to create the plots for question 1, competition() 
is used to create the plots for question 2, and predprey() is used to create 
the plots for question 3.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def dNdt_comp(t, N, a=1,b=2,c=1,d=3):
    '''
    This function calculates the Lotka-Volterra competition equations for
    two species. Given normalized populations, `N1` and `N2`, as well as the
    four coefficients representing population growth and decline,
    calculate the time derivatives dN_1/dt and dN_2/dt and return to the
    caller.
    This function accepts `t`, or time, as an input parameter to be
    compliant with Scipy's ODE solver. However, it is not used in this
    function.
    Parameters
    ----------
    t : float
        The current time (not used here).
    N : two-element list
        The current value of N1 and N2 as a list (e.g., [N1, N2]).
    a, b, c, d : float, defaults=1, 2, 1, 3
        The value of the Lotka-Volterra coefficients.
    Returns
    -------
    dN1dt, dN2dt : floats
        The time derivatives of `N1` and `N2`.
    '''
    # Here, N is a two-element list such that N1=N[0] and N2=N[1]
    dN1 = a*N[0]*(1 - N[0]) - b*N[0]*N[1]
    dN2 = c*N[1]*(1 - N[1]) - d*N[0]*N[1]
    return dN1, dN2

def dNdt_pred(t, N, a=1,b=2,c=1,d=3):
    '''
    This function calculates the Lotka-Volterra predator/prey equations for
    two species. Given a normalized "prey" population, `N1` and "predator" 
    population, `N2`, as well as the four coefficients representing 
    population growth and decline, calculate the time derivatives dN_1/dt 
    and dN_2/dt and return to the caller.
    This function accepts `t`, or time, as an input parameter to be
    compliant with Scipy's ODE solver. However, it is not used in this
    function.
    Parameters
    ----------
    t : float
        The current time (not used here).
    N : two-element list
        The current value of N1 and N2 as a list (e.g., [N1, N2]).
    a, b, c, d : float, defaults=1, 2, 1, 3
        The value of the Lotka-Volterra coefficients.
    Returns
    -------
    dN1dt, dN2dt : floats
        The time derivatives of `N1` and `N2`.
    '''
    # Here, N is a two-element list such that N1=N[0] and N2=N[1]
    dN1 = a*N[0]-b*N[0]*N[1]
    dN2 = -c*N[1]+d*N[0]*N[1]
    return dN1, dN2

def euler_solve(func, N1_init=.5, N2_init=.5, dT=.1, t_final=100.0,
                a=1, b=2, c=1, d=3): 
    '''
    Solve the Lotka-Volterra competition and predator/prey equations using
    the Euler method.
    Parameters
    ----------
    func : function
        A python function that takes `time`, [`N1`, `N2`] as inputs and
        returns the time derivative of N1 and N2.
    N1_init, N2_init : float
        Initial conditions for `N1` and `N2`, ranging from (0,1].
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
    # Create time array from 0 to t_final with stepsize dT
    time = np.arange(0,t_final,dT)
    # Initialize array for N1 population
    N1 = np.zeros(len(time)) 
    N1[0] = N1_init
    # Initialize array for N2 population
    N2 = np.zeros(len(time)) 
    N2[0] = N2_init
    for i in range(1, time.size):
        dN1, dN2 = func(i, [N1[i-1], N2[i-1]], a=a,b=b,c=c,d=d) # Find derivatives
        # Euler
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
    

def verification(dT_comp=1, dT_pred=0.05):
    '''
    Create plots that answer question 1 of the lab by using the Euler and RK8 
    models to solve and plot the Lotka-Voltera competitiion and predator prey 
    models with initial conditions a = 1, b = 2, c = 1, d = 3, N1 = 0.3, 
    N2 = 0.6, T_final = 100 years and the givin dT.
    Parameters
    ----------
    dT_comp : float, default = 1
        Largest timestep allowed in years for the competition model.
    dT_pred : float, default = 0.05
        Largest timestep allowed in years for the predator prey model.
    '''
    # Solve for the two competing population sizes over time using Euler
    time, N1, N2 = euler_solve(dNdt_comp, dT=dT_comp, N1_init=0.3, N2_init=0.6)
    # Plot
    fig,ax = plt.subplots(1,1)
    ax.plot(time, N1, label='N1 Euler', color='orange')
    ax.plot(time, N2, label='N2 Euler', color='blue')
    # Solve for the two competing population sizes over time using RK8
    time, N1, N2 = solve_rk8(dNdt_comp, dT=dT_comp, N1_init=0.3, N2_init=0.6)
    ax.plot(time, N1, label='N1 RK8', color='orange', ls='--')
    ax.plot(time, N2, label='N2 RK8', color='blue', ls='--')
    plt.xlabel('Time')
    plt.ylabel('Population Size')
    plt.title('Lotka-Voltera Competition Model\na=1, b=2, c=1, d=3')
    plt.legend()
    fig.savefig(f'Competition{dT_comp}.png') # Save figure

    # Solve for the predator and prey population sizes over time using Euler
    time, N1, N2 = euler_solve(dNdt_pred, dT=dT_pred, N1_init=0.3, N2_init=0.6)
    # Plot
    fig,ax = plt.subplots(1,1)
    ax.plot(time, N1, label='N1 (Prey) Euler', color='orange')
    ax.plot(time, N2, label='N2 (Predator) Euler', color='blue')
    # Solve for the predator and prey population sizes over time using RK8
    time, N1, N2 = solve_rk8(dNdt_pred, dT=dT_pred, N1_init=0.3, N2_init=0.6)
    ax.plot(time, N1, label='N1 (Prey) RK8', color='orange', ls='--')
    ax.plot(time, N2, label='N2 (Predator) RK8', color='blue', ls='--')
    plt.xlabel('Time (years)')
    plt.ylabel('Population/Varrying Cap.')
    plt.title('Lotka-Voltera Predator Prey Model\na=1, b=2, c=1, d=3')
    plt.legend()
    fig.savefig(f'PredPrey{dT_pred}.png') # Save figure
    return

def competition(N1_init=0.3,N2_init=0.6,a=1,b=2,c=1,d=3):
    '''
    Create plots that answer question 2 of the lab by using the Euler and RK8 
    models to solve and plot the Lotka-Voltera competitiion model with the
    initial conditions passed in. T_finial is 100 years and dT is 1.
    Parameters
    ----------
    N1_init, N2_init : float, default = 0.3,0.6
        Initial conditions for `N1` and `N2`, ranging from (0,1].
    a, b, c, d : float, default=1, 2, 1, 3
        Lotka-Volterra coefficient values
    '''
    # Solve for the two competing population sizes over time using Euler
    time, N1, N2 = euler_solve(dNdt_comp, dT=1, N1_init=N1_init, N2_init=N2_init,
                               a=a,b=b,c=c,d=d)
    # Plot
    fig,ax = plt.subplots(1,1)
    ax.plot(time, N1, label='N1 Euler', color='orange')
    ax.plot(time, N2, label='N2 Euler', color='blue')
    # Solve for the two competing population sizes over time using RK8
    time, N1, N2 = solve_rk8(dNdt_comp, dT=1, N1_init=N1_init, N2_init=N2_init,
                             a=a,b=b,c=c,d=d)
    ax.plot(time, N1, label='N1 RK8', color='orange', ls='--')
    ax.plot(time, N2, label='N2 RK8', color='blue', ls='--')
    plt.xlabel('Time')
    plt.ylabel('Population Size')
    plt.title(f'Lotka-Voltera Competition Model\na={a}, b={b}, c={c}, d={d}')
    plt.legend()
    fig.savefig(f'Competition{a}{b}{c}{d}_{N1_init},{N2_init}.png') # Save figure
    return

def predprey(N1_init=0.3,N2_init=0.6,a=1,b=2,c=1,d=3):
    '''
    Create plots for both time vs population size and prey population vs 
    predator population to answer question 3 of the lab by using the Euler and RK8 
    models to solve and plot the Lotka-Voltera predator prey model with the
    initial conditions passed in. T_finial is 100 years and dT is 0.005.
    Parameters
    ----------
    N1_init, N2_init : float, default = 0.3, 0.6
        Initial conditions for `N1` and `N2`, ranging from (0,1].
    a, b, c, d : float, default=1, 2, 1, 3
        Lotka-Volterra coefficient values
    '''
    # Solve for the two competing population sizes over time using Euler
    time, N1_euler, N2_euler = euler_solve(dNdt_pred, dT=0.005, N1_init=N1_init, N2_init=N2_init,
                               a=a,b=b,c=c,d=d)
    # Solve for the two competing population sizes over time using RK8
    time2, N1_rk8, N2_rk8 = solve_rk8(dNdt_pred, dT=0.05, N1_init=N1_init, N2_init=N2_init,
                             a=a,b=b,c=c,d=d)
    # Plot population size over time
    fig,ax = plt.subplots(1,1)
    ax.plot(time, N1_euler, label='N1 (Prey) Euler', color='orange')
    ax.plot(time, N2_euler, label='N2 (Predator) Euler', color='blue')
    ax.plot(time2, N1_rk8, label='N1 (Prey) RK8', color='orange', ls='--')
    ax.plot(time2, N2_rk8, label='N2 (Predator) RK8', color='blue', ls='--')
    plt.xlabel('Time')
    plt.ylabel('Population Size')
    plt.title(f'Lotka-Voltera Predator Prey Model\na={a}, b={b}, c={c}, d={d}')
    plt.legend()
    fig.savefig(f'Predprey{a}{b}{c}{d}_{N1_init},{N2_init}.png') # Save figure
    # Plot population phase diagram
    fig,ax = plt.subplots(1,1)
    ax.plot(N1_euler, N2_euler, label='Euler', color='orange')
    ax.plot(N1_rk8, N2_rk8, label='RK8', color='blue', ls='--')
    plt.xlabel('N1 (Prey) Population')
    plt.ylabel('N2 (Predator) Population')
    plt.title(f'Lotka-Voltera Predator Prey Phase Diagram\na={a}, b={b}, c={c}, d={d}')
    plt.legend()
    fig.savefig(f'PhaseDiagram{a}{b}{c}{d}_{N1_init},{N2_init}.png') # Save figure
    return
 