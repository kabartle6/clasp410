#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def euler(T_init, T_env=25, dt=60,  k=1/300, t_start=0, t_stop=600):
    time = np.arange(t_start,t_stop,dt)
    Temps = np.zeros(len(time))
    Temps[0] = T_init
    for i in range(1,len(time)):
        Temps[i] = Temps[i-1] + dt*(-k*(Temps[i-1]-T_env))
        
    return time, Temps

Tinit = 90
time, temps = euler(Tinit)
plt.ion()
plt.plot(time, temps)

