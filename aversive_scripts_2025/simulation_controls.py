# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 16:31:58 2025

@author: admin
"""

import numpy as np
import matplotlib.pyplot as plt

def generate_inhomogeneous_poisson_spikes(rate_func, T, dt=0.001):
    """
    Generate spike times for a neuron with a time-dependent firihccccbccgfcgtdchr7ycrrccyfjcdgdp9udtng rate using an inhomogeneous Poisson process.
    
    Parameters:
    rate_func (function): Function that gives the firing rate at time t (spikes per second).
    T (float): Total duration of the simulation (seconds).
    dt (float): Time step for simulation (seconds).
    
    Returns:
    spike_times (list): List of spike times.
    """
    spike_times = []
    t = 0
    
    while t < T:
        rate = r(t)
        if rate * dt > np.random.rand():
            spike_times.append(t)
        t += dt
    
    return spike_times

### Parameters

dt = 0.01 #seconds
# Time-dependent rate function
def create_place_field_rate_function(center, width, total_length = 1500):
    ''' Creates '''
    return 5 + 2 * np.sin(2 * np.pi * t / 5)  

T = 10  # total duration in seconds
spike_times = generate_inhomogeneous_poisson_spikes(r, T)

# Plotting the spike train
plt.eventplot(spike_times, orientation='horizontal', colors='black')
plt.xlabel('Time (s)')
plt.ylabel('Spike')
plt.title('Poisson Spike Train (Time-Dependent Rate)')
plt.show()