"""
@author: ivovollmer
"""
import sys
sys.path.append('PATHTOCODEFUNCTIONS')
import matplotlib.pyplot as plt
import numpy as np
from import_files import *
from functions import *

def compute_a_dot(a, e, u, omega_arg, r, R, S):
    """
    Berechnet a_dot zu einem gegebenen Zeitpunkt.
    Alle Grössen in SI-Einheiten: Meter, Sekunden, Radiant, m/s².
    """
    # G = 6.67430e-11          
    # Me = 5.9722e24           
    # gamma = G * Me   
    gamma = 398600441500000.0        
     
    sqrt_term = np.sqrt(a**3 / (gamma * (1 - e**2)))
    radial_term = e * np.sin(u - omega_arg) * R
    along_track_term = (a * (1 - e**2) / r) * S

    return 2 * sqrt_term * (radial_term + along_track_term)


def rk4_a_propagation(a_n, dt, e_list, u_list, omega_list, r_list, R_list, S_list, n_per_day):
    """
    Führt einen RK4-Schritt über 2*dt durch.
    """
    e_n, e_np1, e_np2 = e_list
    u_n, u_np1, u_np2 = u_list
    omega_n, omega_np1, omega_np2 = omega_list
    r_n, r_np1, r_np2 = r_list
    R_n, R_np1, R_np2 = R_list
    S_n, S_np1, S_np2 = S_list

    k1 = 2 * dt * (86400 / n_per_day) * compute_a_dot(a_n, e_n, u_n, omega_n, r_n, R_n, S_n)
    k2 = 2 * dt * (86400 / n_per_day) * compute_a_dot(a_n + 0.5 * k1, e_np1, u_np1, omega_np1, r_np1, R_np1, S_np1)
    k3 = 2 * dt * (86400 / n_per_day) * compute_a_dot(a_n + 0.5 * k2, e_np1, u_np1, omega_np1, r_np1, R_np1, S_np1)
    k4 = 2 * dt * (86400 / n_per_day) * compute_a_dot(a_n + k3, e_np2, u_np2, omega_np2, r_np2, R_np2, S_np2)

    a_np2 = a_n + (k1 + 2*k2 + 2*k3 + k4) / 6
    return a_np2


def propagate_a_over_time_list(dt, a0, e_array, u_array, omega_array, r_array, R_array, S_array, n_per_day):
    """
    Propagiert a(t) mit RK4 über das gesamte Array in 2*dt-Schritten.
    """
    N = len(R_array)

    a_result = [a0]

    for i in range(0, N - 2*dt, 2*dt):
        
        e_list     = np.array([e_array[i], e_array[i+dt], e_array[i+2*dt]])
        u_list     = np.array([u_array[i], u_array[i+dt], u_array[i+2*dt]])
        omega_list = np.array([omega_array[i], omega_array[i+dt], omega_array[i+2*dt]])
        r_list     = np.array([r_array[i], r_array[i+dt], r_array[i+2*dt]])
        R_list     = np.array([R_array[i], R_array[i+dt], R_array[i+2*dt]])
        S_list     = np.array([S_array[i], S_array[i+dt], S_array[i+2*dt]])

        a_n = a_result[-1]
        a_np2 = rk4_a_propagation(a_n, dt, e_list, u_list, omega_list, r_list, R_list, S_list, n_per_day)
        a_result.append(a_np2)

    return a_result


def int_daily(a_array, e_array, u_array, omega_array, r_array, R_array, S_array, n_per_day, n_days, dt):
    list_out = []
    for i in range(n_days):
        a0_current = a_array[i*n_per_day]
        e_current = e_array[i*n_per_day : i*n_per_day + n_per_day]
        u_current = u_array[i*n_per_day : i*n_per_day + n_per_day]
        omega_current = omega_array[i*n_per_day : i*n_per_day + n_per_day]
        r_current = r_array[i*n_per_day : i*n_per_day + n_per_day]
        R_current = R_array[i*n_per_day : i*n_per_day + n_per_day]
        S_current = S_array[i*n_per_day : i*n_per_day + n_per_day]
        
        a_current = propagate_a_over_time_list(dt, a0_current, e_current, u_current, omega_current, r_current, R_current, S_current, n_per_day)
        list_out.extend(a_current)
     
    return np.array(list_out)

