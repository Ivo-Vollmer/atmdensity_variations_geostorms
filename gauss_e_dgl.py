"""
Code adapted from Levin Walter
(@author: ivovollmer)
"""
import sys
sys.path.append('PATHTOCODEFUNCTIONS')
import numpy as np
import math
import matplotlib.pyplot as plt
from astropy.time import Time
import os
from import_files import load_all_data
from functions import mjd_to_mmdd

day_sec = 24 * 60 * 60
sec_to_day = 1 / day_sec
G  = 6.674e-11 / (sec_to_day**2)
Me = 5.972e24
# μ = G * Me
μ = 398600441500000*86400**2

def convert_str_double_to_float(string):
    """ convert double-string to float
    string = '0.429874776D-08'
    where D means it is a number with double precision and it replaces the exponential
    """
    if (string[0] == ' '):
        new_string = string[1:]
    else:
        new_string = string
    return(float(new_string.replace("D", "E")))

def ele_file_read(foldername, filename):
    """ extract PCAs from ELE file
    foldername: string-like
    filename: string-like
    """
    file = open(foldername + '/' + filename, 'r')
    lines = file.readlines()
    file.close()
    
    c = 0
    while (c < 101):
        row = lines[c].strip()
        if (row[:2] == '11'):
            break
        if (c == 100):
            print("!!! ERROR 1 !!!")
            break
        c += 1
    
    length = int((len(lines) - c) / 3)
    
    array_MJD = np.array([])
    array_11 = np.array([])
    array_12 = np.array([])
    array_13 = np.array([])
    
    for i in range(0, length):
        j = c + 3 * i
        
        MJD_i = float(lines[j].strip()[9 : 20])
        array_MJD = np.append(array_MJD, MJD_i)
        
        acc_11_i_string = lines[j + 0].strip()[37 : 54]
        acc_11_i = convert_str_double_to_float(acc_11_i_string)
        array_11 = np.append(array_11, acc_11_i)
        
        acc_12_i_string = lines[j + 1].strip()[37 : 54]
        acc_12_i = convert_str_double_to_float(acc_12_i_string)
        array_12 = np.append(array_12, acc_12_i)
        
        acc_13_i_string = lines[j + 2].strip()[37 : 54]
        acc_13_i = convert_str_double_to_float(acc_13_i_string)
        array_13 = np.append(array_13, acc_13_i)
    
    c0 = 0
    while (c0 < 101):
        row = lines[c0].strip()
        if (row[:3] == 'L30'):
            L30_string = lines[c0].strip()[37 : 54]
            L30 = convert_str_double_to_float(L30_string)
            L20_string = lines[c0 + 1].strip()[37 : 54]
            L20 = convert_str_double_to_float(L20_string)
            L10_string = lines[c0 + 2].strip()[37 : 54]
            L10 = convert_str_double_to_float(L10_string)
            
            array_11 += L30 #geändert von L10 zu L30 
            array_12 += L20
            array_13 += L10 #geändert von L30 zu L10
            break
        if (c0 == 100):
            print("!!! ERROR 2 !!!")
            break
        c0 += 1
    
    data = np.vstack((array_MJD, array_11, array_12, array_13)).T
    
    array_14 = np.array([]) # absolute acceleration
    for i in range(0, len(data)):
        abs = 0
        for j in range(1, 4):
            abs += (data[i, j])**2
        array_14 = np.append(array_14, np.sqrt(abs))
    data = np.hstack((data, np.array([array_14]).T))
    
    return(data)

def pca_gentxt(foldername, pathsave):
    """ generate .txt file with PCAs
    foldername: string-like
    pathsave: string-like
    """
    entries_old = os.listdir(str(foldername))
    entries = np.array([])
    for i in range(0, len(entries_old)):
        file_i = entries_old[i]
        # get rid of '.DS_Store' and txt files
        if (file_i[-4:] == '.ELE'):
            entries = np.append(entries, file_i)
    entries.sort()
    
    entries_new = np.array([])
    for i in range(0, len(entries)):
        file_i = entries[i]
        file_size = os.path.getsize(foldername + '/' + file_i)
        if (file_size > 1000): # avoid empty files
            entries_new = np.append(entries_new, file_i)
    
    data_year = np.array([[0, 0, 0, 0, 0]])
    for i in range(0, len(entries_new)):
        file_i = entries_new[i]
        data_i = ele_file_read(foldername, file_i)
        data_year = np.vstack((data_year, data_i))
    data_year = data_year[1:]
    
    name_year = pathsave + "PCA.txt"
    np.savetxt(name_year, data_year, delimiter = ' ')


def ele_get_osc(foldername, filename):
    """ extract day values of osculating elements from ELE file
    get values of A, E, I, NODE, PERIGEE, ARG OF LAT
    foldername: string-like
    filename: string-like
    """
    file = open(foldername + '/' + filename, 'r')
    lines = file.readlines()
    file.close()
    
    c = 0
    while (c < 101):
        row = lines[c].strip()
        if (row[:10] == 'ARC-NUMBER'):
            break
        if (c == 100):
            print("!!! ERROR 1 !!!")
            break
        c += 1
    
    mjd = float(row[-18:])
    a_day = float(lines[c + 2][37 : -27].strip())
    e_day = float(lines[c + 3][37 : -27].strip())
    i_day = float(lines[c + 4][37 : -27].strip())
    Ω_day = float(lines[c + 5][37 : -27].strip())
    ω_day = float(lines[c + 6][37 : -27].strip())
    u_day = float(lines[c + 7][37 : -27].strip())
    
    data = np.array([mjd, a_day, e_day, i_day,
                     Ω_day, ω_day, u_day])
    
    return(data)

def ele_gen_txt_osc(foldername, pathsave):
    """ generate .txt file with daily values of osculating elements
    foldername: string-like
    pathsave: string-like
    """
    entries_old = os.listdir(str(foldername))
    
    entries = np.array([])
    for i in range(0, len(entries_old)):
        file_i = entries_old[i]
        # get rid of '.DS_Store' and .txt files
        if (file_i[-4:] == '.ELE'):
            entries = np.append(entries, file_i)
    entries.sort()
    
    entries_new = np.array([])
    for i in range(0, len(entries)):
        file_i = entries[i]
        file_size = os.path.getsize(foldername + '/' + file_i)
        if (file_size > 10000): # avoid empty files
            entries_new = np.append(entries_new, file_i)
    
    data_year = np.array([[0, 0, 0, 0, 0, 0, 0]]) # mjd, a, e, i, Ω, ω, u
    for i in range(0, len(entries_new)):
        file_i = entries_new[i]
        data_i = ele_get_osc(foldername, file_i)
        data_year = np.vstack((data_year, data_i))
    data_year = data_year[1:]
    
    name_year = pathsave + "ele_osc.txt"
    np.savetxt(name_year, data_year, delimiter = ' ')

def master_ele(foldername, pathsave):
    """ creates master ELE files
    Input:
        foldername: string-like
        file_type: string-like
    Output:
        creates master ele file and master osc ele file
    """
    pca_gentxt(foldername, pathsave)
    ele_gen_txt_osc(foldername, pathsave)

def array_modifier(array, MJD_start, n_days):
    """ trim array
    array = [[t1, x1, y1, ...], [t2, x2, y2, ...], ...]
    get interval [MJD_start, MJD_start + n_days)
    The pair at MJD_start + n_days is not included
    MJD_start and n_days do not have to be integers
    """
    a = 0 # find start
    while (array[a + 1, 0] <= MJD_start):
        a += 1
    array = array[a :] # crop
    b = 0 # find end
    if (array[-1, 0] <  MJD_start + n_days):
        return(array)
    else:
        while (array[b, 0] < MJD_start + n_days):
            b += 1
        array = array[: b] # crop
        return(array)

def eccentric_anomaly(v, e):
    """ Berechnet exzentrische Anomalie E aus wahrer Anomalie v (rad) und Exzentrizität e """
    E = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(v / 2))
    if E < 0:
        E += 2 * np.pi
    return E

def a_e_dot_RSW_coupled(a, e, ω, r, u, R, S):
    """ Gauss's perturbation equation for the semi-major axis
    Input:
        osculating elements and RSW accelerations (only R and S)
    Output:
        slope of semi-major axis
    """
    ν = (u - ω) * np.pi / 180
    fac1 = 2 * math.sqrt(a**3 / (μ * (1 - e*e)))
    term1 = e * math.sin(ν) * R
    # term2 = (1 + e * math.cos(ν)) * S
    term2 = (a*(1-e**2)/r)*S
    
    a_dot = fac1 * (term1 + term2)
    #-------
    fac2 = math.sqrt(a*(1-e**2)/μ) 
    E = eccentric_anomaly(ν, e)    
    term3 = math.sin(ν) * R
    term4 = (math.cos(ν)+math.cos(E)) * S
    
    e_dot = fac2*(term3+term4)
    
    return(a_dot, e_dot)

def propagator_coupled(dt, a_n, u_list, R_list, S_list, e_n, omega_list, r_list):
    """ propagates one step = 2 * dt or from a_n to a_np2, np2 = n + 2
    Input:
        dt: sampling
        a_n: semi-major axis at time t_n
        u_list = [u_n, u_(n+1), u_(n+2)]: argument of latitude at times t_n, t_(n+1) and t_(n+2)
        R_list = [R_n, R_(n+1), R_(n+2)]: radial acceleration at times t_n, t_(n+1) and t_(n+2)
        S_list = [S_n, S_(n+1), S_(n+2)]: along-track acceleration at times t_n, t_(n+1) and t_(n+2)
        e: eccentricity (constant)
        ω: argument of perigee (constant)
    Output:
        a_np2: semi-major axis at time t_(n+2)
    """
    u_n, u_np1, u_np2 = u_list[0], u_list[1], u_list[2] # u(t, t + dt, t + 2dt)
    omega_n, omega_np1, omega_np2 = omega_list[0], omega_list[1], omega_list[2] # omega(t, t + dt, t + 2dt)
    r_n, r_np1, r_np2 = r_list[0], r_list[1], r_list[2]
    R_n, R_np1, R_np2 = R_list[0], R_list[1], R_list[2] # R(t, t + dt, t + 2dt)
    S_n, S_np1, S_np2 = S_list[0], S_list[1], S_list[2] # S(t, t + dt, t + 2dt)
    
    k1, k5 = a_e_dot_RSW_coupled(a_n, e_n, omega_n, r_n, u_n, R_n, S_n)
    k1 = 2 * dt * k1
    k5 = 2 * dt * k5
    k2, k6 = a_e_dot_RSW_coupled(a_n + k1 / 2, e_n + k5 / 2, omega_np1, r_np1, u_np1, R_np1, S_np1)
    k2 = 2 * dt * k2
    k6 = 2 * dt * k6
    k3, k7 = a_e_dot_RSW_coupled(a_n + k2 / 2, e_n + k6 / 2, omega_np1, r_np1, u_np1, R_np1, S_np1)
    k3 = 2 * dt * k3
    k7 = 2 * dt * k7
    k4, k8 = a_e_dot_RSW_coupled(a_n + k3, e_n + k7, omega_np2, r_np2, u_np2, R_np2, S_np2)
    k4 = 2 * dt * k4
    k8 = 2 * dt * k8
    
    a_np2 = a_n + (k1 + k4) / 6 + (k2 + k3) / 3
    e_np2 = e_n + (k5 + k8) / 6 + (k6 + k7) / 3
    return(a_np2, e_np2)

def integrator_coupled(a_0, e_0, ω, r, u_data, acc_R_data, acc_S_data):
    """ integrates time interval with Runge-Kutta fourth order method
    Input:
        a_data = [[t_1, a_1], [t_2, a_2], ...]
        e_data = [[t_1, e_1], [t_2, e_2], ...]
        ω_data = [[t_1, ω_1], [t_2, ω_2], ...]
        u_data = [[t_1, u_1], [t_2, u_2], ...]
        acc_R_data = [[t_1, R_1], [t_2, R_2], ...]
        acc_S_data = [[t_1, S_1], [t_2, S_2], ...]
    Output:
        a_int_data = [[t_1, a_1], [t_2, a_2], ...]
            integrated semi-major axis
        a_dot_data = [[t_1, a_dot_1], [t_2, a_dot_2], ...]
            slope of semi-major axis
    """
    u_list_list = u_data[:, 1]
    omega_list_list = ω
    r_list_list = r
    R_list_list = acc_R_data[:, 1] * day_sec * day_sec
    S_list_list = acc_S_data[:, 1] * day_sec * day_sec
    
    # starting point
    t_0 = u_data[0, 0] 
    a_dot_0, e_dot_0 = a_e_dot_RSW_coupled(a_0, e_0, omega_list_list[0], r_list_list[0], u_list_list[0], R_list_list[0], S_list_list[0])
    
    a_int_data = np.array([[t_0, a_0]])
    a_dot_data = np.array([[t_0, a_dot_0]])
    e_int_data = np.array([[t_0, e_0]])
    e_dot_data = np.array([[t_0, e_dot_0]])
    
    dt = u_data[1, 0] - u_data[0, 0] # time step
    for i in range(0, (len(u_data) - 1)// 2):
        # _np2 for_(n+2)
        n = 2 * i
        t_np2 = u_data[n + 2, 0]
        a_n = a_int_data[-1, 1]
        e_n = e_int_data[-1, 1]
        
        u_list = u_list_list[n : n + 2 + 1]
        omega_list = omega_list_list[n : n + 2 + 1]
        r_list = r_list_list[n : n + 2 + 1]
        R_list = R_list_list[n : n + 2 + 1]
        S_list = S_list_list[n : n + 2 + 1]
        
        a_np2, e_np2 = propagator_coupled(dt, a_n, u_list, R_list, S_list, e_n, omega_list, r_list)
        
        a_dot_np2, e_dot_np2 = a_e_dot_RSW_coupled(a_np2, e_np2, omega_list[-1], r_list[-1], u_list[-1], R_list[-1], S_list[-1])
        
        a_int_row = np.array([t_np2, a_np2])
        a_dot_row = np.array([t_np2, a_dot_np2])
        e_int_row = np.array([t_np2, e_np2])
        e_dot_row = np.array([t_np2, e_dot_np2])
        
        a_int_data = np.vstack((a_int_data, a_int_row))
        a_dot_data = np.vstack((a_dot_data, a_dot_row))
        e_int_data = np.vstack((e_int_data, e_int_row))
        e_dot_data = np.vstack((e_dot_data, e_dot_row))
    
    return(a_int_data, a_dot_data, e_int_data, e_dot_data)

def step_data_generation(array, fac):
    """ make step data for PCAs
    array = [[mjd_1, RSW_1], ...]
    fac -> new time step = time step / fac
        example: time step = 6 min, fac = 12
            -> new time step = 6 min / 12 = 30 sec
    """
    if (fac == 1):
        return(array)
    MJD = array[:, 0]
    acc = array[:, 1:]
    
    step = MJD[1] - MJD[0] # assuming equidistant dataset
    
    MJD_new = np.array([])
    acc_new = acc[0]
    
    for i in range(0, len(array)):
        MJD_i = MJD[i]
        MJD_i_array = np.linspace(MJD_i, MJD_i + step, fac + 1)[: -1]
        acc_i_array = np.tile(acc[i], (fac, 1))
        
        MJD_new = np.append(MJD_new, MJD_i_array)
        acc_new = np.vstack((acc_new, acc_i_array))
    array_new = np.hstack((np.array([MJD_new]).T, acc_new[1:]))
    return(array_new)

def master_integrator_coupled(u_data, a_data, e_data, r_data, omega_data, path_PCA, mjd_interval):
    """INPUT:
            u_data: data from LST-Files [[mjd1, u1], [mjd2, u2], ...]
            path_ele_osc: data from ELE-Files (file generated with master_ele function)
            path_PCA: data from ELE-Files (file generated with master_ele function)
            mjd_interval: time interval for integration
    OUTPUT:
        a_int_data: integrated semi-major axis
        a_dot_data: slope of integrated semi-major axis
    """
    print("master_integrator_coupled")
    MJD_start, MJD_end = mjd_interval[0], mjd_interval[1]
    n_days = MJD_end - MJD_start
    
    u_data = array_modifier(u_data, MJD_start, n_days) # crop data
    
    a_0 = a_data[0]
    e_0 = e_data[0]
    ω = omega_data
    r = r_data
    
    acc_R_data = np.loadtxt(path_PCA, usecols = (0, 1))
    acc_S_data = np.loadtxt(path_PCA, usecols = (0, 2))
    
    acc_R_data = array_modifier(acc_R_data, MJD_start, n_days) # crop data
    acc_S_data = array_modifier(acc_S_data, MJD_start, n_days) # crop data
    
    # resample PCA data
    dt_u = u_data[1, 0] - u_data[0, 0]
    dt_acc = acc_R_data[1, 0] - acc_R_data[0, 0]
    fac = int(np.round(dt_acc / dt_u))
    acc_R_data = step_data_generation(acc_R_data, fac)
    acc_S_data = step_data_generation(acc_S_data, fac)
    a_int_data, a_dot_data, e_int_data, e_dot_data = integrator_coupled(a_0, e_0, ω, r, u_data, acc_R_data, acc_S_data)
    return(a_int_data, a_dot_data, e_int_data, e_dot_data)















