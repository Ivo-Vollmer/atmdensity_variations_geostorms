"""
@author: ivovollmer
"""
import numpy as np
from scipy.linalg import lstsq
from math import *
from astropy.time import Time
from scipy.fft import rfft, rfftfreq

#-----------------Fit Modell (ohne Fehlerrechnung)---------------
def fit_model(data_a, time, periods, constrain_strength=0.0, trend_subintervals_per_day=50, multi_constr=False):
    #1. Anzahl Subintervalle + Zeit/Position der Stützstellen
    time_range = time.max() - time.min()
    n_trend_pieces = int(round(time_range * trend_subintervals_per_day))
    trend_nodes = np.linspace(time.min(), time.max(), n_trend_pieces + 1)

    #2. Basisfunktionen für Stückweise lineare interpolation
    def piecewise_linear_basis(time, nodes):
        n_basis = len(nodes)
        basis = np.zeros((len(time), n_basis)) 
        for i in range(1, n_basis - 1):
            left, center, right = nodes[i-1], nodes[i], nodes[i+1]
            mask = (time >= left) & (time <= right)
            basis[mask, i] = np.where(time[mask] <= center,
                                      (time[mask] - left) / (center - left),
                                      (right - time[mask]) / (right - center))
        #Randbedingungen setzen
        basis[time <= nodes[1], 0] = 1
        basis[time >= nodes[-2], -1] = 1
        return basis

    trend_basis = piecewise_linear_basis(time, trend_nodes)

    #3. Harmonische Basis 
    harmonic_basis = []
    for period in periods:
        omega = 2 * np.pi / period
        sin_basis = piecewise_linear_basis(time, trend_nodes) * np.sin(omega * time[:, None])
        cos_basis = piecewise_linear_basis(time, trend_nodes) * np.cos(omega * time[:, None])
        harmonic_basis.append(sin_basis)
        harmonic_basis.append(cos_basis)
    harmonic_basis = np.hstack(harmonic_basis)

    #4. Designmatrix
    design_matrix = np.hstack([trend_basis, harmonic_basis])

    #5. Constraining Matrix 
    C_base = np.zeros((n_trend_pieces - 1, n_trend_pieces + 1))
    for i in range(n_trend_pieces - 1):
        C_base[i, i] = 1
        C_base[i, i+1] = -2
        C_base[i, i+2] = 1
    C = np.kron(np.eye(1 + 2 * len(periods)), C_base) 
    
    #6. Erweiterung Normalengleichung
    N_base = design_matrix.T @ design_matrix
    C_full = C.T @ C
    b = design_matrix.T @ data_a
    strengths = np.atleast_1d(constrain_strength)

    all_params = []
    all_fit_values = []

    for constr_str_value in strengths:
        N = N_base + constr_str_value * C_full
        params = np.linalg.solve(N, b)
        fit_values = design_matrix @ params
        all_params.append(params)
        all_fit_values.append(fit_values)

    if multi_constr:
        return all_params, all_fit_values
    else:
        return all_params[0], all_fit_values[0]




#-----------------Allgemeine Funktionen---------------
def mjd_to_mmdd (t, pos):
        t_obj = Time(str(t), format = 'mjd')
        t_iso = t_obj.iso[5:10].replace('-','.')
        return(t_iso)
    
def mjd_to_mmddhh(x, pos):
    dt = Time(x, format="mjd").to_datetime()
    return dt.strftime("%m.%d\n%H:%M")



def max_freq(amplitudes, frequencies):
    idx_dc = np.argmax(amplitudes)
    amp_dc = amplitudes[idx_dc]
    freq_dc = frequencies[idx_dc]
    amplitudes_copy = amplitudes.copy()
    amplitudes_copy[idx_dc] = -np.inf  
    idx_max = np.argmax(amplitudes_copy)
    amp_max = amplitudes[idx_max]
    freq_max = frequencies[idx_max]
    return freq_max

#-----------------Differential Funktion---------------
def cal_deriv(a_values, delta_t_days, time): #delta_t_days in Tage angeben
    delta_a = np.diff(a_values)  
    da_dt = delta_a / delta_t_days  
    time_mid = (time[:-1] + time[1:]) / 2
    return da_dt, time_mid


#----------------Fehlerfortpflanzung der Differential Funktion---------
def cal_deriv_error(a_values_error, delta_t_days, time): #delta_t_days in Tage angeben
    delta_a_error = a_values_error[1:]**2 + a_values_error[:-1]**2
    da_dt_error = np.sqrt(delta_a_error) / delta_t_days  
    time_mid = (time[:-1] + time[1:]) / 2
    return da_dt_error, time_mid


#--------------Fourier-Analyse:
def fourier_results(data, sampling):
    dt=sampling/86400                                            
    fft_result = rfft(data)                               
    freqs = rfftfreq(len(data), d=dt)               

    amplitudes = 2 * np.abs(fft_result) / len(data)
    amplitudes[0] = np.abs(fft_result[0]) / len(data)     
    periods = np.zeros_like(freqs)  
    periods[1:] = 1 / freqs[1:]                             
    periods[0] = np.inf   
    return amplitudes, freqs

#-------------Gewichteter Savgol-Filter (Eine Seite des Fensters stärker gewichtet als andere Seite), rechnerisch sehr ineffizient, nicht verwendet
def weighted_savgol(data, n_half, poly_order, left_weight=1.0, right_weight=1.0):
    n_data = len(data)
    window_size = 2 * n_half + 1
    x_rel = np.arange(-n_half, n_half + 1) 
    weights = np.where(x_rel < 0, left_weight, np.where(x_rel > 0, right_weight, 1.0))
    T = np.vander(x_rel, poly_order + 1, increasing=True)
    W = np.diag(weights)
    C = np.linalg.pinv(W @ T) @ (W)
    filter_kernel = C[0, :]  

    smoothed = np.convolve(data, filter_kernel[::-1], mode='same')

    # Randbereiche korrigieren
    for i in range(n_half):
        # Linker Rand
        rel_idx = np.arange(0, i + n_half + 1) - i
        y_window = data[0 : i + n_half + 1]
        T_edge = np.vander(rel_idx, poly_order + 1, increasing=True)
        W_edge = np.diag(np.where(rel_idx < 0, left_weight, np.where(rel_idx > 0, right_weight, 1.0)))
        C_edge = np.linalg.pinv(W_edge @ T_edge) @ (W_edge @ y_window)
        smoothed[i] = np.polyval(C_edge[::-1], 0)
     

    for i in range(n_data - n_half, n_data):
        # Rechter Rand
        rel_idx = np.arange(i - n_half, n_data) - i
        y_window = data[i - n_half : n_data]
        T_edge = np.vander(rel_idx, poly_order + 1, increasing=True)
        W_edge = np.diag(np.where(rel_idx < 0, left_weight, np.where(rel_idx > 0, right_weight, 1.0)))
        C_edge = np.linalg.pinv(W_edge @ T_edge) @ (W_edge @ y_window)
        smoothed[i] = np.polyval(C_edge[::-1], 0)
       

    return smoothed 


#---------------Gewichteter moving average Filter, rechnerisch sehr ineffizient, nicht verwendet
def weighted_moving_average(data, win_len, left_weight=2.0, right_weight=1.0):
    half_win = win_len // 2
    kernel = np.ones(win_len)

    kernel[:half_win] *= left_weight   
    kernel[half_win+1:] *= right_weight  
    kernel[half_win] *= 1.0  

    kernel /= np.sum(kernel)

    smoothed = np.convolve(data, kernel, mode='same')
    return smoothed



