"""
@author: ivovollmer
"""
import numpy as np
from scipy.linalg import lstsq
from math import *
from astropy.time import Time
from scipy.fft import rfft, rfftfreq

#-----------------Fit Modell---------------

def fit_model_error(data_a, time, periods, constrain_strength=0.0, trend_subintervals_per_day=50, multi_constr=False):
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
    all_params_error = []
    all_fit_values = []
    all_fit_errors = []
    

    for constr_str_value in strengths:
        N = N_base + constr_str_value * C_full
        params = np.linalg.solve(N, b)
        fit_values = design_matrix @ params
        all_params.append(params)
        all_fit_values.append(fit_values)
                        
        residuals = data_a - fit_values
                                
        trim = int(0.1 * len(residuals))  # 10% der Länge vorne und hinten wegschneiden, da dort fit nicht passend ist und ergebnis verfälschen würde
        residuals_cut = residuals[trim:-trim]  
        
        n_points_cut = len(residuals_cut)
        n_params = design_matrix.shape[1]
        dof_cut = n_points_cut - n_params
        mse_cut = np.sum(residuals_cut**2) / dof_cut
        print(mse_cut)
        # Kovarianzmatrix
        cov_params = mse_cut * np.linalg.inv(N)
        params_std = np.sqrt(np.diag(cov_params))
        all_params_error.append(params_std)
    
        fit_errors = np.sqrt(np.sum(design_matrix * (design_matrix @ cov_params), axis=1))
        all_fit_errors.append(fit_errors)

    if multi_constr:
        return all_params, all_fit_values, all_fit_errors, all_params_error
    else:
        return all_params[0], all_fit_values[0], all_fit_errors[0], all_params_error[0]





















