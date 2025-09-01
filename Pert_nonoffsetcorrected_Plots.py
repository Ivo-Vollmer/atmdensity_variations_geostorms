"""
@author: ivovollmer
"""
import sys
sys.path.append('PATHTOCODEFUNCTIONS')
from gauss_func import *
from import_files import *
from functions import *
from int_daily_function import *
from offset_function_updated import *
from math import *
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

##################################################################################################################################
data_weather_2024 = load_all_data("PATHDSTINDEXDATA", 1, 1, csv=True)
# Grace-FO-C 2024----------------------------------------------------------- 
data_RD_30s = load_all_data("PATHRD30SDATA", 24, 1) #Volle Messreihe GFOC, 2024, RD, 30s sampling
time_30s = data_RD_30s[:, 1]
u_sat_RD = data_RD_30s[:, 17]

#Gauss-Approach alte Daten und alte Methode-------------------------------
master_ele('/PATHRDELEDATA', 'PATHFOLDERSAVEFILE') 
u_data = np.column_stack((time_30s, u_sat_RD))
mjd_interval = [60431, 60492]
a_int_data, a_dot_data = master_integrator(u_data, 'PATHFOLDERSAVEFILE_ELEOSC', 'PATHFOLDERSAVEFILE_PCA', mjd_interval)
time_a = a_int_data[:, 0]
a_gauss = a_int_data[:, 1]
time_dadt = a_dot_data[:, 0]
#Moving average für dadt
win_len_ma = 94+1       
win_len_ma_half = 47                               
dadt_gauss = a_dot_data[:, 1]                                                                                                                                
dadt_gauss_ma = np.convolve(dadt_gauss, np.ones(win_len_ma)/win_len_ma, mode='same')    
dadt_gauss_ma_half = np.convolve(dadt_gauss, np.ones(win_len_ma_half)/win_len_ma_half, mode='same')    



##################################################################################################################################
#------------------------------------------------------------------------------
grav_data = load_all_data("PATHPERTURBATION1SLSTDATA", 28, 301) #GFOC Daten mit Grav-Beschleunigungen zweiter Input von 25 erhöht auf 28 und dritter Input von 31 auf 301
sampling = 1 
dt = 1  

n_per_day = 86400
n_days = 61

time_data = grav_data[:, 1]
a_array = grav_data[:, 20]

a_array_resamp = a_array[::2*dt]
time_resamp = time_data[::2*dt]

e_array = grav_data[:, 21]
u_array = grav_data[:, 17] * np.pi / 180 
omega_array = grav_data[:, 24] * np.pi / 180 
r_array = grav_data[:, 11]
R_array = grav_data[:, 34] * 0.001 
S_array = grav_data[:, 35] * 0.001 

SRP_R = grav_data[:, 37] * 10**-9
SRP_S = grav_data[:, 38] * 10**-9
ref_PRP_R = grav_data[:, 40] * 10**-9
ref_PRP_S = grav_data[:, 41] * 10**-9
em_PRP_R = grav_data[:, 43] * 10**-9
em_PRP_S = grav_data[:, 44] * 10**-9

#------
a_grav = int_daily(a_array, e_array, u_array, omega_array, r_array, R_array, S_array, n_per_day, n_days, dt)
print("a_grav done")
a_SRP = int_daily(a_array, e_array, u_array, omega_array, r_array, SRP_R, SRP_S, n_per_day, n_days, dt)
print("a_SRP done")
a_refPRP = int_daily(a_array, e_array, u_array, omega_array, r_array, ref_PRP_R, ref_PRP_S, n_per_day, n_days, dt)
print("a_refPRP done")
a_emPRP = int_daily(a_array, e_array, u_array, omega_array, r_array, em_PRP_R, em_PRP_S, n_per_day, n_days, dt)
print("a_emPRP done")
#------
a_airres = a_array_resamp - a_grav #- a_SRP - a_refPRP - a_emPRP
n_comparison = 10
# a_airres_cont = offset_correction_direct(a_airres, int(len(a_airres)/n_days), n_days, n_comparison)
a_airres_deriv, time_grav_deriv = cal_deriv(a_airres, 2*dt/86400, time_resamp)

# win_len_ma_half = int(282*10/(2*dt)+1) #half orbit
win_len_ma_full = int(282*10/(2*dt)+1) * 2 #full orbit
movavg_full_a_airres_deriv = np.convolve(a_airres_deriv, np.ones(win_len_ma_full)/win_len_ma_full, mode='same')   
# movavg_half_a_airres_deriv = np.convolve(a_airres_deriv, np.ones(win_len_ma_half)/win_len_ma_half, mode='same')   







# ##################################################################################################################################
plt.rcParams["figure.dpi"] = 750
fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=False)  # zwei Plots nebeneinander
# -------------------------
# Erster Plot
axs[0].plot(time_resamp, a_airres, linewidth=1, color='blue', label="Perturbation approach")
axs[0].xaxis.set_major_formatter(mjd_to_mmddhh)
axs[0].set_xlabel('Date and time (mm.dd HH:MM, GPST, 2024)')
axs[0].set_ylabel('a(t) [m]')
axs[0].grid()
ax0_right = axs[0].twinx()
ax0_right.plot(data_weather_2024[:, 0], data_weather_2024[:, 1], color='grey', linestyle='--', linewidth=1, label='Dst index', zorder=0)
ax0_right.set_ylabel('Dst index [nT]')
ax0_right.set_ylim(-450, 90)
# -------------------------
# Zweiter Plot
axs[1].plot(time_grav_deriv, movavg_full_a_airres_deriv, linewidth=1, color='blue', label="Perturbation approach")
axs[1].xaxis.set_major_formatter(mjd_to_mmddhh)
axs[1].set_xlabel('Date and time (mm.dd HH:MM, GPST, 2024)')
axs[1].set_ylabel(r'$\frac{da}{dt} \; \left[ \frac{m}{d} \right]$')
axs[1].grid()
ax1_right = axs[1].twinx()
ax1_right.plot(data_weather_2024[:, 0], data_weather_2024[:, 1], color='grey', linestyle='--', linewidth=1, label='Dst index', zorder=0)
ax1_right.set_ylabel('Dst index [nT]')
ax1_right.set_ylim(-450, 90)
# -------------------------
lines_left0, labels_left0 = axs[0].get_legend_handles_labels()
lines_left1, labels_left1 = axs[1].get_legend_handles_labels()
lines_right0, labels_right0 = ax0_right.get_legend_handles_labels()
lines_right1, labels_right1 = ax1_right.get_legend_handles_labels()
handles = lines_left0 + lines_left1 + lines_right0 + lines_right1
labels  = labels_left0 + labels_left1 + labels_right0 + labels_right1
unique = {}
for h, l in zip(handles, labels):
    if l not in unique:  
        unique[l] = h
order = ["Perturbation approach",
         "Reference (Gaussian approach)",
         r"$\frac{da}{dt} \; \left[ \frac{m}{d} \right]$",  
         "Dst index"]
handles_final = [unique[l] for l in order if l in unique]
labels_final  = [l for l in order if l in unique]
fig.legend(
    handles=handles_final,
    labels=labels_final,
    loc='lower center',
    ncol=2,
    bbox_to_anchor=(0.5, -0.05),
    frameon=False)
plt.subplots_adjust(bottom=0.25)
plt.tight_layout()
plt.show()
