"""
@author: ivovollmer
"""
import sys
sys.path.append('PATHTOCODEFUNCTIONS')
from gauss_func import *
from import_files import *
from functions import *
from fit_model_error import *
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
dadt_gauss = a_dot_data[:, 1]                                                                                                                                
dadt_gauss_ma = np.convolve(dadt_gauss, np.ones(win_len_ma)/win_len_ma, mode='same')    

win_len_ma_half = 47
dadt_gauss_ma_half = np.convolve(dadt_gauss, np.ones(win_len_ma_half)/win_len_ma_half, mode='same')     
##################################################################################################################################


#Gauss-Approach von ACC-Daten-----------------
data_RD_ACC_len = load_all_data("PATHRD1SLSTDATA", 25, 302) #2024, RD, 1s sampling, bis eine Sekunde vor Mitternacht (pro Tag 86399 Messungen)
ACC_data = load_all_data("PATHACCDATA", 9, 1) #ACC Daten 
#Korrektur der Zeit
n_days = 16
day_length = 86399
base_value = 60435 #MJD
day_offsets = np.arange(base_value, base_value + n_days)
offsets_repeated = np.repeat(day_offsets, day_length)

ACC_data[:, 0] += offsets_repeated
ACC_data_reduced = ACC_data[:, :4]  
ACC_data_reduced[:, 1:4] *= 0.001 
acc_total = np.linalg.norm(ACC_data_reduced[:, 1:4], axis=1)  
ACC_data_final = np.column_stack((ACC_data_reduced, acc_total)) 
np.savetxt('PATHFILEOUTPUTACC', ACC_data_final, delimiter=' ', fmt='%.10e')

u_sat_RD_1s = data_RD_ACC_len[:, 17]
# u_sat_RD_1s_mod = np.delete(u_sat_RD_1s, np.arange(86399, len(u_sat_RD_1s), 86400))
u_data_acc = np.column_stack((ACC_data_final[:, 0], u_sat_RD_1s))
mjd_interval = [60435, 60451]
a_int_data_acc, a_dot_data_acc = master_integrator(u_data_acc, 'PATHFILEELEOSC', 'PATHFILEACC', mjd_interval)
time_a_acc = a_int_data_acc[:, 0]
a_gauss_acc = a_int_data_acc[:, 1]
time_dadt_acc = a_dot_data_acc[:, 0]
#Moving average für dadt
win_len_ma_acc = 2820+1     
win_len_ma_acc_half = 1410+1                                             
dadt_gauss_acc = a_dot_data_acc[:, 1]                                                                                                                                
dadt_gauss_ma_acc = np.convolve(dadt_gauss_acc, np.ones(win_len_ma_acc)/win_len_ma_acc, mode='same')  
dadt_gauss_ma_acc_half = np.convolve(dadt_gauss_acc, np.ones(win_len_ma_acc_half)/win_len_ma_acc_half, mode='same')  



##################################################################################################################################
plt.rcParams["figure.dpi"] = 1000
fig, ax = plt.subplots(figsize=(8, 4)) 
ax.plot(time_dadt, dadt_gauss_ma, linewidth=1, color='red', label="Reference (PCA, Gaussian approach, window width = 94 min)", zorder=0)
ax.plot(time_dadt_acc, dadt_gauss_ma_acc, color='blue', linewidth=1, label='ACC, Gaussian approach (window width = 94 min)')
ax.xaxis.set_major_formatter(mjd_to_mmddhh)
ax.set_xlim(60440, 60443)
ax.set_ylim(-175, 10)
ax.set_xlabel('Date and time (mm.dd HH:MM, GPST, 2024)')
ax.set_ylabel(r'$\frac{da}{dt} \; \left[ \frac{m}{d} \right]$')
ax.grid()
#Rechte Achse
ax_right = ax.twinx()
ax_right.plot(data_weather_2024[:, 0], data_weather_2024[:, 1], color='grey', linestyle='--', linewidth=1, label='Dst index', zorder=0)
ax_right.set_ylabel('Dst index [nT]')
ax_right.set_ylim(-450, 90)
#Legende
lines_left, labels_left = ax.get_legend_handles_labels()
lines_right, labels_right = ax_right.get_legend_handles_labels()
handles = lines_left + lines_right
labels = labels_left + labels_right
order = [1, 0, 2]
handles = [handles[i] for i in order]
labels = [labels[i] for i in order]
fig.legend(
    handles=handles,
    labels=labels,
    loc='lower center',
    ncol=1,
    bbox_to_anchor=(0.5, -0.2),
    frameon=False)
plt.subplots_adjust(bottom=0.2)
plt.show()

plt.rcParams["figure.dpi"] = 1000
fig, ax = plt.subplots(figsize=(8, 4)) 
ax.plot(time_dadt, dadt_gauss_ma, linewidth=1, color='darkviolet', label="Reference (PCA, Gaussian approach, window width = 94 min)", zorder=0)
ax.plot(time_dadt, dadt_gauss_ma_half, linewidth=1, color='red', label="Reference (PCA, Gaussian approach, window width = 47 min)", zorder=1)
ax.plot(time_dadt_acc, dadt_gauss_ma_acc_half, color='blue', linewidth=1, label='ACC, Gaussian approach (window width = 47 min)', zorder=2)
ax.xaxis.set_major_formatter(mjd_to_mmddhh)
ax.set_xlim(60440, 60443)
ax.set_ylim(-175, 10)
ax.set_xlabel('Date and time (mm.dd HH:MM, GPST, 2024)')
ax.set_ylabel(r'$\frac{da}{dt} \; \left[ \frac{m}{d} \right]$')
ax.grid()
#Rechte Achse
ax_right = ax.twinx()
ax_right.plot(data_weather_2024[:, 0], data_weather_2024[:, 1], color='grey', linestyle='--', linewidth=1, label='Dst index', zorder=0)
ax_right.set_ylabel('Dst index [nT]')
ax_right.set_ylim(-450, 90)
#Legende
lines_left, labels_left = ax.get_legend_handles_labels()
lines_right, labels_right = ax_right.get_legend_handles_labels()
handles = lines_left + lines_right
labels = labels_left + labels_right
order = [2, 1, 0, 3]
handles = [handles[i] for i in order]
labels = [labels[i] for i in order]
fig.legend(
    handles=handles,
    labels=labels,
    loc='lower center',
    ncol=1,
    bbox_to_anchor=(0.5, -0.2),
    frameon=False)
plt.subplots_adjust(bottom=0.2)
plt.show()

