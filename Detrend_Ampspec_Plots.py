"""
@author: ivovollmer
"""
import sys
sys.path.append('PATHTOCODEFUNCTIONS')
from gauss_func import *
from import_files import *
from functions_final import *
from fit_model_error import *
from offset_function_updated import *
from math import *
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

##################################################################################################################################
##################################################################################################################################
data_RD_30s = load_all_data("PATHRD30SLSTDATA", 24, 1) #Volle Messreihe GFOC, 2024, RD, 30s sampling
a_array = data_RD_30s[:, 20]
time_array = data_RD_30s[:, 1]

#Trend bestimmen
win_len = 5*188+1

data_a_sI = np.convolve(a_array, np.ones(win_len)/win_len, mode='same')
data_a_sII = np.convolve(data_a_sI, np.ones(win_len)/win_len, mode='same')

#Trend-bereinigt
detrend_a = a_array-data_a_sII


#Plot 1: Amplitudenspektrum
amplitudes_full, freqs = fourier_results(detrend_a, 30)    
freq_max = max_freq(amplitudes_full, freqs)
periods_plot =1440*1/freqs[1:]
amplitudes_plot = amplitudes_full[1:]

#Prominente Perioden bestimmen
mask = (periods_plot >= 10) & (periods_plot <= 100) 
periods_zoom = periods_plot[mask]
amplitudes_zoom = amplitudes_plot[mask]
peaks, _ = find_peaks(amplitudes_zoom, prominence=150)
peak_indices_global = np.where(mask)[0][peaks]
print("Gefundene Perioden mit Peaks:")
print(periods_plot[peak_indices_global])



##################################################################################################################################
#Amplitudenspektrum, passt
plt.rcParams["figure.dpi"] = 1000
fig, ax = plt.subplots(figsize=(8, 4)) 
plt.plot(periods_plot, amplitudes_plot, color='blue', linewidth=1)
plt.xlabel('Period [min]')
plt.ylabel('Amplitude [m]')
plt.xscale('log')  # Optional, log-Skala für bessere Übersicht
plt.yscale('log')
plt.grid()
plt.show()





