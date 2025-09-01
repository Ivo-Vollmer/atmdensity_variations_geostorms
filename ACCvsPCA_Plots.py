"""
@author: ivovollmer
"""
import numpy as np
from import_files import *
import matplotlib.pyplot as plt
import sys
sys.path.append('PATHTOCODEFUNCTIONS')
from functions import *
# %matplotlib auto

file1 = 'PATHFILEPCA'
file2 = 'PATHFILEACC'
data_weather_2024 = load_all_data("PATHDSTINDEXDATA", 1, 1, csv=True)

def load_acc_data(filepath):
    data = np.loadtxt(filepath)
    mjd = data[:, 0]
    acc_R = data[:, 1]
    acc_S = data[:, 2]
    acc_W = data[:, 3]
    acc_total = data[:, 4]
    return mjd, acc_R, acc_S, acc_W, acc_total

mjd1, acc_R1, acc_S1, acc_W1, acc_total1 = load_acc_data(file1)
mjd2, acc_R2, acc_S2, acc_W2, acc_total2 = load_acc_data(file2)

tot_new = np.sqrt((acc_R1)**2+(acc_S1)**2+(acc_W1)**2)




# Plots
plt.rcParams["figure.dpi"] = 1000
fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
# Plot für die R-Komponente
axs[0].plot(mjd1, acc_R1, label='PCA R', color='red', linewidth=1)
axs[0].plot(mjd2, acc_R2, label='ACC R', color='blue', linewidth=1)
axs[0].set_ylabel(r'Acceleration $\left[ \frac{m}{s^2} \right]$')
axs[0].set_xlim(60440, 60443) 
axs[0].legend(loc='upper right')
axs[0].grid()

# Plot für die S-Komponente
axs[1].plot(mjd1, acc_S1, label='PCA S', color='red', linewidth=1)
axs[1].plot(mjd2, acc_S2, label='ACC S', color='blue', linewidth=1)
axs[1].set_ylabel(r'Acceleration $\left[ \frac{m}{s^2} \right]$')
axs[1].legend(loc='upper right')
axs[1].grid()

# Plot für die W-Komponente
axs[2].plot(mjd1, acc_W1, label='PCA W', color='red', linewidth=1)
axs[2].plot(mjd2, acc_W2, label='ACC W', color='blue', linewidth=1)
axs[2].set_ylabel(r'Acceleration $\left[ \frac{m}{s^2} \right]$')
axs[2].legend(loc='upper right')
axs[2].grid()

# Plot für die totale Beschleunigung
axs[3].plot(mjd1, tot_new, label='PCA', color='red', linewidth=1)
axs[3].plot(mjd2, acc_total2, label='ACC', color='blue', linewidth=1)
axs[3].set_ylabel(r'Acceleration $\left[ \frac{m}{s^2} \right]$')
axs[3].set_xlabel('Date and Time (mm.dd HH:MM, 2024)')
axs[3].legend(loc='upper right')
axs[3].grid()
for ax in axs:
    ax.xaxis.set_major_formatter(mjd_to_mmddhh)
plt.tight_layout()
plt.show()


#Einzelplot der S-Komponente
plt.rcParams["figure.dpi"] = 1000
fig, ax = plt.subplots(figsize=(8, 4)) 
ax.plot(mjd1, acc_S1, label='PCA S', color='red', linewidth=1)
ax.plot(mjd2, acc_S2, label='ACC S', color='blue', linewidth=1)
ax.xaxis.set_major_formatter(mjd_to_mmddhh)
ax.set_xlim(60440, 60443)
ax.set_xlabel('Date and time (mm.dd HH:MM, GPST, 2024)')
ax.set_ylabel(r'Acceleration $\left[ \frac{m}{s^2} \right]$')
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
order = [0, 1, 2]
handles = [handles[i] for i in order]
labels = [labels[i] for i in order]
fig.legend(
    handles=handles,
    labels=labels,
    loc='lower center',
    ncol=1,
    bbox_to_anchor=(0.5, -0.15),
    frameon=False)
plt.subplots_adjust(bottom=0.2)
plt.show()






