"""
@author: ivovollmer
"""
from astropy.time import Time
import numpy as np
import re
import glob
import os

def import_data(file_path, skip_row_begin, skip_row_end, csv=False): # Liest eine Datei und verarbeitet Inhalt zu separierten Werten in einem Array
    with open(file_path, "r", encoding="utf-8") as file:
        raw_text = file.read()
    clean_text = re.sub(r"{\\.*?}", "", raw_text)  
    clean_text = re.sub(r"\\[a-zA-Z0-9]+", "", clean_text)  
    clean_text = clean_text.replace("\r\n", "\n")
    lines = clean_text.split("\n")
    data_lines = lines[skip_row_begin:-skip_row_end] if skip_row_end != 0 else lines[skip_row_begin:]
    if csv:
        result = []
        for line in data_lines:
            if line.strip():
                time_part, value = line.split(",")
                date, time = time_part.split()
                result.append([date, time, value])        
        date_times = [f"{date}T{time}" for date, time, _ in result]  
        mjd_times = Time(date_times, format='isot', scale='utc').mjd
        values = [float(value) for _, _, value in result]
        cleaned_data = np.column_stack((mjd_times, values))  
    else:
        data = [line.split() for line in data_lines if line.strip()]    
        cleaned_data = [[value.rstrip("\\") for value in row] for row in data]   
    return np.array(cleaned_data, dtype=float)



def load_all_data(folder_path, skip_row_begin, skip_row_end, csv=False): # Verarbeitet alle Dateien in einem Ordner zu einem gesamten Array über alle Werte
    file_list = glob.glob(os.path.join(folder_path, "*"))
    all_data = []
    for file_path in sorted(file_list):
        data = import_data(file_path, skip_row_begin, skip_row_end, csv=csv)
        all_data.append(data)
    final_data = np.vstack(all_data)
    return final_data







