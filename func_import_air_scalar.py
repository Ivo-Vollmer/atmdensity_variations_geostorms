"""
@author: ivovollmer
"""
from math import *
import numpy as np
import os


def extract_air_value_from_line(file_path, line_number):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        target_line = lines[line_number - 1]

        if not target_line.strip().startswith("AIR"):
            raise ValueError(f"Zeile {line_number} enthaelt nicht 'AIR' in Datei {file_path}")

        parts = target_line.split("=")[1].strip().split()
        second_value = float(parts[1].replace('D', 'E'))
        print("extract_air_value_from_line done")
        return second_value

def process_ele_files(foldername, line_number, pathsave):
    results = []
    for filename in sorted(os.listdir(foldername)):
        print("filename loop")
        if filename.endswith(".ELE"):
            print("ele loop")
            file_path = os.path.join(foldername, filename)
            try:
                air_value = extract_air_value_from_line(file_path, line_number)
                results.append(air_value)
                print(f"Wert {air_value} erfolgreich aus {filename} extrahiert.")
            except Exception as e:
                print(f"Fehler bei Datei {filename}: {e}")

    with open(pathsave, 'w') as output_file:
        for value in results:
            output_file.write(f"{value}\n")
    print("process_ele_file done")
    
    










