# atmdensity_variations_geostorms
Python codes and scripts developed for the Bachelor's thesis by Ivo Vollmer on atmospheric density variations in low Earth orbit during geomagnetic storms.

# Bachelor Thesis Code Repository

## Structure
- All files ending with `_Plots.py` can be executed to reproduce the plots shown in the thesis.
- The other files contain the required functions and must be placed in a folder.
- In each `_Plots.py` script you must adjust the following:
  - At the top, update the `sys.path.append('...')` line to point to the folder where the functions are stored.
  - Update the paths to the input data files as needed.

## Notes
- Some scripts require significant computation time, especially:
  - `Fit_Model_Plots.py`
  - `ACC_Final_Plots.py`

## Contact
For questions or issues, please contact:  
ivo.vollmer@students.unibe.ch


## Additional Remark
In a previous bachelor thesis, Levin Walter analyzed the semi-major axis fit model over long time intervals in great detail and evaluated it in relation to the Gaussian approach. The corresponding Python scripts and thesis are available in his GitHub repository: https://github.com/coresnprogrammer/ISWEOLEOS.git.
