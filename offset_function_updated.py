"""
@author: ivovollmer
"""
import numpy as np
    
def offset_correction_overlap(data, n_per_day, n_overlap, n_days):
    was_1d = False
    if data.ndim == 1:
        data = data[:, np.newaxis]
        was_1d = True
    n_cols = data.shape[1]

    n_per_day_no_overlap = n_per_day - n_overlap

    end_day = np.array([data[i*n_per_day + n_per_day - n_overlap : i*n_per_day + n_per_day] for i in range(n_days - 1)])
    start_day = np.array([data[(i+1)*n_per_day : (i+1)*n_per_day + n_overlap] for i in range(n_days - 1)])

    residuals = start_day - end_day
    offsets = residuals.mean(axis=1)
    corrections = np.vstack([np.zeros((1, n_cols)), np.cumsum(offsets, axis=0)])

    data_corrected = np.empty((n_days * n_per_day_no_overlap, n_cols))
    for i in range(n_days):
        start_in = i * n_per_day
        end_in = start_in + n_per_day_no_overlap
        start_out = i * n_per_day_no_overlap
        end_out = start_out + n_per_day_no_overlap

        data_corrected[start_out:end_out] = data[start_in:end_in] - corrections[i]

    if was_1d:
        return data_corrected.ravel()
    else:
        return data_corrected




def offset_correction_direct(data, n_per_day, n_days, n_comp):
    if data.ndim == 1:
        data = data[:, np.newaxis]
    n_cols = data.shape[1]

    offsets = np.zeros((n_days - 1, n_cols))
    for day in range(n_days - 1):
        idx_end = (day + 1) * n_per_day
        data_end_day = data[idx_end - n_comp : idx_end]
        data_start_next_day = data[idx_end : idx_end + n_comp]

        step_end = np.diff(data_end_day, axis=0).mean(axis=0)
        step_start = np.diff(data_start_next_day, axis=0).mean(axis=0)
        normal_step = (step_end + step_start) / 2

        actual_jump = data[idx_end] - data[idx_end - 1]
        offsets[day] = actual_jump - normal_step

    corrections = np.vstack([np.zeros(n_cols), np.cumsum(offsets, axis=0)])

    data_corrected = np.empty_like(data)
    for day in range(n_days):
        start = day * n_per_day
        end = (day + 1) * n_per_day
        data_corrected[start:end] = data[start:end] - corrections[day]
        
    if n_cols == 1:
        return data_corrected.ravel()
    else:
        return data_corrected


