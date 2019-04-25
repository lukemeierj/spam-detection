import numpy as np
import math

# Splits into equal ranges
def linear_bin(arr, num_bins, minmax = (0, 100)):
    start, end = minmax
    chunksize = (end - start) // num_bins
    return [ v // chunksize for v in arr ]

def logarithmic_bin(arr, num_bins, minmax = (0, 100), base = 2):
    start, end = (math.log(max(v,1), base) for v in minmax)
    bins = np.logspace(start, end, base = base, num = num_bins+1)[1:]
    return [ next((i for i, bin_val in enumerate(bins) 
                                  if bin_val >= v), -1) for v in arr]

def binned_data(data, col_nums, fn, num_bins = 5, minmax = (0, 100)):
    for col in col_nums:
        binned = [row[:] for row in data]
        binned[col] = fn(data[col], num_bins)
    return binned
