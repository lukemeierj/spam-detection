import numpy as np
import math

# Splits into equal ranges
def linear_bin(arr, num_bins, minmax = (0, 100)):
    start, end = minmax
    chunksize = (end - start) // num_bins
    return [ v // chunksize for v in arr ]

def logarithmic_bin(arr, num_bins, minmax = (0, 100), base = 2, minval = 1e-8):
    start, end = (math.log(max(v,minval), base) for v in minmax)
    bins = np.logspace(start, end, base = base, num = num_bins+1)[1:]
    return [ next((i for i, bin_val in enumerate(bins) 
                                  if bin_val >= v), -1) for v in arr]

def binned_data(data, col_nums, fn, num_bins = 5, minmax = (0, 100)):
    binned = data.copy().transpose()
    for col in col_nums:
        binned[col] = fn(binned[col], num_bins)

    return binned.transpose()
