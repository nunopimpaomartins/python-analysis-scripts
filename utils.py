import numpy as np
# from numba import jit

# IO functions


# Normlaization functions
def normalize_min_max(data, new_min=0.0, new_max=1.0):
    """
    Normalize data to the range [new_min, new_max]
    param data: data to be normalized
    param new_min: new minimum value
    para new_max: new maximum value
    returns: normalized data
    """
    data_min = np.min(data)
    data_max = np.max(data)

    return (data - data_min) * (new_max - new_min) / (data_max - data_min) + new_min

def normalize_standardize(data):
    """
    Standardize data
    param data: data to be standardized
    returns: standardized data
    """
    data_mean = np.mean(data)
    data_std = np.std(data)

    return (data - data_mean) / data_std

