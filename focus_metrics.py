import numpy as np
from numba import jit

# ------------------------------------------------------
# Differential image quality metrics
# adapted from Royer et al. 2016, Nat Biotech, doi:10.1038/nbt.3708
# ------------------------------------------------------
@jit
def brenner(data):
    """
    Computes the Brenner score of an image and returns it
    :param data: image data
    :return: Brenner score
    """
    n_px_slice = (data.shape[0] - 2) * (data.shape[1] - 2)

    brenner = 0.0
    for y in range(1, data.shape[0] - 1): # skip first and last pixel in y
        for x in range(1, data.shape[1] - 1): # skip first and last pixel in x
            brenner += pow(data[y-1][x] - data[y+1][x], 2)
    
    return np.float32(brenner / n_px_slice)


@jit
def absolute_laplacian(data):
    """
    Computes the Absolute Laplacian of an image and returns it
    :param data: image data
    :return: Absolute Laplacian score
    """
    n_px_slice = (data.shape[0] - 2) * (data.shape[1] - 2)

    abs_laplacian = 0.0
    for y in range(1, data.shape[0] - 1):  # skip first and last pixel in y
        for x in range(1, data.shape[1] - 1):  # skip first and last pixel in x
            abs_laplacian += np.abs(2 * data[y][x] - data[y][x - 1] - data[y][x + 1]) + np.abs(2 * data[y][x] - data[y - 1][x] - data[y + 1][x])

    return np.float32(abs_laplacian / n_px_slice)


@jit
def squared_laplacian(data):
    """
    Computes the Squared Laplacian of an image and returns it
    :param data: image data
    :return: Squared Laplacian score
    """
    n_px_slice = (data.shape[0] - 2) * (data.shape[1] - 2)

    squared_laplacian = 0.0
    for y in range(1, data.shape[0] - 1):  # skip first and last pixel in y
        for x in range(1, data.shape[1] - 1):  # skip first and last pixel in x
            squared_laplacian += pow(8 * data[y][x] - data[y][x-1] - data[y][x+1] - data[y-1][x] - data[y+1][x] - data[y-1][x-1] - data[y+1][x+1] - data[y-1][x+1] - data[y+1][x-1], 2)

    return np.float32(squared_laplacian / n_px_slice)


@jit
def total_variation(data):
    """
    Computes the Total Variation of an image and returns it
    :param data: image data
    :return: Total Variation score
    """
    n_px_slice = (data.shape[0] - 2) * (data.shape[1] - 2)

    tv = 0.0
    for y in range(1, data.shape[0] - 1): # skip first and last pixel in y
        for x in range(1, data.shape[1] - 1): # skip first and last pixel in x
            tv += np.sqrt(pow(data[y][x + 1] - data[y][x - 1], 2) + pow(data[y + 1][x] - data[y - 1][x], 2))

    return np.float32(tv / n_px_slice)


@jit
def block_total_variation(data, block_size=8):
    """
    Computes the Block Total Variation of an image and returns it
    :param data: image data
    :return: Block Total Variation score
    """
    n_px_slice = (data.shape[0] - block_size) * (data.shape[1] - block_size)

    block_tv = 0.0
    for y in range((block_size/2), data.shape[0] - (block_size/2)): # skip first and last pixels equivalent to half the block size in y
        for x in range((block_size/2), data.shape[1] - (block_size/2)): # skip first and last pixels equivalent to half the block size in x
            for y_block in range(y-(block_size/2), y+(block_size/2)):
                for x_block in range(x-(block_size/2), x+(block_size/2)):
                    block_tv += np.sqrt(pow(data[y][x] - data[y_block][x_block], 2))

    return np.float32(block_tv / n_px_slice)


@jit
def tenengrad(data):
    """
    Computes the Tenengrad of an image and returns it
    :param data: image data
    :return: Tenengrad score
    """
    n_px_slice = (data.shape[0] - 2) * (data.shape[1] - 2)

    tenengrad = 0.0
    for y in range(1, data.shape[0] - 1):  # skip first and last pixel in y
        for x in range(1, data.shape[1] - 1):  # skip first and last pixel in x
            sobel_h = data[y-1][x+1] + 2 * data[y][x+1] + data[y+1][x+1] - data[y-1][x-1] - 2 * data[y][x-1] - data[y+1][x-1]
            sobel_v = data[y+1][x-1] + 2 * data[y+1][x] + data[y+1][x+1] - data[y-1][x-1] - 2 * data[y-1][x] - data[y-1][x+1]
            tenengrad += pow(sobel_h, 2) + pow(sobel_v, 2)

    return np.float32(tenengrad / n_px_slice)


# ------------------------------------------------------
# Correlative image quality metrics
# adapted from Royer et al. 2016, Nat Biotech, doi:10.1038/nbt.3708
# ------------------------------------------------------
@jit
def vollath_f4(data):
    """
    Computes the Vollath F4 score of an image and returns it
    :param data: image data
    :return: Vollath F4 score
    """
    n_px_slice = (data.shape[0] - 2) * (data.shape[1] - 3)

    vollath_f4 = 0.0
    for y in range(1, data.shape[0] - 1): # skip the first and the last pixel in y
        for x in range(1, data.shape[1] - 2):  # skip the first and the last two pixels in x
            vollath_f4 += data[y][x] * (data[y][x+1] - data[y][x+2])

    return np.float32(vollath_f4 / n_px_slice)


@jit
def vollath_f5(data):
    """
    Computes the Vollath F5 score of an image and returns it
    :param data: image data
    :return: Vollath F5 score
    """
    n_px_slice = (data.shape[0] - 2) * (data.shape[1] - 2)

    vollath_f5 = 0.0
    for y in range(1, data.shape[0] - 1):  # skip first and last two pixel in y
        for x in range(1, data.shape[1] - 1): # skip first and last pixel in x
            vollath_f5 += data[y][x] * data[y][x+1]
    
    sum = pow(np.sum(data), 2) / n_px_slice
    vollath_f5 -= sum

    return np.float32(vollath_f5 / n_px_slice)

@jit
def symmetric_vollath_f4(data):
    """
    Computes the Symmetric Vollath F4 score of an image and returns it
    :param data: image data
    :return: Symmetric Vollath F4 score
    """
    n_px_slice = (data.shape[0] - 2) * (data.shape[1] - 2)

    symmetric_vollath_f4 = 0.0
    for y in range(1, data.shape[0] - 1):  # skip the first and the last pixel in y
        for x in range(1, data.shape[1] - 2):  # skip the first and the last two pixels in x
            symmetric_vollath_f4 += np.abs(data[y][x] * (data[y][x+1] - data[y][x+2])) + \
                np.abs(data[y][x] * (data[y][x-1] - data[y][x-2])) + \
                np.abs(data[y][x] * (data[y+1][x] - data[y+2][x])) + \
                np.abs(data[y][x] * (data[y-1][x] - data[y-2][x]))

    return np.float32(symmetric_vollath_f4 / n_px_slice)

