import numpy as np
from scipy.ndimage.filters import *

def stereo_disparity_fast(Il, Ir, bbox, maxd):
    """
    Fast stereo correspondence algorithm.

    This function computes a stereo disparity image from left stereo 
    image Il and right stereo image Ir. Only disparity values within
    the bounding box region (inclusive) are evaluated.

    Parameters:
    -----------
    Il    - Left stereo image, m x n pixel np.array, greyscale.
    Ir    - Right stereo image, m x n pixel np.array, greyscale.
    bbox  - 2x2 np.array, bounding box, relative to left image, from top left
            corner to bottom right corner (inclusive). (x, y) in columns.
    maxd  - Integer, maximum disparity value; disparities must be within zero
            to maxd inclusive (i.e., don't search beyond maxd).

    Returns:
    --------
    Id  - Disparity image (map) as np.array, same size as Il.
    """
    # Hints:
    #
    #  - Loop over each image row, computing the local similarity measure, then
    #    aggregate. At the border, you may replicate edge pixels, or just avoid
    #    using values outside of the image.
    #
    #  - You may hard-code any parameters you require in this function.
    #
    #  - Use whatever window size you think might be suitable.
    #
    #  - Optimize for runtime AND for clarity.

    #--- FILL ME IN ---

    window_size = 16; # < 20 for this bbox
    t = window_size // 2 + 1
    
    Id = np.zeros(Il.shape)
    n = Il.shape[1]
    
    SAD_matrix = np.full((Il.shape[0], n, maxd + 1), np.inf)
    
    # loop over all d - search in one direction
    for d in range(1, maxd + 1):
            
        # calculate single image for image difference at d
        Idiff = np.abs(Il[:, d : n] - Ir[:, : n - d])
                       
        # Integral image calculation for efficient window summation
        Iint = Idiff.cumsum(axis=0).cumsum(axis=1)
        
        # sum over all windows using integral image calculation
        SAD_matrix[bbox[1, 0]:bbox[1, 1] + 1, t + d: -t, d] = \
            Iint[bbox[1,0] + t: bbox[1,1] + t + 1,  2 * t:] + \
            Iint[bbox[1,0] - t: bbox[1,1] - t + 1,  : - 2 * t] - \
            Iint[bbox[1,0] - t: bbox[1,1] - t + 1,  2 * t:] - \
            Iint[bbox[1,0] + t: bbox[1,1] + t + 1,  : - 2 * t]

    # find disparity with minimum correlation
    Id = SAD_matrix.argmin(2)
    
    #------------------

    correct = isinstance(Id, np.ndarray) and Id.shape == Il.shape

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Id