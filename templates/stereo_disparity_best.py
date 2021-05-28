import numpy as np
from scipy.ndimage.filters import *

def stereo_disparity_best(Il, Ir, bbox, maxd):
    """
    Best stereo correspondence algorithm.

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

    #--- FILL ME IN ---
    '''
    Algorithm from:
        Hirschmu ̈ller, H. 2002. Real-time correlation-based stereo vision with 
        reduced border errors. IJCV, 47(1/2/3):229–246.
    
    Real-time correlation-based stereo vision with reduced border errors:
        
        - Local method to maintain speed while improving quality
        - Improves on the correlation based SAD approach of fast method
        
        The general method is as follows:
        
            1. Compute correlations for each disparity using SAD, 1D search, window
            
            2. Multiple supporting windows are used to validate correlation readings
               A number of surrounding readings are compared and the best are 
               added to the correlation value
               
            3. Reject general errors - if best disparity correlation is close to
               the second best we have uncertainty so we reject this reading and
               fill in with the nearest valid reading   
    '''
    
    
    # 1. SAD based approach for calculating all pixel correlations
    window_size = 16; # < 20 for this BBox
    t = window_size // 2 + 1
    
    Id = np.zeros(Il.shape)
    n = Il.shape[1]
    m = Il.shape[0]
    SAD_matrix = np.full((m, n, 2 * maxd + 1), np.inf)
    
    # loop over all disparity d
    for d in range(- maxd, maxd + 1):
        
        if d == 0:
            continue
            
        # calculate single image for all differences at d
        Idiff = np.abs(Il[:, np.max((0, d)): n + np.min((0, d))] - \
                       Ir[:, np.max((0, - d)): n - np.max((0, d))])
                       
        # Integral image calculation for efficient window summation
        Iint = Idiff.cumsum(axis=0).cumsum(axis=1)
        
        # sum over all windows using integral image
        SAD_matrix[bbox[1, 0]:bbox[1, 1] + 1, \
                   t + np.max((0, d)): -t - np.max((0, -d)), d + maxd] = \
            Iint[bbox[1,0] + t: bbox[1,1] + t + 1,  2 * t:] + \
            Iint[bbox[1,0] - t: bbox[1,1] - t + 1,  : - 2 * t] - \
            Iint[bbox[1,0] - t: bbox[1,1] - t + 1,  2 * t:] - \
            Iint[bbox[1,0] + t: bbox[1,1] + t + 1,  : - 2 * t]

    # 2. Compute combined correlation for supporting windows
    w = 7 # window size
    SAD_matrix_comb = SAD_matrix.copy()

    for d in range(0, 2 * maxd + 1):
        
        # initialize supporting values
        Cout = np.full((m, n, 4), np.inf)
        
        # find supporting values in this window
        Cout[w:-w , w:-w , 0] = SAD_matrix[2 * w:  , 2 * w: , d]
        Cout[w:-w , w:-w , 1] = SAD_matrix[2 * w:  , :-2 * w, d]
        Cout[w:-w , w:-w , 2] = SAD_matrix[:-2 * w , 2 * w: , d]
        Cout[w:-w , w:-w , 3] = SAD_matrix[:-2 * w , :-2 * w, d]
        
        # take the minimum of the supporting values
        Cout.sort(axis = 2)
        Cout[Cout == np.inf] = 0 

        # sum each pixel correlation with 2 best supporting values
        SAD_matrix_comb[:,:, d] = np.sum(np.concatenate((SAD_matrix[:,:,[d]], Cout[:,:,:2]),
                                                        axis = 2),
                                         axis = 2)
    
    # 3. Reject disparities which have uncertainty in correlation
    # reject with Cd score above C_thresh
    C_thresh = 0.03
    
    # find minimum correlations too close to second smallest
    Icorr = np.partition(SAD_matrix_comb, 1, axis =2)[:,:,:2]
    Icorr[Icorr == np.inf] = 10e-2
    Icorr[Icorr == 0] = 10e-2
    Icorr_cd = (Icorr[:,:,1] - Icorr[:,:,0])/Icorr[:,:,0] 
    
    # mask of valid disparity readings above threshold
    mask = Icorr_cd > C_thresh 
    
    # Create disparity map with indices of minimums
    disparity_vals = np.arange(- maxd, maxd + 1)
    Id_unfiltered = np.abs(disparity_vals[np.argmin(SAD_matrix_comb, axis =2)])
    
    # For each invalid disparity, use the minimum value of the nearest valid disparity
    Id = Id_unfiltered.copy()
    
    for y in range(bbox[1, 0], bbox[1, 1]):
        
        # in an invalid region with minimum disparity at inf
        in_invalid = True
        size_invalid = 0
        left_d = np.inf
        
        for x in range(1, n - 1):
            
            if mask[y, x]:
                # reaches a valid reading, save this value
                left_d = Id[y, x]
                continue
            
            if in_invalid and (mask[y, x + 1] or x == n - 2):
                # next reading is valid, take minimum of left and right readings
                Id[y, x - size_invalid:x + 1] = np.min((left_d, Id[y, x + 1]))
                size_invalid = 0
                in_invalid = False
            
            elif in_invalid and not mask[y, x + 1]:
                # still in region of invalid readings
                in_invalid = True
                size_invalid += 1
                
            else:
                # enter an invalid region
                in_invalid = True
                size_invalid = 1
                left_d = Id[y, x - 1]

    #------------------

    correct = isinstance(Id, np.ndarray) and Id.shape == Il.shape

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Id