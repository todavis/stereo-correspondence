# stereo-correspondence

## Overview
The goal of stereo correspondence is to produce a dense depth map. The algorithm produces a disparity between pixel matches and is inversely proportional to the depth. The first task is to implement an efficient local stereo correspondence algorithm. The second task is to improved the performance of the correspondence algorithm.

## Implemenation Details - fast correspondence
A fixed window matching routine using the sum-of-absolute-difference (SAD) as the similarity measure is selected. A winner-takes-all strategy is implemented to select the best match. The function accepts an image pair (greyscale, e.g., two example images in the stereo directory) and produces a disparity image (map). The function accepts a bounding box that indicates the valid overlap region (which we will supply), to avoid attempting to match points that are not visible in both images. Finally, the makes use of a maximum disparity parameter (which is supplied) to bound the 1D search.

The window size used by the SAD algorithm is adjusted via some experimentation to achieve the best performance. The RMS error between a ground truth disparity image and the computed images is used to evaluate performance. 

## Implemenation Details - best correspondence

The improvement to performance can be done by adding filtering to the naive fast correspondence. This function implements improvements to a local method. Global correspondence algorithms can also be used for this task.

Real-time correlation-based stereo vision with reduced border errors:

* Local method to maintain speed while improving quality
* Improves on the correlation based SAD approach of fast method

Implementation of Algorithm from:
* Hirschmuller, H. 2002. Real-time correlation-based stereo vision with reduced border errors. IJCV, 47(1/2/3):229–246.

The general method is as follows:

1. Compute correlations for each disparity using SAD, 1D search, window

2. Multiple supporting windows are used to validate correlation readings
   A number of surrounding readings are compared and the best are 
   added to the correlation value

3. Reject general errors - if best disparity correlation is close to
   the second best we have uncertainty so we reject this reading and
   fill in with the nearest valid reading   
