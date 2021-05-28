import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from stereo_disparity_score import stereo_disparity_score
from stereo_disparity_fast import stereo_disparity_fast

# Load the stereo images and ground truth.
Il = imread("../stereo/cones/cones_image_02.png", as_gray = True)
Ir = imread("../stereo/cones/cones_image_06.png", as_gray = True)

# The cones and teddy datasets have ground truth that is 'stretched'
# to fill the range of available intensities - here we divide to undo.
It = imread("../stereo/cones/cones_disp_02.png",  as_gray = True)/4.0

# Load the appropriate bounding box.
bbox = np.load("../data/cones_02_bounds.npy")

import time
t = time.time()
Id = stereo_disparity_fast(Il, Ir, bbox, 55)
print(time.time()-t)
N, rms, pbad = stereo_disparity_score(It, Id, bbox)
print("Valid pixels: %d, RMS Error: %.2f, Percentage Bad: %.2f" % (N, rms, pbad))
plt.imshow(Id, cmap = "gray")
plt.show() 