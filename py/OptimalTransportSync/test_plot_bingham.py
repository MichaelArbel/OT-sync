
# You need to install matlab with python bindings!
import matlab.engine
eng = matlab.engine.start_matlab()

import numpy as np
import cv2

import bingham.plot_help as db

eng.addpath('bingham', nargout=0)
eng.addpath('bingham/tools', nargout=0)
eng.addpath('bingham/visualization', nargout=0)

# The quality of the rendering. It is super slow so for testing I always set it to 50 and for the final renderings back to 400!
quality = 50

# Each element of the list is a list of quaternions for the bingham distribution
distributions = []

# random distributions made of quaternions for testing
bingham_1 = [[0.7, 0.3, -0.1, 0.2], [0.6, 0.35, -0.16, 0.15]]
bingham_2 = [[0.5, 0.2, -0.4, -0.2], [0.4, 0.1, -0.4, -0.2]]

# normalize quats
bingham_1 = np.asarray(bingham_1)
bingham_2 = np.asarray(bingham_2)

bingham_1 /= np.linalg.norm(bingham_1, axis=1, keepdims=True)
bingham_2 /= np.linalg.norm(bingham_2, axis=1, keepdims=True)

# adding the distributions to the list and convert them to matlab
distributions.append(matlab.double(bingham_1.tolist()))
distributions.append(matlab.double(bingham_2.tolist()))


# GT if not set to None then a green cross will be superimposed at this position as GT quaternion
# gt = [q_gt, -q_gt]
gt = np.asarray([[0.7, 0.3, -0.1, 0.2], [-0.7, -0.3, 0.1, -0.2]])
gt /= np.linalg.norm(gt, axis=1, keepdims=True)

# plotting!
bingham = db.get_bingham(eng, distributions, GT=None, precision=quality)  / 255. # without ground truth
bingham_gt = db.get_bingham(eng, distributions, GT=gt, precision=quality) / 255. # with ground truth

# show on display with and without gt
cv2.imshow('bingham', cv2.hconcat([bingham, bingham_gt]))
cv2.waitKey(0)