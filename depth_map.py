import cv2
import numpy as np


def depth_map(img1, img2):
    block_size = 7
    min_disp = -40
    max_disp = 125

    num_disp = max_disp - min_disp

    uniquenessRatio = 10

    speckleWindowSize = 70

    speckleRange = 2
    disp12MaxDiff = 0

    stereo = cv2.StereoSGBM_create(
       minDisparity=min_disp,
       numDisparities=num_disp,
       blockSize=block_size,
       uniquenessRatio=uniquenessRatio,
       speckleWindowSize=speckleWindowSize,
       speckleRange=speckleRange,
       disp12MaxDiff=disp12MaxDiff,
       P1=8 * 1 * block_size * block_size, 
       P2=32 * 1 * block_size * block_size
    )   

    disparity_SGBM = stereo.compute(img1, img2) 
    disparity_SGBM = cv2.normalize(disparity_SGBM, disparity_SGBM, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX)
    disparity_SGBM = np.uint8(disparity_SGBM)

    return disparity_SGBM