import cv2
import numpy as np


def disp_map(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


    block_size = 11
    min_disp = 1
    max_disp = 16 * 12

    num_disp = max_disp - min_disp

    uniquenessRatio = 5

    speckleWindowSize = 170

    speckleRange = 1
    disp12MaxDiff = 5
    P1=3*block_size*block_size
    P2=55*block_size*block_size

    stereo = cv2.StereoSGBM_create(
       minDisparity=min_disp,
       numDisparities=num_disp,
       blockSize=block_size,
       uniquenessRatio=uniquenessRatio,
       speckleWindowSize=speckleWindowSize,
       speckleRange=speckleRange,
       disp12MaxDiff=disp12MaxDiff,
       P1=P1, 
       P2=P2
    )   

    #stereo  = cv2.StereoSGBM_create(numDisparities=135, blockSize=21)

    disparity_SGBM = stereo.compute(img1, img2) 
    print(np.max(disparity_SGBM))
    print(np.min(disparity_SGBM))
    
    #disparity_SGBM = cv2.normalize(disparity_SGBM, disparity_SGBM, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
    #disparity_SGBM = np.uint16(disparity_SGBM)

    return disparity_SGBM