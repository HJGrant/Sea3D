import cv2
import numpy as np
from vpi_pipeline import vpi_pipeline
from stereo_rectification_calibrated import stereo_rectification_calibrated
import vpi
import os
from tqdm import tqdm

if __name__ == "__main__":

    left_image = cv2.imread("19:02:26.153_left.jpg")
    right_image = cv2.imread("19:02:26.153_right.jpg")

    depth_map, color, disparity, stereo = vpi_pipeline(left_image, right_image)


    depth_map = cv2.resize(depth_map, (960, 480), interpolation=cv2.INTER_LINEAR)
    disparity = cv2.resize(disparity, (960, 480), interpolation=cv2.INTER_LINEAR)
    color = cv2.resize(color, (960, 480), interpolation=cv2.INTER_LINEAR)
    
    cv2.imshow('STEREO', stereo)
    cv2.imshow('DISPARITY', disparity)
    cv2.imshow("DEPTH", depth_map)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()