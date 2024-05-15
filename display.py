import cv2
import numpy as np
import time
from gstreamer.gstreamer_base_code import __gstreamer_pipeline
from rectification.stereo_rectification_calibrated import stereo_rectification_calibrated
from depth_map.vpi_depth_map import vpi_stereo
import matplotlib.pyplot as plt
print(cv2.__version__)

#initialise video capture object   
cam1 = cv2.VideoCapture(__gstreamer_pipeline(camera_id=1, flip_method=0), cv2.CAP_GSTREAMER)
cam2 = cv2.VideoCapture(__gstreamer_pipeline(camera_id=0, flip_method=0), cv2.CAP_GSTREAMER)

#check if video capture object was properly initialised and able to open
if not cam1.isOpened():
 print("Cannot open camera 1")
 exit()

if not cam2.isOpened():
 print("Cannot open camera 2")
 exit()

#get rectification maps from calibration params
maps_left_cam, maps_right_cam, ROI1, ROI2 = stereo_rectification_calibrated()

while True:
    ret1, frame1 = cam1.read()
    ret2, frame2 = cam2.read()

    #remap images based on the maps recieved from stereoRectify() and initUndistortRectifyMap() from stereo_rectification_calibrated()
    left_frame_rectified = cv2.remap(frame1, maps_left_cam[0], maps_left_cam[1], cv2.INTER_LANCZOS4)
    right_frame_rectified = cv2.remap(frame2, maps_right_cam[0], maps_right_cam[1], cv2.INTER_LANCZOS4)
 
    #set the ROI for both images
    #left_frame_rectified = left_frame_rectified[ROI1[1]:ROI1[3], ROI1[0]:ROI1[2]] #minus 1 to set shape to same dimensions TODO: solve this better
    #right_frame_rectified = right_frame_rectified[ROI2[1]:ROI2[3], ROI2[0]:ROI2[2]]

    #create a depth map based on the rectified images
    #disparity = vpi_stereo(left_frame_rectified, right_frame_rectified)

    cv2.imshow('LEFT FRAME UDIST',left_frame_rectified)
    cv2.imshow('RIGHT FRAME UDIST', right_frame_rectified)
    #cv2.imshow('DISPARITY', disparity)
    cv2.moveWindow('LEFT FRAME UDIST', 100, 250)
    cv2.moveWindow('RIGHT FRAME UDIST', 1100, 250)
    #cv2.moveWindow('DISPARITY', 100, 850)
     
    if cv2.waitKey(1)==ord('q'):
        break

#close video capture object and close opencv window   
cam1.release()
cam2.release()
cv2.destroyAllWindows()