import cv2
import numpy as np
from gstreamer.gstreamer_base_code import __gstreamer_pipeline
from rectification.stereo_rectification_calibrated import stereo_rectification_calibrated

#initialise camera sreams
#cam1 = cv2.VideoCapture(__gstreamer_pipeline(camera_id=1, flip_method=0), cv2.CAP_GSTREAMER)
#cam2 = cv2.VideoCapture(__gstreamer_pipeline(camera_id=0, flip_method=0), cv2.CAP_GSTREAMER)

#get stere rectification params
maps_left_cam, maps_right_cam, ROI1, ROI2 = stereo_rectification_calibrated()

print(np.asarray(maps_left_cam[:][:][:][0]).shape)