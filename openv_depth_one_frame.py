import cv2
import numpy as np
from gstreamer.gstreamer_base_code import __gstreamer_pipeline
from stereo_rectification_calibrated import stereo_rectification_calibrated
#from stereo_rectification_uncalibrated import stereo_rectification_uncalibrated
#from undistort_only import undistort_only
from depth_map import depth_map
import matplotlib.pyplot as plt

def draw_horizontal_lines(img, num_lines=20, color=(0, 255, 0), thickness=1):
    height = img.shape[0]
    step = height // num_lines
    for i in range(0, height, step):
        cv2.line(img, (0, i), (img.shape[1], i), color, thickness)
    return img

def display_stereo_images(side_by_side_img):
    cv2.imshow('Rectified Stereo Pair', side_by_side_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



#get rectification maps
maps_left_cam, maps_right_cam, ROI1, ROI2 = stereo_rectification_calibrated()

frame2 = cv2.imread("frame_04950_left.png", cv2.IMREAD_GRAYSCALE)
frame1 = cv2.imread("frame_04950_right.png", cv2.IMREAD_GRAYSCALE)

#h1, w1 = frame1.shape
#h2, w2 = frame2.shape
#H1, H2 = stereo_rectification_uncalibrated(frame2, frame1)

#remap images based on the maps recieved from stereoRectify() and initUndistortRectifyMap() from stereo_rectification_calibrated()
left_frame_rectified = cv2.remap(frame1, maps_left_cam[0], maps_left_cam[1], cv2.INTER_LANCZOS4)
right_frame_rectified = cv2.remap(frame2, maps_right_cam[0], maps_right_cam[1], cv2.INTER_LANCZOS4)

#left_frame_rectified_uncalib = cv2.warpPerspective(frame1, H1, (w1, h1))
#right_frame_rectified_uncalib = cv2.warpPerspective(frame2, H2, (w2, h2))

#set the ROI for both images
#left_frame_rectified = left_frame_rectified[ROI1[1]:ROI1[3]-1, ROI1[0]:ROI1[2]] #minus 1 to set shape to same dimensions TODO: solve this better
#right_frame_rectified = right_frame_rectified[ROI2[1]:ROI2[3], ROI2[0]:ROI2[2]]

# Concatenate images side by side
#left_frame_rectified = cv2.resize(left_frame_rectified, (960,480))
#right_frame_rectified = cv2.resize(right_frame_rectified, (960,480))

#left_frame_rectified_uncalib = cv2.resize(left_frame_rectified_uncalib, (960,480))
#right_frame_rectified_uncalib = cv2.resize(right_frame_rectified_uncalib, (960,480))

#side_by_side_uncalib = np.hstack((left_frame_rectified_uncalib, right_frame_rectified_uncalib))
side_by_side_img = np.hstack((left_frame_rectified, right_frame_rectified))

# Draw horizontal lines
#stereo_uncalib_w_lines = draw_horizontal_lines(side_by_side_uncalib)
side_by_side_img_with_lines = draw_horizontal_lines(side_by_side_img)

#create a depth map based on the rectified images
#disp_uncalib = depth_map(left_frame_rectified_uncalib, right_frame_rectified_uncalib)
disparity = depth_map(left_frame_rectified, right_frame_rectified)


cv2.imshow('stereo_rectified',side_by_side_img_with_lines)
#cv2.imshow('stereo_uncalib', stereo_uncalib_w_lines)
#cv2.imshow("disp_uncalib", disp_uncalib)
cv2.imshow('DISPARITY', disparity)
cv2.moveWindow('stereo_rectified', 100, 250)
cv2.moveWindow('DISPARITY', 100, 950)

cv2.imwrite('rectified_stereo_no_dist_coeffs.png', side_by_side_img_with_lines)
cv2.imwrite('distaprity_no_dist_coeffs.png', disparity)

cv2.waitKey(0)