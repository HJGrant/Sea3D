import cv2
import numpy as np
from gstreamer.gstreamer_base_code import __gstreamer_pipeline
from stereo_rectification_calibrated import stereo_rectification_calibrated
import vpi
import os
from tqdm import tqdm

def draw_horizontal_lines(img, num_lines=20, color=(0, 255, 0), thickness=1):
    height = img.shape[0]
    step = height // num_lines
    for i in range(0, height, step):
        cv2.line(img, (0, i), (img.shape[1], i), color, thickness)
    return img

maps_left_cam, maps_right_cam, ROI1, ROI2 = stereo_rectification_calibrated()

def calculate_depth(disparity_map, f, B):
    disparity_map[disparity_map == 0] = 0.1
    depth_map = (f * B) / disparity_map
    return depth_map

def normalize_disparity(disparity_map, min_disp, max_disp):
    disparity_map = np.clip(disparity_map, min_disp, max_disp)
    normalized_disp_map = 255 * (disparity_map - min_disp) / (max_disp - min_disp)
    return normalized_disp_map.astype(np.uint8)

def vpi_pipeline(left_frame, right_frame, min_depth, max_depth, f, B_meters, median=False, bilateral=False):
    max_disparity = 245
    min_disparity = 2
    block_size = 7
    uniquenessRatio = 0.99
    quality = 8
    P1=175
    P2=195
    p2alpha = 4
    numPasses=2
    confthreshold=60000

    scale = 1
    downscale = 0.5

    streamLeft = vpi.Stream()
    streamRight = vpi.Stream()

    with vpi.Backend.CUDA:
        with streamLeft:
            left_frame = cv2.remap(left_frame, maps_left_cam[0], maps_left_cam[1], cv2.INTER_LANCZOS4)
            left_frame = left_frame[ROI1[1]:ROI1[3], ROI1[0]:ROI1[2]]
            left = vpi.asimage(np.asarray(left_frame)).convert(vpi.Format.Y16_ER, scale=scale)
        with streamRight:
            right_frame = cv2.remap(right_frame, maps_right_cam[0], maps_right_cam[1], cv2.INTER_LANCZOS4)
            right_frame = right_frame[ROI1[1]:ROI1[3], ROI1[0]:ROI1[2]]
            right = vpi.asimage(np.asarray(right_frame)).convert(vpi.Format.Y16_ER, scale=scale)

    outWidth = (left.size[0] + downscale - 1) // downscale
    outHeight = (left.size[1] + downscale - 1) // downscale

    streamStereo = streamLeft

    with streamStereo, vpi.Backend.CUDA:
        disparityU16 = vpi.stereodisp(left, right, window=block_size, maxdisp=max_disparity,confthreshold=confthreshold, mindisp=min_disparity, 
                                    quality=quality, uniqueness=uniquenessRatio, includediagonals=True, numpasses=numPasses, p1=P1, p2=P2, p2alpha=p2alpha)
        
        if median == True:
            disparityU16 = disparityU16.median_filter((3, 3))
        
        if bilateral == True:
            disparityU16 = disparityU16.bilateral_filter(7, 0.5, 0.1)

    with streamStereo, vpi.Backend.CUDA:
        disparityU16 = disparityU16.cpu()

        # Check disparity values
        print(f"Max disparity: {np.max(disparityU16)}")
        print(f"Min disparity: {np.min(disparityU16)}")

        # Calculate depth map
        depth_map = calculate_depth(disparityU16, f, B_meters)

        # Apply depth thresholding
        depth_map[depth_map < min_depth] = min_depth
        depth_map[depth_map > max_depth] = max_depth

        side_by_side_img = np.hstack((left_frame, right_frame))
        stereo_uncalib_w_lines = draw_horizontal_lines(side_by_side_img)

        # Validate depth map values
        print(f"Max depth: {np.max(depth_map)}")
        print(f"Min depth: {np.min(depth_map)}")

        print(f"Depth type: {depth_map.dtype}")
        print(f"Disp type: {disparityU16.dtype}")

        #depth_map = normalize_disparity(depth_map, min_disparity, max_disparity)

        return depth_map, left_frame, disparityU16, stereo_uncalib_w_lines
