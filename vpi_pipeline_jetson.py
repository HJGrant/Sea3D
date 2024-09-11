import cv2
import numpy as np
from gstreamer.gstreamer_base_code import __gstreamer_pipeline
#from rectification.stereo_rectification_calibrated import stereo_rectification_calibrated
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

def normalize_for_display(depth_map):
    # Normalize depth map to 0-255 for visualization
    min_val = np.min(depth_map)
    max_val = np.max(depth_map)
    norm_depth_map = 65535 * (depth_map - min_val) / (max_val - min_val)
    return norm_depth_map.astype(np.uint16)


def vpi_compute_disp(left_frame, right_frame, depth_path, out_path, disparity_path):
    maxDisparity = 255
    min_disp = 1         #original 16
    block_size = 26           #original 8
    uniquenessRatio = 4       #original 1
    quality = 8
    p1 = 130
    p2 = 190
    numPasses=3

    scale=1
    downscale=0.5

    f=1075.9
    B=107.9599

    #initialise 2 streams for reading and preprocessing of frames
    streamLeft = vpi.Stream()
    streamRight = vpi.Stream()


    #TODO: VPI STAGE 2: PROCESSING LOOP
    #while True:
    with vpi.Backend.CUDA:   #or CUDA
        with streamLeft:
            #frame2 = cv2.imread("frame_04950_left.png")
            #ret1, frame1 = cam1.read()
            left_frame = cv2.remap(left_frame, maps_left_cam[0], maps_left_cam[1], cv2.INTER_LANCZOS4)
            left_frame = left_frame[ROI1[1]:ROI1[3], ROI1[0]:ROI1[2]] #minus 1 to set shape to same dimensions TODO: solve this better
            left = vpi.asimage(np.asarray(left_frame)).convert(vpi.Format.Y16_ER, scale=scale)
        with streamRight:
            #ret2, frame2 = cam2.read()
            #frame1 = cv2.imread("frame_04950_right.png")
            right_frame = cv2.remap(right_frame, maps_right_cam[0], maps_right_cam[1], cv2.INTER_LANCZOS4)
            right_frame = right_frame[ROI1[1]:ROI1[3], ROI1[0]:ROI1[2]]
            right = vpi.asimage(np.asarray(right_frame)).convert(vpi.Format.Y16_ER, scale=scale)

    #with vpi.Backend.CUDA:
    #    with streamLeft:
    #        left_1 = left.convert(vpi.Format.Y16_ER_BL)
    #    with streamRight:
    #        right_1 = right.convert(vpi.Format.Y16_ER_BL)


    #get output width and height
    outWidth = (left.size[0] + downscale - 1) // downscale
    outHeight = (left.size[1] + downscale - 1) // downscale

    streamStereo = streamLeft

    #estimate stereo disparity
    with streamStereo, vpi.Backend.CUDA:
        disparityS16 = vpi.stereodisp(left, right, window=block_size, maxdisp=maxDisparity, mindisp=min_disp, 
                                    quality=quality, uniqueness=uniquenessRatio, includediagonals=True, numpasses=numPasses, p1=p1, p2=p2)

    if disparityS16.format == vpi.Format.S16_BL:
                disparityS16 = disparityS16.convert(vpi.Format.S16, backend=vpi.Backend.CUDA)

    with streamStereo, vpi.Backend.CUDA:
        #scale and convert disparity map
        disparityU16 = disparityS16.convert(vpi.Format.U16, scale=65535.0 / (32 * maxDisparity)).cpu()
     
        left = left.convert(vpi.Format.U8).cpu()
        right = right.convert(vpi.Format.U8).cpu()

        side_by_side_img = np.hstack((left_frame, right_frame))

        # Draw horizontal lines
        stereo_uncalib_w_lines = draw_horizontal_lines(side_by_side_img)
        
        depth_map = calculate_depth(disparityU16, f, B)

        normalized_depth_map = normalize_for_display(depth_map)

        #cv2.imshow('STEREO', stereo_uncalib_w_lines)
        #cv2.imshow('DISPARITY', disparityU8)
        #cv2.moveWindow('LEFT FRAME UDIST', 100, 250)
        #cv2.moveWindow('RIGHT FRAME UDIST', 1100, 250)
        #cv2.moveWindow('DISPARITY', 100, 850)
        cv2.imwrite(depth_path, normalized_depth_map)
        cv2.imwrite(out_path, left_frame)
        cv2.imwrite(disparity_path, disparityU16)
        #cv2.imwrite("disparity_map_color.png", disparityColor)
        cv2.imwrite('rectified_stereo.png', stereo_uncalib_w_lines)
        #cv2.imwrite("left_rectified.png", img_left)
        #cv2.imwrite("rright_rectified.png", img_right)
        
    #cv2.waitKey()
    #if cv2.waitKey(1)==ord('q'):
    #    break


    #cam1.release()
    #cam2.release()
    cv2.destroyAllWindows()