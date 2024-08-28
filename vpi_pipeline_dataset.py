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

#TODO: VPI STAGE 1: INITIALIZATION
#data (e.g. numpy areas) need to be wrapped in a VPI image, which can then be used for further processing

#initialise camera sreams
#cam1 = cv2.VideoCapture(__gstreamer_pipeline(camera_id=1, flip_method=0), cv2.CAP_GSTREAMER)
#cam2 = cv2.VideoCapture(__gstreamer_pipeline(camera_id=0, flip_method=0), cv2.CAP_GSTREAMER)

#get stere rectification params
maps_left_cam, maps_right_cam, ROI1, ROI2 = stereo_rectification_calibrated()
#warp_left = vpi.WarpMap(vpi.WarpGrid((1920,1080)))
#maps_left_cam_t = maps_left_cam.transpose(2, 1, 0)
#wx_l, wy_l = np.asarray(warp_left).transpose(2,1,0)
#wx_l = maps_left_cam_t[0]
#wy_l = maps_left_cam_t[1]

#warp_right = vpi.WarpMap(vpi.WarpGrid((1920,1080)))
#maps_right_cam_t = maps_right_cam.transpose(2, 1, 0)
#wx_r, wy_r = np.asarray(warp_right).transpose(2,1,0)
#wx_r = maps_right_cam_t[0]
#wy_r = maps_right_cam_t[1]

def calculate_depth(disparity_map, f, B):
    # Avoid division by zero
    disparity_map[disparity_map == 0] = 0.1
    # Compute depth
    depth_map = (f * B) / disparity_map
    return depth_map

def normalize_for_display(depth_map):
    # Normalize depth map to 0-255 for visualization
    min_val = np.min(depth_map)
    max_val = np.max(depth_map)
    norm_depth_map = 65535 * (depth_map - min_val) / (max_val - min_val)
    return norm_depth_map.astype(np.uint16)


def vpi_compute_disp(left_frame, right_frame, depth_path, out_path):
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

if __name__ == "__main__":
    # Paths to the folders containing the stereo images
    left_folder = '/media/seaclear/639f8c93-fac2-4b84-a4fe-b261541674e9/lab_tests/lab_test_2808/28-08-2024_16_15_06/left'
    right_folder = '/media/seaclear/639f8c93-fac2-4b84-a4fe-b261541674e9/lab_tests/lab_test_2808/28-08-2024_16_15_06/right'
    depth_folder = '/media/seaclear/639f8c93-fac2-4b84-a4fe-b261541674e9/lab_tests/lab_test_2808/28-08-2024_16_15_06/depth_vpi'
    out_img_folder = '/media/seaclear/639f8c93-fac2-4b84-a4fe-b261541674e9/lab_tests/lab_test_2808/28-08-2024_16_15_06/color'

    # List all files in the left and right folders
    left_images = sorted(os.listdir(left_folder))
    right_images = sorted(os.listdir(right_folder))

    for i in tqdm(range(len(left_images))):
        depth_path = os.path.join(depth_folder, str(i).zfill(6)+'.png')
        out_path = os.path.join(out_img_folder, str(i).zfill(6)+'.jpg')

        # Load the left and right images
        left_image_path = os.path.join(left_folder, left_images[i])
        right_image_path = os.path.join(right_folder, right_images[i])

        left_image = cv2.imread(left_image_path)
        right_image = cv2.imread(right_image_path)

        if left_image is None or right_image is None:
            print(f"Skipping pair {i}: Unable to load one of the images.")
            continue
        
        vpi_compute_disp(left_image, right_image, depth_path, out_path)
        