import cv2
import numpy as np
from gstreamer.gstreamer_base_code import __gstreamer_pipeline
from rectification.stereo_rectification_calibrated import stereo_rectification_calibrated
import vpi

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

maxDisparity = 256
min_disp = 0         #original 16
block_size = 2            #original 8
uniquenessRatio = 0       #original 1
quality = 8
p1 = 5
p2 = 210
numPasses=3

scale=1
downscale=1

#initialise 2 streams for reading and preprocessing of frames
streamLeft = vpi.Stream()
streamRight = vpi.Stream()


#TODO: VPI STAGE 2: PROCESSING LOOP
#while True:
with vpi.Backend.CUDA:   #or CUDA
    with streamLeft:
        img_left = cv2.imread("left_image_0.png")
        #ret1, frame1 = cam1.read()
        img_left = cv2.remap(img_left, maps_left_cam[0], maps_left_cam[1], cv2.INTER_LANCZOS4)
        left = vpi.asimage(np.asarray(img_left)).convert(vpi.Format.Y16_ER, scale=scale)
    with streamRight:
        #ret2, frame2 = cam2.read()
        img_right = cv2.imread("right_image_0.png")
        img_right = cv2.remap(img_right, maps_right_cam[0], maps_right_cam[1], cv2.INTER_LANCZOS4)
        right = vpi.asimage(np.asarray(img_right)).convert(vpi.Format.Y16_ER, scale=scale)

with vpi.Backend.VIC:
    with streamLeft:
        left_1 = left.convert(vpi.Format.Y16_ER_BL)
    with streamRight:
        right_1 = right.convert(vpi.Format.Y16_ER_BL)


#get output width and height
outWidth = (left.size[0] + downscale - 1) // downscale
outHeight = (left.size[1] + downscale - 1) // downscale

#use left stream to consolidate actual stereo processing
streamStereo = streamLeft

#estimate stereo disparity
with streamStereo, vpi.Backend.OFA:
    disparityS16 = vpi.stereodisp(left_1, right_1, window=block_size, maxdisp=maxDisparity, mindisp=min_disp, 
                                   quality=quality, uniqueness=uniquenessRatio, includediagonals=False, numpasses=numPasses, p1=p1, p2=p2)

#TODO: VPI STAGE 3: CLEANUP
#must convert to pitch-linear if block-linear format
if disparityS16.format == vpi.Format.S16_BL:
            disparityS16 = disparityS16.convert(vpi.Format.S16, backend=vpi.Backend.VIC)

with streamStereo, vpi.Backend.CUDA:
    #scale and convert disparity map
    disparityU8 = disparityS16.convert(vpi.Format.U8, scale=255.0/(32*maxDisparity)).cpu()

    #convert to color JET map
    disparityColor = cv2.applyColorMap(disparityU8, cv2.COLORMAP_JET)
    
    left = left.convert(vpi.Format.U8).cpu()
    right = right.convert(vpi.Format.U8).cpu()

    cv2.imshow('LEFT FRAME UDIST',img_left)
    cv2.imshow('RIGHT FRAME UDIST', img_right)
    cv2.imshow('DISPARITY', disparityU8)
    cv2.moveWindow('LEFT FRAME UDIST', 100, 250)
    cv2.moveWindow('RIGHT FRAME UDIST', 1100, 250)
    cv2.moveWindow('DISPARITY', 100, 850)
    cv2.imwrite("disparity_map.png", disparityU8)
    cv2.imwrite("disparity_map_color.png", disparityColor)
    cv2.imwrite("left_rectified.png", img_left)
    cv2.imwrite("rright_rectified.png", img_right)
    
cv2.waitKey()
#if cv2.waitKey(1)==ord('q'):
#    break


#cam1.release()
#cam2.release()
cv2.destroyAllWindows()