import cv2
import numpy as np
from gstreamer.gstreamer_base_code import __gstreamer_pipeline
#from rectification.stereo_rectification_calibrated import stereo_rectification_calibrated
from stereo_rectification_calibrated import stereo_rectification_calibrated
import vpi


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

maxDisparity = 256
min_disp = 1         #original 16
block_size = 3           #original 8
uniquenessRatio = 12       #original 1
quality = 2
p1 = 10
p2 = 125
numPasses=3

scale=0.9
downscale=0.5

#initialise 2 streams for reading and preprocessing of frames
streamLeft = vpi.Stream()
streamRight = vpi.Stream()


#TODO: VPI STAGE 2: PROCESSING LOOP
#while True:
with vpi.Backend.CUDA:   #or CUDA
    with streamLeft:
        frame2 = cv2.imread("frame_07978_left.png")
        #ret1, frame1 = cam1.read()
        frame2 = cv2.remap(frame2, maps_left_cam[0], maps_left_cam[1], cv2.INTER_LANCZOS4)
        left = vpi.asimage(np.asarray(frame2)).convert(vpi.Format.Y16_ER, scale=scale)
    with streamRight:
        #ret2, frame2 = cam2.read()
        frame1 = cv2.imread("frame_07978_right.png")
        frame1 = cv2.remap(frame1, maps_right_cam[0], maps_right_cam[1], cv2.INTER_LANCZOS4)
        right = vpi.asimage(np.asarray(frame1)).convert(vpi.Format.Y16_ER, scale=scale)

#with vpi.Backend.CUDA:
#    with streamLeft:
#        left_1 = left.convert(vpi.Format.Y16_ER_BL)
#    with streamRight:
#        right_1 = right.convert(vpi.Format.Y16_ER_BL)


#get output width and height
outWidth = (left.size[0] + downscale - 1) // downscale
outHeight = (left.size[1] + downscale - 1) // downscale

#use left stream to consolidate actual stereo processing
streamStereo = streamLeft

#estimate stereo disparity
with streamStereo, vpi.Backend.CUDA:
    disparityS16 = vpi.stereodisp(left, right, window=block_size, maxdisp=maxDisparity, mindisp=min_disp, 
                                   quality=quality, uniqueness=uniquenessRatio, includediagonals=False, numpasses=numPasses, p1=p1, p2=p2)

#TODO: VPI STAGE 3: CLEANUP
#must convert to pitch-linear if block-linear format
if disparityS16.format == vpi.Format.S16_BL:
            disparityS16 = disparityS16.convert(vpi.Format.S16, backend=vpi.Backend.CUDA)

with streamStereo, vpi.Backend.CUDA:
    #scale and convert disparity map
    disparityU8 = disparityS16.convert(vpi.Format.U8, scale=255.0/(32*maxDisparity)).cpu()

    #convert to color JET map
    disparityColor = cv2.applyColorMap(disparityU8, cv2.COLORMAP_JET)
    
    left = left.convert(vpi.Format.U8).cpu()
    right = right.convert(vpi.Format.U8).cpu()

    side_by_side_img = np.hstack((frame2, frame1))

    # Draw horizontal lines
    stereo_uncalib_w_lines = draw_horizontal_lines(side_by_side_img)
    

    cv2.imshow('STEREO', stereo_uncalib_w_lines)
    cv2.imshow('DISPARITY', disparityU8)
    cv2.moveWindow('LEFT FRAME UDIST', 100, 250)
    cv2.moveWindow('RIGHT FRAME UDIST', 1100, 250)
    cv2.moveWindow('DISPARITY', 100, 850)
    cv2.imwrite("disparity_map.png", disparityU8)
    cv2.imwrite("disparity_map_color.png", disparityColor)
    cv2.imwrite('rectified_stereo.png', stereo_uncalib_w_lines)
    #cv2.imwrite("left_rectified.png", img_left)
    #cv2.imwrite("rright_rectified.png", img_right)
    
cv2.waitKey()
#if cv2.waitKey(1)==ord('q'):
#    break


#cam1.release()
#cam2.release()
cv2.destroyAllWindows()