import cv2
import numpy as np
from gstreamer.gstreamer_base_code import __gstreamer_pipeline
from rectification.stereo_rectification_calibrated import stereo_rectification_calibrated
import vpi

#initialise camera sreams
cam1 = cv2.VideoCapture(__gstreamer_pipeline(camera_id=1, flip_method=0), cv2.CAP_GSTREAMER)
cam2 = cv2.VideoCapture(__gstreamer_pipeline(camera_id=0, flip_method=0), cv2.CAP_GSTREAMER)

ret1, left_frame = cam1.read()
print(left_frame.shape[0])
print(left_frame.shape[1])

#get stere rectification params
maps_left_cam, maps_right_cam, ROI1, ROI2 = stereo_rectification_calibrated()
#warp_left = vpi.WarpMap(vpi.WarpGrid((left_frame.shape[1], left_frame.shape[0])))
#maps_left_cam = maps_left_cam.transpose(2, 1, 0)
#wx_l, wy_l = np.asarray(warp_left).transpose(2,1,0)
#wx_l = maps_left_cam[0]
#wy_l = maps_left_cam[1]

#warp_right = vpi.WarpMap(vpi.WarpGrid((left_frame.shape[1], left_frame.shape[0])))
#maps_righ_cam = maps_right_cam.transpose(2, 1, 0)
#wx_r, wy_r = np.asarray(warp_right).transpose(2,1,0)
#wx_r = maps_right_cam[0]
#wy_r = maps_right_cam[1]

maxDisparity = 256
min_disp = 0         #original 16
block_size = 2            #original 8
uniquenessRatio = 0       #original 1
quality = 5
p1 = 5
p2 = 210
numPasses=2

scale=1
downscale=1

#initialise 2 streams for reading and preprocessing of frames
streamLeft = vpi.Stream()
streamRight = vpi.Stream()

while True:
    with vpi.Backend.CUDA:   #or CUDA
        with streamLeft:
            ret1, left_frame = cam1.read()
            left_frame = cv2.remap(left_frame, maps_left_cam[0], maps_left_cam[1], cv2.INTER_LANCZOS4)
            left = vpi.asimage(np.asarray(left_frame)).convert(vpi.Format.Y16_ER, scale=scale)
        with streamRight:
            ret2, right_frame = cam2.read()
            right_frame = cv2.remap(right_frame, maps_right_cam[0], maps_right_cam[1], cv2.INTER_LANCZOS4)
            right = vpi.asimage(np.asarray(right_frame)).convert(vpi.Format.Y16_ER, scale=scale)

    with vpi.Backend.VIC:
        with streamLeft:
            left = left.convert(vpi.Format.Y16_ER_BL)
        with streamRight:
            right = right.convert(vpi.Format.Y16_ER_BL)


    #get output width and height
    outWidth = (left.size[0] + downscale - 1) // downscale
    outHeight = (left.size[1] + downscale - 1) // downscale

    #use left stream to consolidate actual stereo processing
    streamStereo = streamLeft

    #estimate stereo disparity
    with streamStereo, vpi.Backend.OFA:
        disparityS16 = vpi.stereodisp(left, right, window=block_size, maxdisp=maxDisparity, mindisp=min_disp, 
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

        #cv2.imshow('LEFT FRAME UDIST',left)
        #cv2.imshow('RIGHT FRAME UDIST',right)
        cv2.imshow('DISPARITY', disparityU8)
        #cv2.moveWindow('LEFT FRAME UDIST', 100, 250)
        #cv2.moveWindow('RIGHT FRAME UDIST', 1100, 250)
        #cv2.moveWindow('DISPARITY', 100, 850))
        
    if cv2.waitKey(1)==ord('q'):
        break


cam1.release()
cam2.release()
cv2.destroyAllWindows()