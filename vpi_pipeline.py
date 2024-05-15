import cv2
import numpy as np
from gstreamer.gstreamer_base_code import __gstreamer_pipeline
from rectification.stereo_rectification_calibrated import stereo_rectification_calibrated
import vpi

#TODO: VPI STAGE 1: INITIALIZATION
#data (e.g. numpy areas) need to be wrapped in a VPI image, which can then be used for further processing

#initialise camera sreams
cam1 = cv2.VideoCapture(__gstreamer_pipeline(camera_id=1, flip_method=0), cv2.CAP_GSTREAMER)
cam2 = cv2.VideoCapture(__gstreamer_pipeline(camera_id=0, flip_method=0), cv2.CAP_GSTREAMER)

#get stere rectification params
maps_left_cam, maps_right_cam, ROI1, ROI2 = stereo_rectification_calibrated()

#stereo calibration params
minDisparity=0
maxDisparity=256
includeDiagonals=False
numPasses=1
downscale=1                         #what does downscale do exactly ? 
windowSize=(960, 540)
quality=6
scale=1

#initialise 2 streams for reading and preprocessing of frames
streamLeft = vpi.Stream()
streamRight = vpi.Stream()


#TODO: VPI STAGE 2: PROCESSING LOOP
while True:
    with vpi.Backend.CUDA:   #or CUDA
        with streamLeft:
            ret1, frame1 = cam1.read()
            frame1 = cv2.remap(frame1, maps_left_cam[0], maps_left_cam[1], cv2.INTER_LANCZOS4)
            left = vpi.asimage(np.asarray(frame1)).convert(vpi.Format.Y16_ER, scale=scale)
        with streamRight:
            ret2, frame2 = cam2.read()
            frame2 = cv2.remap(frame2, maps_right_cam[0], maps_right_cam[1], cv2.INTER_LANCZOS4)
            right = vpi.asimage(np.asarray(frame2)).convert(vpi.Format.Y16_ER, scale=scale)

    #convert to block-linear becuase needed for OFA
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
        #disparityS16 = vpi.stereodisp(left, right, downscale=downscale, window=windowSize, maxdisp=maxDisparity, quality=quality, mindisp=minDisparity)
        disparityS16 = vpi.stereodisp(left, right, window=25, maxdisp=256, uniqueness=0.5,includediagonals=False)

    #TODO: VPI STAGE 3: CLEANUP
    #must convert to pitch-linear if block-linear format
    if disparityS16.format == vpi.Format.S16_BL:
                disparityS16 = disparityS16.convert(vpi.Format.S16, backend=vpi.Backend.VIC)

    with streamStereo, vpi.Backend.CUDA:
        #scale and convert disparity map
        disparityU8 = disparityS16.convert(vpi.Format.U8, scale=255.0/(32*maxDisparity)).cpu()

        #convert to color JET map
        disparityColor = cv2.applyColorMap(disparityU8, cv2.COLORMAP_JET)

        cv2.imshow('LEFT FRAME UDIST',frame1)
        cv2.imshow('RIGHT FRAME UDIST',frame2)
        cv2.imshow('DISPARITY', disparityColor)
        cv2.moveWindow('LEFT FRAME UDIST', 100, 250)
        cv2.moveWindow('RIGHT FRAME UDIST', 1100, 250)
        cv2.moveWindow('DISPARITY', 100, 850)
        
        if cv2.waitKey(1)==ord('q'):
            break


cam1.release()
cam2.release()
cv2.destroyAllWindows()