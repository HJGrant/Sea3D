import cv2
import numpy as np
from gstreamer.gstreamer_base_code import __gstreamer_pipeline
from rectification.stereo_rectification_calibrated import stereo_rectification_calibrated
import vpi

#initialise camera sreams
cam1 = cv2.VideoCapture(__gstreamer_pipeline(camera_id=1, flip_method=0), cv2.CAP_GSTREAMER)
#cam2 = cv2.VideoCapture(__gstreamer_pipeline(camera_id=0, flip_method=0), cv2.CAP_GSTREAMER)

ret1, frame1 = cam1.read()

#get stere rectification params
maps_left_cam, maps_right_cam, ROI1, ROI2 = stereo_rectification_calibrated()

vpi_img = vpi.asimage(np.asarray(frame1))

warp = vpi.WarpMap(vpi.WarpGrid((1920,1080)))
maps_left_cam = maps_left_cam.transpose(2, 1, 0)
wx, wy = np.asarray(warp).transpose(2,1,0)
wx = maps_left_cam[0]
wy = maps_left_cam[1]

#print(np.asarray(warp).shape)

with vpi.Backend.CUDA:
    output = vpi_img.remap(warp)
    out = output.cpu()

while True:
    cv2.imshow('OUT', np.asarray(out))
    cv2.imshow('ORIGINAL', frame1)

    if cv2.waitKey(1)==ord('q'):
        break

cv2.destroyAllWindows()