import cv2
from time import sleep, time
from gstreamer.gstreamer_base_code import __gstreamer_pipeline
from rectification.stereo_rectification_calibrated import stereo_rectification_calibrated
print(cv2.__version__)

#initialise video capture object   
cam1 = cv2.VideoCapture(__gstreamer_pipeline(camera_id=0, flip_method=0), cv2.CAP_GSTREAMER)
cam2 = cv2.VideoCapture(__gstreamer_pipeline(camera_id=1, flip_method=0), cv2.CAP_GSTREAMER)

#check if video capture object was properly initialised and able to open
if not cam1.isOpened():
 print("Cannot open camera 1")
 exit()

if not cam2.isOpened():
 print("Cannot open camera 2")
 exit()

maps_left_cam, maps_right_cam, ROI1, ROI2 = stereo_rectification_calibrated()

#Main loop
index = 0
while True:
    ret1, frame1 = cam1.read()
    ret2, frame2 = cam2.read()

    #left_frame_rectified = cv2.remap(frame1, maps_left_cam[0], maps_left_cam[1], cv2.INTER_LANCZOS4)
    #right_frame_rectified = cv2.remap(frame2, maps_right_cam[0], maps_right_cam[1], cv2.INTER_LANCZOS4)

    #print(ROI1) #1831x1003
    #print(ROI2) #1843x1026

    #set the ROI for both images
    #left_frame_rectified = left_frame_rectified[ROI1[1]:ROI1[3], ROI1[0]:ROI1[2]] #minus 1 to set shape to same dimensions TODO: solve this better
    #right_frame_rectified = right_frame_rectified[ROI1[1]:ROI1[3], ROI1[0]:ROI1[2]]

    cv2.imshow('FRAMOS1',frame1)
    cv2.imshow('FRAMOS2', frame2)
    cv2.moveWindow('FRAMOS1', 0, 250)
    cv2.moveWindow('FRAMOS2', 1100, 250)

    keyEvent = cv2.waitKey(1)

    if keyEvent == ord('i'):
        sleep(0.5)
        cv2.imwrite('left_image_'+str(index)+'.png', frame1)
        cv2.imwrite('right_image_'+str(index)+'.png', frame2)
        index += 1

    if keyEvent == ord('d'):
        start = time()
        while time()- start <= 5:
                ret1, frame1 = cam1.read()
                ret2, frame2 = cam2.read()
                cv2.imshow('FRAMOS1',frame1)
                cv2.imshow('FRAMOS2', frame2)
                cv2.moveWindow('FRAMOS1', 0, 250)
                cv2.moveWindow('FRAMOS2', 1100, 250)

        cv2.imwrite('left_image_'+str(index)+'.png', frame1)
        cv2.imwrite('right_image_'+str(index)+'.png', frame2)
        index += 1

    if keyEvent==ord('q'):
        break

#close video capture object and close opencv window   
cam1.release()
cam2.release()
cv2.destroyAllWindows()