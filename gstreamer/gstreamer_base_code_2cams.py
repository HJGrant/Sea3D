import cv2
from time import sleep, time
print(cv2.__version__)

#funciton for defining the gstreamer pipelin string
#Note: you may need to find a setting here to set the latency of gstreamer to 0
def __gstreamer_pipeline(
        camera_id,
        sensor_mode=2,          #4 = 3856x2180 and 90fps; 3 = 3856 x 2180 FR = 29.999999 fps; 2 = 1928 x 1090 FR = 59.999999 fps;
        capture_width=1928,
        capture_height=1090,
        display_width=1920,
        display_height=1080,
        framerate=30,
        flip_method=0,
        tnr_mode=1,                     #2=NoiseReduction_HighQualit, 1=NoiseReduction_Fast, 0=NoiseReduction_Off
        tnr_strength=1,                 #Noise Reduction Strength, Range: -1 to 1
        white_balance=4,                #
        exposure_compensation=-1.5,     #Range: -2 and 2
        brightness=-0.05,               #Range: -1 to 1 
        contrast=1.5,                   #Range: 0 to 2
        saturation=1.5,                 #Range: 0 to 2 
        hue=0.05                        #Range -1 to 1
    ):
    return (
            "nvarguscamerasrc sensor-id=%d sensor-mode=%d tnr_mode=%d tnr_strength=%f wbmode=%d exposurecompensation=%f ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d,"
            "format=(string)NV12 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! "
            "videobalance brightness=%f contrast=%f saturation=%d hue=%f ! "
            "appsink "
            % (
                    camera_id,
                    sensor_mode,
                    tnr_mode,
                    tnr_strength,
                    white_balance,
                    exposure_compensation,
                    capture_width,
                    capture_height,
                    flip_method,
                    display_width,
                    display_height,
                    brightness,
                    contrast,
                    saturation,
                    hue,

            )
    )

def __gstreamer_pipeline_no_extras(
        camera_id,
        sensor_mode=4,          #4 = 3856x2180 and 90fps; 3 = 3856 x 2180 FR = 29.999999 fps; 2 = 1928 x 1090 FR = 59.999999 fps;
        capture_width=3856,
        capture_height=2180,
        display_width=960,
        display_height=540,
        framerate=30,
        flip_method=0,
    ):
    return (
            "nvarguscamerasrc sensor-id=%d sensor-mode=%d ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d,"
            "format=(string)NV12 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! "
            "appsink "
            % (
                    camera_id,
                    sensor_mode,
                    capture_width,
                    capture_height,
                    flip_method,
                    display_width,
                    display_height
            )
    )


#initialise video capture object with edits 
#cam1 = cv2.VideoCapture(__gstreamer_pipeline(camera_id=0, flip_method=0), cv2.CAP_GSTREAMER)
#cam2 = cv2.VideoCapture(__gstreamer_pipeline(camera_id=1, flip_method=0), cv2.CAP_GSTREAMER)

#initialise video capture without edits
cam1 = cv2.VideoCapture(__gstreamer_pipeline_no_extras(camera_id=0, flip_method=0), cv2.CAP_GSTREAMER)
cam2 = cv2.VideoCapture(__gstreamer_pipeline_no_extras(camera_id=1, flip_method=0), cv2.CAP_GSTREAMER)

#check if video capture object was properly initialised and able to open
if not cam1.isOpened():
        print("Cannot open camera 1")
        exit()

if not cam2.isOpened():
        print("Cannot open camera 2")
        exit()

#Main loop
index = 0
delay = 5
while True:
        ret1, frame1 = cam1.read()
        ret2, frame2 = cam2.read()

        #frame1 = cv2.resize(frame1, (964, 545))
        #frame2 = cv2.resize(frame2, (964, 545))

        cv2.imshow('FRAMOS1', frame1)
        cv2.imshow('FRAMOS2', frame2)
        cv2.moveWindow('FRAMOS1', 100, 250)
        cv2.moveWindow('FRAMOS2', 1100, 250)
        

        keyEvent = cv2.waitKey(1)

        if keyEvent==ord('q'):
                break

#close video capture object and close opencv window   
cam1.release()
cam2.release()
cv2.destroyAllWindows()