import cv2
import numpy as np
import yaml

def stereo_rectification_calibrated():
    with open("./calibration_params.yaml", 'r') as file:
        #load the parameters form the yaml file created with matlab
        data = yaml.load(file)
        
        #read the left camera matrix
        leftCameraMatrix = np.asarray(data['cameraMatrixLeft'])
        rightCameraMatrix = np.asarray(data['cameraMatrixRight'])
        distCoeffsLeft = np.asarray(data['distCoeffsLeft'])
        distCoeffsRight = np.asarray(data['distCoeffsRight'])
        R = np.asarray(data['R'])
        T = np.asarray(data['T'])

        R1, R2, P1, P2, Q, ROI1, ROI2 = cv2.stereoRectify(leftCameraMatrix, distCoeffsLeft, 
                                                    rightCameraMatrix, distCoeffsRight, 
                                                    (1920, 1080), R, T, 1, newImageSize=(1920, 1080) )
        
        maps_left_cam = []
        maps_right_cam = []
        maps_left_cam = cv2.initUndistortRectifyMap(leftCameraMatrix, distCoeffsLeft, R1, P1, (1920, 1080), cv2.CV_16SC2)
        maps_right_cam = cv2.initUndistortRectifyMap(rightCameraMatrix, distCoeffsRight, R2, P2, (1920, 1080), cv2.CV_16SC2)

        return maps_left_cam, maps_right_cam, ROI1, ROI2