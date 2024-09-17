import cv2
import numpy as np

def stereo_rectification_calibrated():
    # Read the intrinsic parameters from YAML file
    fs = cv2.FileStorage("calibration_params_1009.yml", cv2.FILE_STORAGE_READ)
    calibParams = {
        'cameraMatrixLeft': fs.getNode("cameraMatrixLeft").mat(),
        'distCoeffsLeft': fs.getNode("distCoeffsLeft").mat(),
        'cameraMatrixRight': fs.getNode("cameraMatrixRight").mat(),
        'distCoeffsRight': fs.getNode("distCoeffsRight").mat(),
        'R': fs.getNode("R").mat(),
        'T': fs.getNode("T").mat()
    }
    fs.release()  # Close the FileStorage object

    # Image size (should match the size of the images used for calibration)
    image_size = (1920, 1080)

    # Get optimal new camera matrix
    new_camera_matrix_left, roi_left = cv2.getOptimalNewCameraMatrix(
        calibParams['cameraMatrixLeft'], calibParams['distCoeffsLeft'], 
        image_size, alpha=-1
    )
    new_camera_matrix_right, roi_right = cv2.getOptimalNewCameraMatrix(
        calibParams['cameraMatrixRight'], calibParams['distCoeffsRight'], 
        image_size, alpha=-1
    )

    # Compute the rectification transforms
    R1, R2, P1, P2, Q, ROI1, ROI2 = cv2.stereoRectify(
        calibParams['cameraMatrixLeft'], calibParams['distCoeffsLeft'],
        calibParams['cameraMatrixRight'], calibParams['distCoeffsRight'],
        image_size, calibParams['R'], calibParams['T'],
        alpha=-1, newImageSize=image_size
    )

    # Compute the undistortion and rectification transformation maps
    mapx_left, mapy_left = cv2.initUndistortRectifyMap(
        new_camera_matrix_left, calibParams['distCoeffsLeft'], R1, P1,
        image_size, cv2.CV_16SC2
    )
    mapx_right, mapy_right = cv2.initUndistortRectifyMap(
        new_camera_matrix_right, calibParams['distCoeffsRight'], R2, P2,
        image_size, cv2.CV_16SC2
    )

    return (mapx_left, mapy_left), (mapx_right, mapy_right), ROI1, ROI2