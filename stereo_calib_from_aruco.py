import cv2
import numpy as np
import glob
from tqdm import tqdm

# Parameters
marker_length = 0.05  # The length of the markers' side in meters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Prepare object points (assuming a grid pattern)
def create_object_points():
    object_points = []
    for i in range(6):  # Number of marker rows
        for j in range(9):  # Number of marker columns
            object_points.append([i * marker_length, j * marker_length, 0])
    return np.array(object_points, dtype=np.float32)

# Load images from folders
def load_images(folder):
    images = glob.glob(f"{folder}/*.png")
    images.sort()  # Ensure that images are in order
    return [cv2.imread(img) for img in images]

# Find and draw markers
def find_markers(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        cv2.imwrite()
    return corners, ids

def find_common_markers(left_ids, right_ids):
    common_ids = np.intersect1d(left_ids.flatten(), right_ids.flatten())
    left_indices = np.where(np.isin(left_ids.flatten(), common_ids))[0]
    right_indices = np.where(np.isin(right_ids.flatten(), common_ids))[0]
    return left_indices, right_indices

# Main calibration function
def calibrate_stereo(left_images_folder, right_images_folder):
    left_images = load_images(left_images_folder)
    right_images = load_images(right_images_folder)

    if len(left_images) != len(right_images):
        raise ValueError("The number of images in the left and right folders must be the same.")

    object_points = create_object_points()
    left_corners_all = []
    right_corners_all = []
    obj_points_all = []
    
    for left_img, right_img in zip(left_images, right_images):
        left_corners, left_ids = find_markers(left_img)
        right_corners, right_ids = find_markers(right_img)

        if left_ids is not None and right_ids is not None:
            print('Finding Commmon Markers!')
            left_indices, right_indices = find_common_markers(left_ids.flatten(), right_ids.flatten())
            
            if len(left_indices) > 0 and len(right_indices) > 0:
                matched_left_corners = [left_corners[i] for i in left_indices]
                matched_right_corners = [right_corners[i] for i in right_indices]
                
                left_corners_all.append(matched_left_corners)
                right_corners_all.append(matched_right_corners)
                obj_points_all.append(object_points[:len(matched_left_corners)]) 

    # Stereo calibration
    ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(
        obj_points_all,
        left_corners_all,
        right_corners_all,
        None,
        None,
        None,
        None,
        left_images[0].shape[:1],
        criteria=cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        flags=cv2.CALIB_FIX_INTRINSIC
    )

    if not ret:
        raise RuntimeError("Stereo calibration failed.")

    # Save calibration results
    np.savez("stereo_calibration.npz", mtx1=mtx1, dist1=dist1, mtx2=mtx2, dist2=dist2, R=R, T=T)

    print("Stereo calibration completed and results saved.")

# Run the calibration
calibrate_stereo('data/images/left', 'data/images/right')
