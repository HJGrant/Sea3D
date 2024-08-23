import cv2
import os

def detect_aruco_markers(image_path, aruco_dict, parameters, detector):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)

    # Draw detected markers
    if ids is not None:
        print("Sucess!")
        print(str(image_path))
        cv2.aruco.drawDetectedMarkers(image, corners, ids)

    return image

def process_images_in_folder(folder_path):
    # Define ArUco dictionary and parameters
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # Process each image in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            output_image = detect_aruco_markers(image_path, aruco_dict, parameters, detector)

            # Save the output image with detected markers
            output_path = os.path.join(folder_path, 'detected_' + filename)
            cv2.imshow('ARUCO', output_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # Folder containing images
    folder_path = 'data/images/right'
    if os.path.exists(folder_path):
        process_images_in_folder(folder_path)
    else:
        print(f"Folder '{folder_path}' does not exist.")
