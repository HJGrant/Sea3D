import cv2
import numpy as np
import os
from gstreamer.gstreamer_base_code import __gstreamer_pipeline
from stereo_rectification_calibrated import stereo_rectification_calibrated
from depth_map import depth_map
import matplotlib.pyplot as plt

def draw_horizontal_lines(img, num_lines=20, color=(0, 255, 0), thickness=1):
    height = img.shape[0]
    step = height // num_lines
    for i in range(0, height, step):
        cv2.line(img, (0, i), (img.shape[1], i), color, thickness)
    return img

def display_stereo_images(side_by_side_img):
    cv2.imshow('Rectified Stereo Pair', side_by_side_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def calculate_depth(disparity_map, f, B):
    disparity_map[disparity_map == 0] = 0.1
    depth_map = (f * B) / disparity_map
    return depth_map

def get_sorted_filenames(folder):
    return sorted([f for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

def process_image_pairs(left_folder, right_folder, start_filename):
    # Get sorted lists of image filenames from both folders
    left_images = get_sorted_filenames(left_folder)
    right_images = get_sorted_filenames(right_folder)

    # Start processing from the given filename
    if start_filename in left_images:
        start_index = left_images.index(start_filename)
    else:
        print(f"Error: {start_filename} not found in {left_folder}")
        return
    
    # Initialize rectification maps
    maps_left_cam, maps_right_cam, ROI1, ROI2 = stereo_rectification_calibrated()

    # Process each image pair
    for i in range(start_index, len(left_images)):
        left_image_path = os.path.join(left_folder, left_images[i])
        right_image_path = os.path.join(right_folder, right_images[i])

        print(f"Processing pair: {left_images[i]} and {right_images[i]}")
        
        # Read the images
        frame1 = cv2.imread(left_image_path)
        frame2 = cv2.imread(right_image_path)

        if frame1 is None or frame2 is None:
            print(f"Error reading {left_image_path} or {right_image_path}")
            continue

        # Rectify the images
        left_frame_rectified = cv2.remap(frame1, maps_left_cam[0], maps_left_cam[1], cv2.INTER_LANCZOS4)
        right_frame_rectified = cv2.remap(frame2, maps_right_cam[0], maps_right_cam[1], cv2.INTER_LANCZOS4)

        # Set the ROI for both images
        left_frame_rectified = left_frame_rectified[ROI1[1]:ROI1[3], ROI1[0]:ROI1[2]]
        right_frame_rectified = right_frame_rectified[ROI1[1]:ROI1[3], ROI1[0]:ROI1[2]]

        # Resize for display purposes
        left_frame_rectified = cv2.resize(left_frame_rectified, (960, 480))
        right_frame_rectified = cv2.resize(right_frame_rectified, (960, 480))

        # Combine into side-by-side image
        side_by_side_img = np.hstack((left_frame_rectified, right_frame_rectified))

        # Draw horizontal lines on the side-by-side image
        side_by_side_img_with_lines = draw_horizontal_lines(side_by_side_img)

        # Compute disparity
        disparity = depth_map(right_frame_rectified, left_frame_rectified)

        # Display images and disparity map
        cv2.imshow('stereo_rectified', side_by_side_img_with_lines)
        cv2.imshow('DISPARITY', disparity)
        cv2.moveWindow('stereo_rectified', 100, 250)
        cv2.moveWindow('DISPARITY', 100, 950)

        # Save the results
        output_name = os.path.splitext(left_images[i])[0]
        cv2.imwrite(f'rectified_{output_name}.png', side_by_side_img_with_lines)
        cv2.imwrite(f'disparity_{output_name}.png', disparity)

        # Wait for a key press before processing the next pair
        cv2.waitKey(0)

    # Close all windows after processing
    cv2.destroyAllWindows()

# Example usage
right_folder = "/media/seaclear/639f8c93-fac2-4b84-a4fe-b261541674e9/lab_tests/lab_test_1609/16-09-2024_18:29:52/right"
left_folder = "/media/seaclear/639f8c93-fac2-4b84-a4fe-b261541674e9/lab_tests/lab_test_1609/16-09-2024_18:29:52/left"
start_filename = "18_30_00.441.jpg"

f=1103.963199
B=107.9599

process_image_pairs(left_folder, right_folder, start_filename)
