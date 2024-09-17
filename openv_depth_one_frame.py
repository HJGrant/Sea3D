import cv2
import numpy as np
import os
from gstreamer.gstreamer_base_code import __gstreamer_pipeline
from stereo_rectification_calibrated import stereo_rectification_calibrated
from depth_map import disp_map
import matplotlib.pyplot as plt
import open3d as o3d
from tqdm import tqdm

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
    disparity_map[disparity_map == 0] = 0.001
    depth_map = (f * B) / disparity_map
    return depth_map

def get_sorted_filenames(folder):
    return sorted([f for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

def smooth_depth_map(depth_map, kernel_size=5):
    """
    Apply a median filter to smoothen the depth map.

    Parameters:
    - depth_map: The input depth map (2D NumPy array).
    - kernel_size: Size of the kernel used for the median filter (should be odd).

    Returns:
    - Smoothed depth map.
    """
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")

    # Apply median filter
    smoothed_depth_map = cv2.medianBlur(depth_map.astype(np.uint8), kernel_size)

    return smoothed_depth_map

def generate_point_cloud(depth, rgb, fx, fy, U0, V0):
    pcd = o3d.geometry.PointCloud()

    # Set a minimum depth value to avoid division errors (adjust as needed)
    min_depth = np.min(depth)

    # Create a meshgrid of pixel coordinates
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # Calculate x, y, z coordinates for each pixel
    x = (u - U0) * depth / fx
    y = (v - V0) * depth / fy
    z = depth

    # Flatten the arrays
    pc_points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T

    # Normalize RGB values
    imgLU = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    pc_color = (imgLU / 255).reshape(-1, 3)

    # Filter out points with depth <= 0 or depth == max value
    max_depth = np.max(depth)
    print(max_depth)
    mask = (depth > min_depth) & (depth < max_depth)

    pc_points = pc_points[mask.flatten()]
    pc_color = pc_color[mask.flatten()]

    # Create point cloud
    pcd.points = o3d.utility.Vector3dVector(pc_points)
    pcd.colors = o3d.utility.Vector3dVector(pc_color)

    # Downsample point cloud
    voxel_size = 0.00001  # Adjust voxel size here
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([voxel_down_pcd],
                                      zoom=0.1,  # Adjust zoom level
                                      front=[0.4257, -0.2125, -0.8795],  # Adjust front view
                                      lookat=[2.6172, 2.0475, 1.532],  # Adjust lookat point
                                      up=[-0.0694, -0.9768, 0.2024])  # Adjust up vector

    # Optionally, save the downsampled point cloud
    o3d.io.write_point_cloud('ocv_point_cloud.ply', voxel_down_pcd)


def process_image_pairs(parent_folder, start_filename, fx, fy, U0, V0):
    # Define left and right folders based on parent folder
    left_folder = os.path.join(parent_folder, 'left')
    right_folder = os.path.join(parent_folder, 'right')
    
    # Create output folders if they don't exist
    color_folder = os.path.join(parent_folder, 'color')
    depth_folder = os.path.join(parent_folder, 'depth')
    os.makedirs(color_folder, exist_ok=True)
    os.makedirs(depth_folder, exist_ok=True)

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
        left_frame_rectified_disp = cv2.resize(left_frame_rectified, (960, 480))
        right_frame_rectified_disp = cv2.resize(right_frame_rectified, (960, 480))

        # Combine into side-by-side image
        side_by_side_img = np.hstack((left_frame_rectified, right_frame_rectified))

        # Draw horizontal lines on the side-by-side image
        side_by_side_img_with_lines = draw_horizontal_lines(side_by_side_img)

        # Compute disparity
        print(left_frame_rectified.dtype)
        disparity = disp_map(left_frame_rectified, right_frame_rectified)
        print(disparity.dtype)
        #disparity = smooth_depth_map(disparity, kernel_size=3)
        #for i in range(5):
        #    disparity = smooth_depth_map(disparity, kernel_size=3)
        #disparity = cv2.bilateralFilter(disparity, d=9, sigmaColor=35, sigmaSpace=75)
        disp_map_disp = cv2.resize(disparity, (960, 480))

        # Check disparity values
        print(f"Max disparity: {np.max(disparity)}")
        print(f"Min disparity: {np.min(disparity)}")

        # Calculate depth map
        depth_map = calculate_depth(disparity, fx, B_meters)
        #depth_map = smooth_depth_map(depth_map, kernel_size=1)
        print(depth_map.dtype)
        depth_map_disp = cv2.resize(depth_map, (960, 480))

        # Apply depth thresholding
        depth_map[depth_map < min_depth] = min_depth
        depth_map[depth_map > max_depth] = max_depth

        side_by_side_img = np.hstack((left_frame_rectified_disp, right_frame_rectified_disp))
        stereo_w_lines = draw_horizontal_lines(side_by_side_img)

        # Validate depth map values
        print(f"Max depth: {np.max(depth_map)}")
        print(f"Min depth: {np.min(depth_map)}")

        # Display images and disparity map
        cv2.imshow('stereo_rectified', stereo_w_lines)
        cv2.imshow('DEPTH', depth_map_disp)
        cv2.imshow('DISPARITY', disp_map_disp)
        cv2.moveWindow('stereo_rectified', 100, 250)
        cv2.moveWindow('DISPARITY', 100, 950)

        # Save rectified and depth map images
        print(depth_map.shape)
        left_frame_rectified_reshape = cv2.resize(left_frame_rectified, (depth_map.shape[1], depth_map.shape[0]))
        output_name = os.path.splitext(left_images[i])[0]

        cv2.imwrite(os.path.join(color_folder, f'rectified_{output_name}.jpg'), left_frame_rectified_reshape)
        depth_map_u16 = cv2.normalize(depth_map, None, alpha=65535, beta=0, norm_type=cv2.NORM_MINMAX)
        depth_map_u16 = np.uint16(depth_map_u16)
        cv2.imwrite(os.path.join(depth_folder, f'disparity_{output_name}.png'), depth_map_u16)

        # Wait for a key press before processing the next pair
        cv2.waitKey(0)

        print(depth_map.shape)
        print(left_frame_rectified.shape)

        generate_point_cloud(depth_map, left_frame_rectified, fx, fy, U0, V0)

    # Close all windows after processing
    cv2.destroyAllWindows()

# Example usage
parent_folder = "/media/seaclear/639f8c93-fac2-4b84-a4fe-b261541674e9/lab_tests/lab_test_1609/16-09-2024_18:29:52"
start_filename = "18_30_00.441.jpg"

fx = 1103.963199
fy= 1096.068540
U0= 950.492852
V0= 531.100260
B = 107.6305
B_meters = B / 1000
min_depth = -2
max_depth = 4.0

process_image_pairs(parent_folder, start_filename, fx, fy, U0, V0)
