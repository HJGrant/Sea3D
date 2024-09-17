import cv2
import numpy as np
from vpi_pipeline import vpi_pipeline
from stereo_rectification_calibrated import stereo_rectification_calibrated
import vpi
import os
from tqdm import tqdm
import open3d as o3d

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

    # Filter out points with depth <= 0 or depth == max valueqqq
    max_depth = np.max(depth)
    print(max_depth)
    mask = (depth > min_depth) & (depth < max_depth)

    pc_points = pc_points[mask.flatten()]
    pc_color = pc_color[mask.flatten()]

    # Create point cloud
    pcd.points = o3d.utility.Vector3dVector(pc_points)
    pcd.colors = o3d.utility.Vector3dVector(pc_color)

    # Downsample point cloud
    #voxel_size = 0.00001  # Adjust voxel size here
    #voxel_down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd],
                                      zoom=0.1,  # Adjust zoom level
                                      front=[0.4257, -0.2125, -0.8795],  # Adjust front view
                                      lookat=[2.6172, 2.0475, 1.532],  # Adjust lookat point
                                      up=[-0.0694, -0.9768, 0.2024])  # Adjust up vector
    
    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Apply Poisson surface reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    #o3d.visualization.draw_geometries([mesh])

    # Optionally, save the downsampled point cloud
    #o3d.io.write_point_cloud('vpi_point_cloud.ply', pcd)
    return pcd

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])

def calculate_depth(disparity_map, f, B):
    disparity_map[disparity_map == 0] = 0.1
    depth_map = (f * B) / disparity_map
    return depth_map

def process_stereo_images(left_image_path, right_image_path, output_dir):
    left_image = cv2.imread(left_image_path)
    right_image = cv2.imread(right_image_path)

    # Assuming vpi_pipeline generates depth, color, disparity, and stereo
    depth, color, disparity, stereo = vpi_pipeline(left_image, right_image, min_depth, max_depth, fx, B_mm, True, True)

    depth_map_u16 = cv2.normalize(depth, None, alpha=65535, beta=0, norm_type=cv2.NORM_MINMAX)
    depth_map_u16 = np.uint16(depth_map_u16)

    disparity_map_u16 = cv2.normalize(disparity, None, alpha=65535, beta=0, norm_type=cv2.NORM_MINMAX)
    disparity_map_u16 = np.uint16(disparity_map_u16)

    # Create separate folders for color, depth, disparity, and point clouds
    color_dir = os.path.join(output_dir, "color")
    depth_dir = os.path.join(output_dir, "depth")
    disparity_dir = os.path.join(output_dir, "disparity")
    point_cloud_dir = os.path.join(output_dir, "point_clouds")

    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(disparity_dir, exist_ok=True)
    os.makedirs(point_cloud_dir, exist_ok=True)

    # Save results to the appropriate directories
    left_image_basename = os.path.basename(left_image_path)
    basename_no_ext = os.path.splitext(left_image_basename)[0]
    print(left_image_basename)
    cv2.imwrite(os.path.join(color_dir, f"color_{left_image_basename}.png"), color)
    cv2.imwrite(os.path.join(depth_dir, f"depth_{left_image_basename}.png"), depth_map_u16)
    cv2.imwrite(os.path.join(disparity_dir, f"disparity_{left_image_basename}.png"), disparity_map_u16)

    print(left_image_basename)

    # Display results
    cv2.imshow('STEREO', stereo)
    cv2.imshow('color', color)
    cv2.imshow("DEPTH", depth_map_u16)
    cv2.moveWindow("DEPTH", 1500, 1000)
    cv2.imshow('DISPARITY', disparity_map_u16)

    # Generate and save point cloud if the 'p' key is pressed
    if cv2.waitKey(0) == ord('p'):
        point_cloud = generate_point_cloud(depth, color, fx, fy, U0, V0)
        point_cloud_path = os.path.join(point_cloud_dir, f"point_cloud_{basename_no_ext}.ply")
        save_point_cloud(point_cloud, point_cloud_path)

    cv2.destroyAllWindows()

def save_point_cloud(point_cloud, filepath):
    print(filepath)
    o3d.io.write_point_cloud(str(filepath), point_cloud)
    pass

# Intrinsic and stereo parameters
fx = 1103.963199
fy = 1096.068540
U0 = 950.492852
V0 = 531.100260
B_mm = 107.6305

min_depth = 4
max_depth = 1900

if __name__ == "__main__":
    left_folder = '/media/seaclear/639f8c93-fac2-4b84-a4fe-b261541674e9/lab_tests/lab_test_1009/10-09-2024_19:00:23/right'
    right_folder = '/media/seaclear/639f8c93-fac2-4b84-a4fe-b261541674e9/lab_tests/lab_test_1009/10-09-2024_19:00:23/left'
    output_dir = '/media/seaclear/639f8c93-fac2-4b84-a4fe-b261541674e9/lab_tests/lab_test_1009/output'

    # Specify the filename to start from
    start_filename = '19:02:56.285.jpg'

    # List all images in the folders and sort them
    left_images = sorted(os.listdir(left_folder))
    right_images = sorted(os.listdir(right_folder))

    # Find the starting index based on the left image
    try:
        start_index = left_images.index(start_filename)
    except ValueError:
        raise ValueError(f"Start file {start_filename} not found in {left_folder}")

    # Cycle through the image pairs starting from the start_index
    for left_img, right_img in zip(left_images[start_index:], right_images[start_index:]):
        left_image_path = os.path.join(left_folder, left_img)
        right_image_path = os.path.join(right_folder, right_img)

        # Process stereo image pair
        process_stereo_images(left_image_path, right_image_path, output_dir)
