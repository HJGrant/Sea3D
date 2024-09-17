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
    o3d.visualization.draw_geometries([mesh])


    # Optionally, save the downsampled point cloud
    o3d.io.write_point_cloud('vpi_point_cloud.ply', pcd)

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

fx = 1103.963199
fy= 1096.068540
U0= 950.492852
V0= 531.100260
B_mm = 107.6305
#B_meters = B 

min_depth = 4
max_depth = 1900

if __name__ == "__main__":

    left_image = cv2.imread("frame_07740_left.png")
    right_image = cv2.imread("frame_07740_right.png")

    depth, color, disparity, stereo = vpi_pipeline(left_image, right_image, min_depth, max_depth, fx, B_mm, True, True)

    depth_map_u16 = cv2.normalize(depth, None, alpha=65535, beta=0, norm_type=cv2.NORM_MINMAX)
    depth_map_u16 = np.uint16(depth_map_u16)

    disparity_map_u16 = cv2.normalize(disparity, None, alpha=65535, beta=0, norm_type=cv2.NORM_MINMAX)
    disparity_map_u16 = np.uint16(disparity_map_u16)

    depth_map_u8 = cv2.normalize(depth, None, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX)
    depth_map_u8 = np.uint8(depth_map_u8)

    diparity_map_u8 = cv2.normalize(disparity, None, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX)
    disparity_map_u8 = np.uint8(diparity_map_u8)

    #color = cv2.resize(color, (960, 480), interpolation=cv2.INTER_LINEAR)
    #depth = cv2.resize(depth, (960, 480), interpolation=cv2.INTER_LINEAR)
    #disparity = cv2.resize(disparity, (960, 480), interpolation=cv2.INTER_LINEAR)

    #postProcStream = vpi.Stream()
    #with postProcStream, vpi.Backend.CUDA:

            #vpi_depth_map = vpi.asimage(depth_map)
    #        vpi_color = vpi.asimage(color)
    #        vpi_disparity = vpi.asimage(disparity)

            # Resize the image
            #resized_depth = vpi_depth_map.rescale((960, 480))  # Resize to half the original size
    #        resized_disparity = vpi_disparity.rescale((960, 480))

            #median_filtered = resized_depth.median_filter((3, 3))  # 5x5 kernel

            # Apply Median Filter
            #for i in range(3):
            #    median_filtered = median_filtered.median_filter((4, 4))  # 5x5 kernel

            # Apply Bilateral Filter
            #bilateral_filtered = median_filtered.bilateral_filter(9, 20, 20)  # Bilateral filter with kernel size 9, sigma_color 0.1, sigma_space 2.0
            
            #for i in range(3):
            #     bilateral_filtered = bilateral_filtered.bilateral_filter(9, 21, 20)

            # Convert back to OpenCV image for saving/displaying
            #result_depth = resized_depth.cpu()
            #result_depth = 6555 - result_depth
            #result_depth = bilateral_filtered.convert(vpi.Format.U16, scale=65535.0 / (32 * max_disparity)).cpu()

    cv2.imshow('STEREO', stereo)
    cv2.imshow('color', color)
    cv2.imshow('DISPARITY', disparity_map_u16)
    cv2.imshow("DEPTH", depth_map_u16)
    cv2.waitKey(0)

    cv2.imwrite("current_depth.png", depth_map_u16)
    cv2.imwrite("current_disp.png", disparity_map_u16)
    cv2.imwrite("current_color.png", color)

    generate_point_cloud(depth, color, fx, fy, U0, V0)

    cv2.destroyAllWindows()