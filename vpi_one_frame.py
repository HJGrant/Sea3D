import cv2
import numpy as np
from vpi_pipeline import vpi_pipeline
from stereo_rectification_calibrated import stereo_rectification_calibrated
import vpi
import os
from tqdm import tqdm
import open3d as o3d


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


f=1075.9
B=107.9599
B_meters = B / 1000

if __name__ == "__main__":

    left_image = cv2.imread("test_imgs/14:37:14.390_right.jpg")
    right_image = cv2.imread("test_imgs/14:37:14.390_left.jpg")

    depth, color, disparity, stereo = vpi_pipeline(left_image, right_image)

    color = cv2.resize(color, (960, 480), interpolation=cv2.INTER_LINEAR)
    depth = cv2.resize(depth, (960, 480), interpolation=cv2.INTER_LINEAR)

    postProcStream = vpi.Stream()
    with postProcStream, vpi.Backend.CUDA:

            #vpi_depth_map = vpi.asimage(depth_map)
            vpi_color = vpi.asimage(color)
            vpi_disparity = vpi.asimage(disparity)

            # Resize the image
            #resized_depth = vpi_depth_map.rescale((960, 480))  # Resize to half the original size
            resized_disparity = vpi_disparity.rescale((960, 480))

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
    cv2.imshow('DISPARITY', disparity)
    cv2.imshow("DEPTH", depth)
    cv2.waitKey(0)


    pcd = o3d.geometry.PointCloud()

    #  pc_points: array(Nx3), each row composed with x, y, z in the 3D coordinate
    #  pc_color: array(Nx3), each row composed with R G,B in the rage of 0~1
    pc_points = np.array([], np.float32) # 픽셀의 좌표 값
    pc_color = np.array([], np.float32) # 픽셀의 RGB 값 (floating point 원래는 0~255이지만 0~1로 최대 max를 1로하기! -> 255만 나누기!)

    # 3D reconstruction
    # Concatenate pc_points and pc_color
    # ****************************** Your code here (M-4) ******************************
    # Get intrinsic parameter
    # Focal length
    fx = 503.5825325
    fy =  501.958523
    # Principal point
    U0 = 469.031023
    V0 = 268.1415965


    # RGB to BGR for pc_points
    imgLU = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

    # depth = inverse of disparity
    #depth = 255 - disp8
    depth = depth
    h, w = depth.shape

    for v in tqdm(range(h)):
        for u in range(w):
            if(depth[v][u] > 0): # ignore disparity (threshold 이하의 값을 가지는 pixel)
                # pc_points
                x = (u - U0) * depth[v][u] / fx
                y = (v - V0) * depth[v][u] / fy
                z = depth[v][u]
                pc_points = np.append(pc_points, np.array(np.float32(([x, y, z]))))
                pc_points = np.reshape(pc_points, (-1, 3))
                # pc_colors
                pc_color = np.append(pc_color, np.array(np.float32(imgLU[v][u] / 255)))
                pc_color = np.reshape(pc_color, (-1, 3))

    # **********************************************************************************
    #  add position and color to point cloud
    pcd.points = o3d.utility.Vector3dVector(pc_points)
    pcd.colors = o3d.utility.Vector3dVector(pc_color)

    o3d.io.write_point_cloud('my_point_cloud.ply', pcd)
    
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.02)

    o3d.visualization.draw_geometries([pcd],
                                    zoom=0.0412,
                                    front=[0.4257, -0.2125, -0.8795],
                                    lookat=[2.6172, 2.0475, 1.532],
                                    up=[-0.0694, -0.9768, 0.2024])
    
    print("Statistical oulier removal")
    cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=20,
                                                        std_ratio=2.0)
    display_inlier_outlier(voxel_down_pcd, ind)

    cv2.destroyAllWindows()