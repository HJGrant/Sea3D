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

####OAK-D###
f = 521.92
B = 75

if __name__ == "__main__":

    left_image = cv2.imread("19_07_42.719_left.jpg")
    right_image = cv2.imread("19_07_42.719_right.jpg")

    #depth_map, color, disparity, stereo = vpi_pipeline(left_image, right_image)

    disp8 = cv2.imread("oak-d_disparity_screenshot_16.09.2024.png", cv2.IMREAD_GRAYSCALE)
    color = cv2.imread("oak-d_rgb_screenshot_16.09.2024.png")
    
    disp16 = (disp8.astype(np.uint16)) * 257
    
    depth_map = calculate_depth(disp16, f, B)
    
    #cv2.imshow('STEREO', stereo)
    cv2.imshow('color', color)
    cv2.imshow('DISPARITY', disp16)
    cv2.imshow("DEPTH", depth_map)
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
    #fx = 7.1760e+02
    #fy = 7.1267e+02
    # Principal point
    #U0 = 4.7560e+02
    #V0 = 2.7285e+02

    ####OAK-D####
    fx = 521.92
    fy = 578.95

    U0 = 318.21
    V0 = 202.92


    # RGB to BGR for pc_points
    imgLU = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

    # depth = inverse of disparity
    #depth = 255 - disp8
    depth = depth_map
    h, w = depth_map.shape

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
    
    #voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.1)

    o3d.visualization.draw_geometries([pcd],
                                    zoom=0.0412,
                                    front=[0.4257, -0.2125, -0.8795],
                                    lookat=[2.6172, 2.0475, 1.532],
                                    up=[-0.0694, -0.9768, 0.2024])
    
    print("Statistical oulier removal")
    #cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=20,
                                                        #td_ratio=2.0)
    #display_inlier_outlier(voxel_down_pcd, ind)

    cv2.destroyAllWindows()