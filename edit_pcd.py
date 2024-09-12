
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


def point_cloud_to_mesh(pcd, depth=9):
    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    # Perform Poisson surface reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    
    return mesh

print("Load a ply point cloud, print it, and render it")
pcd = o3d.io.read_point_cloud("my_point_cloud_1209.ply")


print("Downsample the point cloud with a voxel of 0.02")
voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.01)

print("Radius oulier removal")
cl, ind = voxel_down_pcd.remove_radius_outlier(nb_points=30, radius=20)
display_inlier_outlier(voxel_down_pcd, ind)

print("Statistical oulier removal")
cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=30,
                                                    std_ratio=3.0)
display_inlier_outlier(voxel_down_pcd, ind)

o3d.visualization.draw_geometries([voxel_down_pcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])


depth = 9  # Adjust depth for the level of detail
mesh = point_cloud_to_mesh(voxel_down_pcd, depth)

# Visualize the mesh
o3d.visualization.draw_geometries([mesh])