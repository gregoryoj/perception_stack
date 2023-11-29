import time

import open3d as o3d
import os
import numpy as np
import matplotlib.pyplot as plt


def preprocessing(point_clouds):
    return point_clouds


def get_distances(last_point_clouds):
    pcd = last_point_clouds[1]
    distance = np.asarray(pcd.compute_point_cloud_distance(last_point_clouds[0]))
    return distance


def main():
    image_to_display = 250
    # load in point clouds
    path_to_data = "dataset/PointClouds"
    files = sorted(os.listdir(path_to_data), key=lambda x: int(x.split(".")[0]))
    point_clouds = []
    last_point_clouds = [o3d.io.read_point_cloud(path_to_data + '/' + files[0]),
                         o3d.io.read_point_cloud(path_to_data + '/' + files[5])]

    count = 0

    for i in range(0, len(files)):
        path_to_point_cloud = path_to_data + '/' + files[i]
        pcd = o3d.io.read_point_cloud(path_to_point_cloud)

        if i >= 5: last_point_clouds[0] = o3d.io.read_point_cloud(path_to_data + '/' + files[i - 5])
        last_point_clouds[1] = pcd
        distance_to_last_pcds = get_distances(last_point_clouds)
        indexes_of_changed_points = np.where(distance_to_last_pcds > 0.00000001)[0]
        pcd_without_background = pcd.select_by_index(indexes_of_changed_points)

        point_clouds.append(pcd)

        count += 1
        if count > image_to_display + 1: break

    # preprocessing step
    point_clouds = preprocessing(point_clouds)

    # visualize a single point cloud
    point_cloud = point_clouds[image_to_display]

    # o3d.visualization.draw_geometries(point_clouds)

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # geometry is the point cloud used in your animaiton
    geometry = point_clouds[1]
    vis.add_geometry(geometry)

    time.sleep(2)

    for i in range(image_to_display):
        geometry.points = point_clouds[i].points
        vis.update_geometry(geometry)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.10)

    vis.run()

    # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    #    labels = np.array(point_cloud.cluster_dbscan(eps=1, min_points=10, print_progress=True))


"""    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    point_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([point_cloud],
                                      zoom=0.455,
                                      front=[-0.4999, -0.1659, -0.8499],
                                      lookat=[2.1813, 2.0619, 2.0999],
                                      up=[0.1204, -0.9852, 0.1215])
"""

if __name__ == "__main__":
    main()
