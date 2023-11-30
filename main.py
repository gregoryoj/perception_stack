import time

import open3d as o3d
import os
import numpy as np
import matplotlib.pyplot as plt
import open3d.geometry

DISTANCE_TO_LAST_PCD = 0.001


def preprocessing(point_clouds): # todo fix the first frame having 0 points
    point_clouds = point_clouds[1:]
    return point_clouds


def get_distances(last_point_clouds):
    pcd = last_point_clouds[1]
    distance = np.asarray(pcd.compute_point_cloud_distance(last_point_clouds[0]))
    return distance


def main():
    image_to_display = 125
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
        bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(np.asarray(pcd.points)))
        bbox.max_bound = (bbox.max_bound[0], bbox.max_bound[1], bbox.min_bound[2] + 8)
        pcd = open3d.geometry.PointCloud.crop(pcd, bbox)

        if i >= 5: last_point_clouds[0] = o3d.io.read_point_cloud(path_to_data + '/' + files[i - 5])
        last_point_clouds[1] = pcd
        distance_to_last_pcds = get_distances(last_point_clouds)
        indexes_of_changed_points = np.where(distance_to_last_pcds > DISTANCE_TO_LAST_PCD)[0]
        pcd_without_background = pcd.select_by_index(indexes_of_changed_points)

        point_clouds.append(pcd_without_background)

        count += 1
        if count > image_to_display + 1: break

    # preprocessing step
    point_clouds = preprocessing(point_clouds)

    # visualize a single point cloud
    point_cloud = point_clouds[image_to_display]

    # o3d.visualization.draw_geometries(point_clouds)

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    time.sleep(2)

    for i in range(image_to_display):
        vis.clear_geometries()

        geometry = point_clouds[i]
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Warning) as cm:
            labels = np.array(geometry.cluster_dbscan(eps=10, min_points=3, print_progress=False))
        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")
        colors = plt.get_cmap("cool")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0

        geometry.colors = o3d.utility.Vector3dVector(colors[:, :3])
        vis.add_geometry(geometry)
        vis.poll_events()
        vis.update_renderer()

        time.sleep(0.10)

    vis.run()

if __name__ == "__main__":
    main()
