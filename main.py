import time

import open3d as o3d
import os
import numpy as np
import matplotlib.pyplot as plt
import open3d.geometry

DISTANCE_TO_LAST_PCD = 0.001
COLORS = ["cool", "Wistia", "Greens", "winter", "PuOr", "seismic"]

def preprocessing(point_clouds): # todo fix the first frame having 0 points
    point_clouds = point_clouds[1:]
    return point_clouds


def get_distances(last_point_clouds):
    pcd = last_point_clouds[1]
    distance = np.asarray(pcd.compute_point_cloud_distance(last_point_clouds[0]))
    return distance


def main():
    start_image = 420
    end_image = 430

    for i in range(0, 5):
        # load in point clouds
        path_to_data = "dataset/PointClouds"
        files = sorted(os.listdir(path_to_data), key=lambda x: int(x.split(".")[0]))
        point_clouds = []
        last_point_clouds = [o3d.io.read_point_cloud(path_to_data + '/' + files[0]),
                             o3d.io.read_point_cloud(path_to_data + '/' + files[5])]

        count = 0

        for i in range(start_image, len(files)):
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
            if count > end_image + 1: break

        # preprocessing step
        point_clouds = preprocessing(point_clouds)

        # clustering step
        epsilon = 10
        min_points_per_cluster = 3

        car_cluster_info = []
        car_cluster_info_actual = []
        for i in range(0, len(point_clouds)):
            epsilon = 1.5
            min_points_per_cluster = 8
            pcd = point_clouds[i]
            with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Warning) as cm:
                labels = np.array(pcd.cluster_dbscan(eps=epsilon, min_points=min_points_per_cluster, print_progress=False))
            max_label = labels.max()
            print("There are " + str(max_label + 1) + " clusters")
            clusters = []
            for i in range (0, max_label + 1):
                cluster = pcd.select_by_index(np.where(labels == i)[0])
                colors = plt.get_cmap(COLORS[i % 6])(np.ones(len(cluster.points)))[:, :3]
                cluster.colors = o3d.utility.Vector3dVector(colors)
                clusters.append(cluster)
            colors = plt.get_cmap("cool")(labels / (max_label if max_label > 0 else 1))
            colors[labels < 0] = 0
            pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
            car_cluster_info.append(pcd)
            car_cluster_info_actual.append(clusters)

        # visualization step
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        time.sleep(2)

        for i in range(end_image - start_image):
            vis.clear_geometries()
            vis.add_geometry(car_cluster_info[i])
            #for cluster in car_cluster_info_actual[i]:
            #    vis.add_geometry(cluster)
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.10)

        vis.run()

if __name__ == "__main__":
    main()
