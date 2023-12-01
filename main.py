import time

import open3d as o3d
import os
import numpy as np
import matplotlib.pyplot as plt
import open3d.geometry

DISTANCE_TO_LAST_PCD = 0.001
COLORS = ["cool", "Wistia", "Greens", "winter", "PuOr", "seismic"]


def preprocessing(point_clouds):  # todo fix the first frame having 0 points
    point_clouds = point_clouds[1:]
    return point_clouds


def get_distances(last_point_clouds):
    pcd = last_point_clouds[1]
    distance = np.asarray(pcd.compute_point_cloud_distance(last_point_clouds[0]))
    return distance


def clustering(pcd, epsilon, min_points):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Warning) as cm:
        labels = np.array(pcd.cluster_dbscan(eps=epsilon, min_points=min_points, print_progress=False))
    max_label = labels.max()
    clusters = []
    for i in range(0, max_label + 1):
        cluster = pcd.select_by_index(np.where(labels == i)[0])
        colors = plt.get_cmap(COLORS[i % 6])(np.ones(len(cluster.points)))[:, :3]
        cluster.colors = o3d.utility.Vector3dVector(colors)
        clusters.append(cluster)
    return clusters


def visualization(frame_count, car_group_clusters, car_clusters):
    # visualization step
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    time.sleep(2)

    for i in range(0, frame_count):
        vis.clear_geometries()
        for car_cluster in car_clusters[i]:
            vis.add_geometry(car_cluster)
        for car_group_cluster in car_group_clusters[i]:
            vis.add_geometry(car_group_cluster)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.10)

    vis.run()


def predict_next_bbox(previous_bbox, current_bbox):
    difference_min = current_bbox.min_bound - previous_bbox.min_bound
    difference_max = current_bbox.max_bound - previous_bbox.max_bound
    estimated_next_min = current_bbox.min_bound + difference_min * 1.1  # lenience factor
    estimated_next_max = current_bbox.max_bound + difference_max * 1.1  # lenience factor
    predicted_next_min = []
    predicted_next_max = []
    for i in range(3):
        predicted_next_min.append(min(estimated_next_min[i], estimated_next_max[i]))
        predicted_next_max.append(max(estimated_next_min[i], estimated_next_max[i]))
    return open3d.geometry.AxisAlignedBoundingBox(np.asarray(predicted_next_min), np.asarray(predicted_next_max))


def valid_bbox(bbox):
    min_bound = bbox.min_bound
    max_bound = bbox.max_bound
    if (min_bound != 0).all() and (max_bound != 0).all():
        return True
    return False


def main():
    start_image = 1
    end_image = 10

    # load in point clouds
    path_to_data = "dataset/PointClouds"
    files = sorted(os.listdir(path_to_data), key=lambda x: int(x.split(".")[0]))
    point_clouds = []
    last_point_clouds = [o3d.io.read_point_cloud(path_to_data + '/' + files[0]),
                         o3d.io.read_point_cloud(path_to_data + '/' + files[5])]

    for i in range(start_image, end_image + 1):
        path_to_point_cloud = path_to_data + '/' + files[i]
        pcd = o3d.io.read_point_cloud(path_to_point_cloud)
        bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(np.asarray(pcd.points)))
        bbox.max_bound = (bbox.max_bound[0], bbox.max_bound[1], bbox.min_bound[2] + 8)
        pcd = open3d.geometry.PointCloud.crop(pcd, bbox)

        if i >= 5: last_point_clouds[0] = o3d.io.read_point_cloud(path_to_data + '/' + files[i - 5])
        last_point_clouds[1] = pcd
        distance_to_last_pcds = get_distances(last_point_clouds)
        indexes_of_changed_points = np.where(distance_to_last_pcds > DISTANCE_TO_LAST_PCD)[0]
        pcd_without_background = pcd.select_by_index(indexes_of_changed_points)

        point_clouds.append(pcd_without_background)

    # preprocessing step
    point_clouds = preprocessing(point_clouds)

    # clustering step
    epsilon = 10
    min_points_per_cluster = 3

    car_group_clusters = []
    bounding_boxes_of_cars = []
    individual_car_clusters = []
    for i in range(0, len(point_clouds)):
        # epsilon = 1.2
        # min_points_per_cluster = 5
        pcd = point_clouds[i]

        car_clusters = []
        bounding_boxes = []
        car_group_clustering = clustering(pcd, epsilon, min_points_per_cluster)
        print("Frame 1")
        print("There were " + str(len(car_group_clustering)) + " clusters detected")
        count = 0
        for car_group_cluster in car_group_clustering:
            if i > 1:
                previous_bbox_1 = bounding_boxes_of_cars[i-2][count]
                current_bbox_1 = bounding_boxes_of_cars[i-1][count]
                previous_bbox_2 = bounding_boxes_of_cars[i-2][count + 1]
                current_bbox_2 = bounding_boxes_of_cars[i-1][count+1]
                predicted_next_bbox_1 = predict_next_bbox(previous_bbox_1, current_bbox_1)
                predicted_next_bbox_2 = predict_next_bbox(previous_bbox_2, current_bbox_2)
                count += 2
                if (valid_bbox(predicted_next_bbox_1) and valid_bbox(predicted_next_bbox_2)):
                    car_1 = open3d.geometry.PointCloud.crop(car_group_cluster, predicted_next_bbox_1)
                    car_2 = open3d.geometry.PointCloud.crop(car_group_cluster, predicted_next_bbox_2)
                    individual_cars = [car_1, car_2]
                else:
                    individual_cars = clustering(car_group_cluster, 2, 2)
            else:
                individual_cars = clustering(car_group_cluster, 2, 2)

            for car in individual_cars[:2]:
                car_clusters.append(car)
                bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
                    o3d.utility.Vector3dVector(np.asarray(car.points)))
                bounding_boxes.append(bbox)

        print("There were " + str(len(car_clusters)) + " cars detected")
        car_group_clusters.append(car_group_clustering)
        individual_car_clusters.append(car_clusters)
        bounding_boxes_of_cars.append(bounding_boxes)
        print()
        
    visualization(end_image - start_image, car_group_clusters, individual_car_clusters)


if __name__ == "__main__":
    main()
