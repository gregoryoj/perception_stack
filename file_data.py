import time

import open3d as o3d
import os
import numpy as np
import matplotlib.pyplot as plt
import open3d.geometry

DISTANCE_TO_LAST_PCD = 0.001
COLORS = ["cool", "Wistia", "Greens", "winter", "PuOr", "seismic"]
PRINT_DEBUG = True

# data = {
#     'Frame': [],
#     'Vehicle_ID': [],
#     'Pos_X': [],
#     'Pos_Y': [],
#     'Pos_Z': [],
#     'BBox_X_Min': [],
#     'BBox_X_Max': [],
#     'BBox_Y_Min': [],
#     'BBox_Y_Max': [],
#     'BBox_Z_Min': [],
#     'BBox_Z_Max': [],
#     'MVec_X': [],
#     'MVec_Y': [],
#     'MVec_Z': []
# }

# def clear_data():
#     d = {
#         'Frame': [],
#         'Vehicle_ID': [],
#         'Pos_X': [],
#         'Pos_Y': [],
#         'Pos_Z': [],
#         'BBox_X_Min': [],
#         'BBox_X_Max': [],
#         'BBox_Y_Min': [],
#         'BBox_Y_Max': [],
#         'BBox_Z_Min': [],
#         'BBox_Z_Max': [],
#         'MVec_X': [],
#         'MVec_Y': [],
#         'MVec_Z': []
#     }
#     return d


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
        #for car_group_cluster in car_group_clusters[i]:
        #    vis.add_geometry(car_group_cluster)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.10)

    vis.run()


def predict_next_bbox(previous_bbox, current_bbox):
    difference_min = current_bbox.min_bound - previous_bbox.min_bound
    difference_max = current_bbox.max_bound - previous_bbox.max_bound
    estimated_next_min = current_bbox.min_bound + difference_min * 1.25  # lenience factor
    estimated_next_max = current_bbox.max_bound + difference_max * 1.25  # lenience factor
    predicted_next_min = []
    predicted_next_max = []
    for i in range(3):
        predicted_next_min.append(min(estimated_next_min[i], estimated_next_max[i]))
        predicted_next_max.append(max(estimated_next_min[i], estimated_next_max[i]))
    return open3d.geometry.AxisAlignedBoundingBox(np.asarray(predicted_next_min), np.asarray(predicted_next_max))


def valid_bbox(bbox):
    min_bound = bbox.min_bound
    max_bound = bbox.max_bound
    if (min_bound != 0).all() and (max_bound != 0).all() and (min_bound != max_bound).any():
        return True
    return False


def main():
    start_image = 458
    end_image = 499
    global data

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

        frame_csv_name = 'frame' + str(i) + '.csv'

        pcd = point_clouds[i]

        car_clusters = []
        bounding_boxes = []

        car_group_clustering = clustering(pcd, epsilon, min_points_per_cluster)

        if epsilon == 10 and i > 415: epsilon = 4.5

        epsilon = 3
        min_points_per_cluster = 5
        o3d.visualization.draw_geometries([pcd])
        car_group_clustering = clustering(pcd, epsilon, min_points_per_cluster)[2:]
        return

        tested_epsilon_value = []

        while len(car_group_clustering) != 3:
            print(epsilon)
            while epsilon in tested_epsilon_value:
                if len(car_group_clustering) < 3 and round(epsilon - 0.05) not in tested_epsilon_value and epsilon - 0.05 > 0:
                    epsilon = round(epsilon - 0.05, 2)
                elif len(car_group_clustering) > 3 or epsilon <= 0:
                    epsilon = round(epsilon + 0.05, 2)

            car_group_clustering = clustering(pcd, epsilon, min_points_per_cluster)
            tested_epsilon_value.append(epsilon)

        if PRINT_DEBUG:
            print("Frame " + str(i + 1))
            print("There were " + str(len(car_group_clustering)) + " clusters detected")
        count = 0
        car_num = 0
        individual_cars = []
        for car_group_cluster in car_group_clustering:
            if i > 1:
                previous_bbox_1 = bounding_boxes_of_cars[i-2][count]
                current_bbox_1 = bounding_boxes_of_cars[i-1][count]
                previous_bbox_2 = bounding_boxes_of_cars[i-2][count + 1]
                current_bbox_2 = bounding_boxes_of_cars[i-1][count+1]
                predicted_next_bbox_1 = predict_next_bbox(previous_bbox_1, current_bbox_1)
                predicted_next_bbox_2 = predict_next_bbox(previous_bbox_2, current_bbox_2)
                count += 2
                if valid_bbox(predicted_next_bbox_1) and valid_bbox(predicted_next_bbox_2):
                    car_1 = open3d.geometry.PointCloud.crop(car_group_cluster, predicted_next_bbox_1)
                    car_2 = open3d.geometry.PointCloud.crop(car_group_cluster, predicted_next_bbox_2)
                    individual_cars = [car_1, car_2]
            no_cars_found = len(individual_cars) == 0 or len(individual_cars[0].points) == 0
            no_cars_found = no_cars_found or len(individual_cars) == 2 and len(individual_cars[1].points) == 0
            if len(individual_cars) == 0 or no_cars_found:
                for j in range(0, 11):
                    individual_cars = clustering(car_group_cluster, 2 - j/10, 1)
                    if len(individual_cars) == 2:
                        break
            if len(individual_cars) != 2:
                bounding_boxes.append(predicted_next_bbox_1)
                bounding_boxes.append(predicted_next_bbox_2)

            for car in individual_cars[:2]:
                if PRINT_DEBUG:
                    print(car_num)
                # data['Frame'].append(i)
                # data['Vehicle_ID'].append(car_num)
                car_clusters.append(car)
                if len(individual_cars) == 2:
                    bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
                        o3d.utility.Vector3dVector(np.asarray(car.points)))
                    
                    # print("BBOX:", bbox)
                    # data['BBox_X_Min'].append(bbox.get_min_bound()[0])
                    # data['BBox_Y_Min'].append(bbox.get_min_bound()[1])
                    # data['BBox_Z_Min'].append(bbox.get_min_bound()[2])
                    # data['BBox_X_Max'].append(bbox.get_max_bound()[0])
                    # data['BBox_Y_Max'].append(bbox.get_max_bound()[1])
                    # data['BBox_Z_Max'].append(bbox.get_max_bound()[2])

                    bounding_boxes.append(bbox)
                car_num += 1

            # df = pd.DataFrame(data)
            # df.to_csv('./perception_results/' + framecsv_name, index=False)
            # data = clear_data()

        if PRINT_DEBUG:
            print("There were " + str(len(car_clusters)) + " cars detected")
            print("There were " + str(len(bounding_boxes)) + " bounding boxes detected")
        car_group_clusters.append(car_group_clustering)
        individual_car_clusters.append(car_clusters)
        bounding_boxes_of_cars.append(bounding_boxes)
        if PRINT_DEBUG:
            print()
        
    visualization(end_image - start_image, car_group_clusters, individual_car_clusters)


if __name__ == "__main__":

    if 'perception_results' not in os.listdir():
        os.mkdir('./perception_results')

    main()
