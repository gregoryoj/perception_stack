import time

import open3d as o3d
import os
import numpy as np
import matplotlib.pyplot as plt

DISTANCE_TO_LAST_PCD = 0.0005
DISTANCE_FRAME = 10
COLORS = ["cool", "Wistia", "Greens", "winter", "PuOr", "seismic"]
PRINT_DEBUG = True


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


def bbox_to_corner_points(bbox):
    corner_boxes = np.zeros((8, 3))
    translation = bbox.get_center()
    extent = bbox.get_extent()
    h, w, l = extent / 2

    bounding_box = np.array([
        [-l, -l, l, l, -l, -l, l, l],
        [w, -w, -w, w, w, -w, -w, w],
        [-h, -h, -h, -h, h, h, h, h]])

    rotation_matrix = np.array([
        [np.cos(0), -np.sin(0), 0.0],
        [np.sin(0), np.cos(0), 0.0],
        [0.0, 0.0, 1.0]])

    eight_points = np.tile(translation, (8, 1))
    corner_box = np.dot(rotation_matrix, bounding_box) + eight_points.transpose()
    return corner_box.transpose()


def visualization(point_clouds, car_points_per_frame, bounding_boxes):
    # visualization step
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    time.sleep(2)

    for i in range(0, len(car_points_per_frame)):
        vis.clear_geometries()
        vis.add_geometry(point_clouds[i])
        # for car_cluster in car_points_per_frame[i]:
        #    vis.add_geometry(car_cluster)
        for bounding_box in bounding_boxes[i]:
            vis.add_geometry(bounding_box)
        ctr = vis.get_view_control()
        ctr.set_zoom(0.1)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.1)

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
    return o3d.geometry.AxisAlignedBoundingBox(np.asarray(predicted_next_min), np.asarray(predicted_next_max))


def valid_bbox(bbox):
    min_bound = bbox.min_bound
    max_bound = bbox.max_bound
    if (min_bound != 0).all() and (max_bound != 0).all() and (min_bound != max_bound).any():
        return True
    return False


def preprocessing(point_clouds):  # todo fix the first frame having 0 points

    processed_point_clouds = []
    last_point_clouds = [point_clouds[0], point_clouds[DISTANCE_FRAME]]

    for i in range(0, len(point_clouds)):
        pcd = point_clouds[i]
        pcd_np = np.asarray(pcd.points)
        bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(pcd_np))
        bbox.min_bound = (bbox.min_bound[0], bbox.min_bound[1], bbox.min_bound[2] + 0.5)
        bbox.max_bound = (bbox.max_bound[0], bbox.max_bound[1], bbox.min_bound[2] + 8)
        pcd_cropped = pcd.crop(bbox)

        if i >= DISTANCE_FRAME:
            last_point_clouds[0] = point_clouds[i - DISTANCE_FRAME].crop(bbox)
        last_point_clouds[1] = pcd_cropped
        distance_to_last_pcd = get_distances(last_point_clouds)
        indexes_of_changed_points = np.where(distance_to_last_pcd > DISTANCE_TO_LAST_PCD)[0]
        pcd_without_background = pcd_cropped.select_by_index(indexes_of_changed_points)

        processed_point_clouds.append(pcd_without_background)

    processed_point_clouds = processed_point_clouds[1:]
    return processed_point_clouds


def process_point_clouds(directory_path):
    files = sorted(os.listdir(directory_path), key=lambda x: int(x.split(".")[0]))
    point_clouds = []

    for i in range(0, len(files)):
        path_to_point_cloud = directory_path + '/' + files[i]
        pcd = o3d.io.read_point_cloud(path_to_point_cloud)
        point_clouds.append(pcd)

    return point_clouds


def calculate_change_in_movement(initial_bbox, moved_bbox):
    change_in_min = moved_bbox.min_bound - initial_bbox.min_bound
    change_in_max = moved_bbox.min_bound - initial_bbox.min_bound
    return (change_in_min, change_in_max)


def estimate_next_bbox(old_bbox, change_in_movement):
    leniency = 1.25
    new_min = (old_bbox.min_bound + change_in_movement[0]) * leniency
    new_max = (old_bbox.max_bound + change_in_movement[1]) * leniency
    predicted_next_min = []
    predicted_next_max = []
    for i in range(3):
        predicted_next_min.append(min(new_min[i], new_max[i]))
        predicted_next_max.append(max(new_min[i], new_max[i]))
    predicted_next_min[2] = -1
    predicted_next_max[2] = 5
    return o3d.geometry.AxisAlignedBoundingBox(np.asarray(predicted_next_min), np.asarray(predicted_next_max))


def standardize_bbox_y(bbox):
    bbox.min_bound = np.asarray([bbox.min_bound[0], bbox.min_bound[1], -1])
    bbox.max_bound = np.asarray([bbox.max_bound[0], bbox.max_bound[1], 5])


def get_frame_info_with_clustering(point_clouds):
    epsilon = 10
    min_points_per_cluster = 1

    cars = clustering(point_clouds[480], 2.4, 5)
    car_bboxes = [
        o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(np.asarray(pcd.points))) for
        pcd in cars]

    cars_moving = clustering(point_clouds[479], 2.4, 5)
    car_moved_bboxes = [
        o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(np.asarray(pcd.points))) for
        pcd in cars_moving]

    car_movement = []
    for i in range(0, 6):
        car_movement.append(car_bboxes[i].get_center() - car_moved_bboxes[i].get_center())

    car_movements_per_frame = [car_movement]
    car_bounding_boxes_per_frame = [car_bboxes]
    car_points_per_frame = [cars]

    count = 0
    i = 479
    while i > 0:
        pcd = point_clouds[i]

        car_movements = []
        car_bboxes = []
        car_points = []
        for j in range(0, 6):
            old_car_bbox = car_bounding_boxes_per_frame[count][j]
            predicted_current_bbox = old_car_bbox.get_axis_aligned_bounding_box().translate(
                car_movements_per_frame[count][j])
            standardize_bbox_y(predicted_current_bbox)
            cropped_pcd = pcd.crop(predicted_current_bbox)
            if len(cropped_pcd.points) == 0:
                cropped_pcd.points = o3d.utility.Vector3dVector([predicted_current_bbox.get_center()])
            new_position = clustering(cropped_pcd, epsilon, min_points_per_cluster)
            accurate_movement = new_position[0].get_axis_aligned_bounding_box().get_center() - old_car_bbox.get_center()
            adjusted_new_bbox = old_car_bbox.get_axis_aligned_bounding_box().translate(accurate_movement)
            standardize_bbox_y(adjusted_new_bbox)
            car_movements.append(accurate_movement)
            car_bboxes.append(adjusted_new_bbox)
            car_points.append(cropped_pcd)

        car_movements_per_frame.append(car_movements)
        car_bounding_boxes_per_frame.append(car_bboxes)
        car_points_per_frame.append(car_points)
        i -= 1
        count += 1

    return car_movements_per_frame, car_bounding_boxes_per_frame, car_points_per_frame


def extrapolate_information(car_movements, car_bboxes, car_points):
    for i in range(0, 20):
        car_movement, car_bbox, car_point = [], [], []
        for k in range(0, 6):
            average_movement = car_movements[479 + i][k]
            for j in range(1, 10):
                average_movement += car_movements[479 + i - j][k]
            average_movement = average_movement / 10
            car_movement.append(average_movement)
            car_bbox.append(car_bboxes[479 + i][k].translate(average_movement))
            car_point.append(car_points[479 + i][k].translate(average_movement))
        car_movements.append(car_movement)
        car_bboxes.append(car_bbox)
        car_points.append(car_point)

    return car_movements, car_bboxes, car_points


def categorize_information(car_movements, car_bboxes):
    information = []
    vehicle_id_start = 146
    for i in range(0, 500):
        for j in range(0, 6):
            bbox = car_bboxes[i][j]
            position = bbox.get_center()
            bbox_min = bbox.min_bound
            bbox_max = bbox.max_bound
            information.append(i)
            information.append(vehicle_id_start - j)
            information.append(position[0])
            information.append(position[1])
            information.append(position[2])
            for k in range(0, 3):
                information.append(bbox_min[k])
                information.append(bbox_max[k])
            movement = car_movements[i][j]
            for k in range(0, 3):
                information.append(movement[k])
    return information


def identify_cars_from_point_cloud():
    point_clouds = process_point_clouds("dataset/PointClouds")

    processed_point_clouds = preprocessing(point_clouds)

    car_movements, car_bboxes, car_points = get_frame_info_with_clustering(
        processed_point_clouds)  # all of the per frame info

    car_movements.reverse()
    car_bboxes.reverse()
    car_points.reverse()

    car_movements, car_bboxes, car_points = extrapolate_information(car_movements, car_bboxes, car_points)
    visualization(point_clouds, car_points, car_bboxes)

    print(categorize_information(car_movements, car_bboxes))
    return categorize_information(car_movements, car_bboxes)
