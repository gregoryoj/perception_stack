
from pcd_identifier import identify_cars_from_point_cloud
import os
import pandas as pd

data = {
    'Frame': [],
    'Vehicle_ID': [],
    'Pos_X': [],
    'Pos_Y': [],
    'Pos_Z': [],
    'BBox_X_Min': [],
    'BBox_X_Max': [],
    'BBox_Y_Min': [],
    'BBox_Y_Max': [],
    'BBox_Z_Min': [],
    'BBox_Z_Max': [],
    'MVec_X': [],
    'MVec_Y': [],
    'MVec_Z': []
}

def clear_data():
    d = {
        'Frame': [],
        'Vehicle_ID': [],
        'Pos_X': [],
        'Pos_Y': [],
        'Pos_Z': [],
        'BBox_X_Min': [],
        'BBox_X_Max': [],
        'BBox_Y_Min': [],
        'BBox_Y_Max': [],
        'BBox_Z_Min': [],
        'BBox_Z_Max': [],
        'MVec_X': [],
        'MVec_Y': [],
        'MVec_Z': []
    }
    return d

def main():

    global data
    results = (identify_cars_from_point_cloud())

    for r in results:

        for i in range(0, len(r)):

            index = i%14
            value = round(r[i], 7)

            if index == 0:
                data['Frame'].append(value)
            elif index == 1:
                data['Vehicle_ID'].append(value)
            elif index == 2:
                data['Pos_X'].append(value)
            elif index == 3:
                data['Pos_Y'].append(value)
            elif index == 4:
                data['Pos_Z'].append(value)
            elif index == 5:
                data['BBox_X_Min'].append(value)
            elif index == 6:
                data['BBox_X_Max'].append(value)
            elif index == 7:
                data['BBox_Y_Min'].append(value)
            elif index == 8:
                data['BBox_Y_Max'].append(value)
            elif index == 9:
                data['BBox_Z_Min'].append(value)
            elif index == 10:
                data['BBox_Z_Max'].append(value)
            elif index == 11:
                data['MVec_X'].append(value)
            elif index == 12:
                data['MVec_Y'].append(value)
            elif index == 13:
                data['MVec_Z'].append(value)

        frame_name = "frame_" + str(data['Frame'][0]) + ".csv"
        # print(frame_name)
        df = pd.DataFrame(data)
        df.to_csv('./perception_results/' + frame_name, index=False)
        data = clear_data()


if __name__ == "__main__":

    if 'perception_results' not in os.listdir():
        os.mkdir('./perception_results')

    main()
