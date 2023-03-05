import datetime
import os
import re
from collections import Counter
from datetime import datetime, timedelta

import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import seaborn as sns
import argparse
import json

activity_mapping = {
                              "Cook_Breakfast": "Cook",
                              "Cook_Lunch": "Cook",
                              "Cook_Dinner": "Cook",
                              "Eat_Breakfast": "Eat",
                              "Eat_Lunch": "Eat",
                              "Eat_Dinner": "Eat",
                              "Morning_Meds": "Take_Medicine",
                              "Evening_Meds": "Take_Medicine",
                              "Wash_Breakfast_Dishes": "Wash_Dishes",
                              "Wash_Lunch_Dishes": "Wash_Dishes",
                              "Wash_Dinner_Dishes": "Wash_Dishes",
                              "Work_At_Table": "Work",
                              "Watch_TV": "Relax",
                              "Read": "Work",
                              "Entertain_Guests": "Relax",
                              "Sleep_Out_Of_Bed": "Relax",
                              "Step_Out": "Leave_Home",
                     }
FTWs = [720, 540, 360, 180, 60, 30, 15, 5, 3, 2, 1, 0, 0]
FIB_FTWs = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89][::-1]
FIB_PLUS_FTWs = [0, 1, 1, 2, 3, 5, 8]

ESTWs = [12*(i) for i in range(13)]
SESTWs = [20*(i) for i in range(8)]

# ftw_window = 9

def get_time_window(dataset, start_time, delta, offset):
    return dataset[start_time+offset*delta: start_time+(offset+1)*delta]

def is_triggered_sensor(sensor2id, tup):
    timestamp, sensor, message = tup
    if message in ['ON', 'OPEN']:
        return timestamp, sensor2id[sensor]
    # try:
    #     if int(message) >=50:
    #         return timestamp, sensor2id[sensor]
    # except:
    #     pass
    return timestamp, -1

def compute_sensor_activation(sensor2id, dataset, start_time, end_time, weight):
    sensor_activation = np.zeros(len(sensor2id))
    sensors_detail = list(dataset[start_time: end_time][['Sensor ID', 'Message']].itertuples(name=None))
    # print(sensors_detail)
    for j in sensors_detail:
        timestamp, sensor_triggered = is_triggered_sensor(sensor2id, j)
        if sensor_triggered > -1:
            if weight == 'flat':
                weight = 1
            elif weight == 'left':
                weight = (timestamp-start_time)/(end_time-start_time)
            elif weight == 'right':
                weight = (end_time-timestamp)/(end_time-start_time)
            sensor_activation[sensor_triggered] = weight
    return sensor_activation


def computing_feature_wo(input_file, delta=20):
    ann_dataset = pd.read_csv(input_file, sep='\t')

    raw_columns = ['Date & Time', 'Sensor ID', 'Room-level', 'Sensor location', 'Message', 'Sensor Type']
    ann_columns = raw_columns + ['Activity']

    ann_dataset.columns = ann_columns

    ann_dataset['Activity'] = ann_dataset['Activity'].apply(lambda x: activity_mapping[x] if x in activity_mapping else x)

    activity2id = {}
    count = 0
    for act in ann_dataset['Activity'].unique():
        if act != 'Other_Activity':
            activity2id[act] = count
            count += 1
    activity2id['Other_Activity'] = count
    
    ann_dataset['Date & Time'] = pd.to_datetime(ann_dataset['Date & Time'], format='%Y-%m-%d %H:%M:%S')
    start_time, end_time = ann_dataset['Date & Time'].min(), ann_dataset['Date & Time'].max()
    timeframed_dataset = ann_dataset.set_index(['Date & Time'])

    delta = timedelta(minutes=int(delta))

    number_of_time_window = int(np.ceil((end_time - start_time) / delta))
    activities = np.zeros((number_of_time_window, len(activity2id)-1))
    for i in range(number_of_time_window):
        activity_within_range = timeframed_dataset[start_time+i*delta: start_time+(i+1)*delta]['Activity'].unique()
        for j in activity_within_range:
            if j == "Other_Activity":
                continue
            else:
                activities[i][activity2id[j]] = 1

    sensors_list = list(filter(lambda x : (x[0] != 'T') and (x[0] != 'L'), timeframed_dataset['Sensor ID'].unique()))
    sensor2id = {sensor: i for i , sensor in enumerate(sensors_list)}

    # ftw features (Shape: (number_of_time_window, ftw_window_size, no_sensors))
    features = np.zeros((number_of_time_window, ftw_window, len(sensors_list)))
    for i in trange(number_of_time_window):
        t_star = start_time+(i+1)*delta
        for j in range(ftw_window):
            l4, l3, l2, l1 = FTWs[j:j+4]
            l4, l3, l2, l1 = timedelta(minutes=l4), timedelta(minutes=l3), timedelta(minutes=l2), timedelta(minutes=l1)

            left_slope = compute_sensor_activation(sensor2id, timeframed_dataset, t_star-l2, t_star-l1, weight='left')
            flat_part = compute_sensor_activation(sensor2id, timeframed_dataset, t_star-l3, t_star-l2, weight='flat')
            right_slope = compute_sensor_activation(sensor2id, timeframed_dataset, t_star-l4, t_star-l3, weight='right')

            features[i][j] = np.maximum(np.maximum(left_slope, flat_part), right_slope)

    print(features.shape, activities.shape)

    # feature_out = args.output_path.rsplit('.', 1)[0] + '_feature.npy'
    # activity_out = args.output_path.rsplit('.', 1)[0] + '_activity.npy'
        
    # np.save(feature_out, features)
    # np.save(activity_out, activities)

    return features, activities

def computing_feature(args):
    ann_dataset = pd.read_csv(args.input_file, sep='\t')

    raw_columns = ['Date & Time', 'Sensor ID', 'Room-level', 'Sensor location', 'Message', 'Sensor Type']
    ann_columns = raw_columns + ['Activity']

    ann_dataset.columns = ann_columns

    ann_dataset['Activity'] = ann_dataset['Activity'].apply(lambda x: activity_mapping[x] if x in activity_mapping else x)

    activity2id = {}
    count = 0
    for act in ann_dataset['Activity'].unique():
        if act != 'Other_Activity':
            activity2id[act] = count
            count += 1
    activity2id['Other_Activity'] = count
    
    feature_out = args.output_path.rsplit('.', 1)[0] + '_mapping.json'
    with open(feature_out, 'w') as f:
        json.dump(activity2id, f)

    ann_dataset['Date & Time'] = pd.to_datetime(ann_dataset['Date & Time'], format='%Y-%m-%d %H:%M:%S')
    start_time, end_time = ann_dataset['Date & Time'].min(), ann_dataset['Date & Time'].max()
    timeframed_dataset = ann_dataset.set_index(['Date & Time'])

    delta = timedelta(minutes=int(args.delta))

    number_of_time_window = int(np.ceil((end_time - start_time) / delta))
    activities = np.zeros((number_of_time_window, len(activity2id)-1))
    times = np.zeros((number_of_time_window, 1))

    for i in range(number_of_time_window):
        activity_within_range = timeframed_dataset[start_time+i*delta: start_time+(i+1)*delta]['Activity'].unique()
        times[i] = i* float(args.delta)
        for j in activity_within_range:
            if j == "Other_Activity":
                continue
            else:
                activities[i][activity2id[j]] = 1

    sensors_list = list(filter(lambda x : (x[0] != 'T'), timeframed_dataset['Sensor ID'].unique()))
    sensor2id = {sensor: i for i , sensor in enumerate(sensors_list)}

    ftw_window = None
    if args.window == 'FTWs':
        ftw_window = FTWs
    elif args.window == 'FIB_FTWs':
        ftw_window = FIB_FTWs
    elif args.window == 'ESTWs':
        ftw_window = ESTWs
    elif args.window == 'SESTWs':
        ftw_window = SESTWs
    else:
        raise ValueError (f'Wrong {args.window}')

    ftw_window_len = len(ftw_window)-1 if args.window in ['ESTWs', 'SESTWs'] else len(ftw_window) - 3
    ftw_plus_window_len = 0 if args.window in ['ESTWs', 'SESTWs'] else len(FIB_PLUS_FTWs) - 3 
    # ftw features (Shape: (number_of_time_window, ftw_window_size, no_sensors))
    features = np.zeros((number_of_time_window, ftw_window_len + ftw_plus_window_len, len(sensors_list)))
    for i in trange(number_of_time_window):
        t_star = start_time+(i+1)*delta
        if args.window in ['ESTWs', 'SESTWs']:
            for j in range(ftw_window_len):
                l, r = ftw_window[j:j+2]
                l, r = timedelta(minutes=l), timedelta(minutes=r)

                features[i][j] = compute_sensor_activation(sensor2id, timeframed_dataset, t_star-r, t_star-l, weight='flat')
        else:
            for j in range(ftw_window_len):
                l4, l3, l2, l1 = ftw_window[j:j+4]
                l4, l3, l2, l1 = timedelta(minutes=l4), timedelta(minutes=l3), timedelta(minutes=l2), timedelta(minutes=l1)

                left_slope = compute_sensor_activation(sensor2id, timeframed_dataset, t_star-l2, t_star-l1, weight='left')
                flat_part = compute_sensor_activation(sensor2id, timeframed_dataset, t_star-l3, t_star-l2, weight='flat')
                right_slope = compute_sensor_activation(sensor2id, timeframed_dataset, t_star-l4, t_star-l3, weight='right')

                features[i][j] = np.maximum(np.maximum(left_slope, flat_part), right_slope)
            for j in range(ftw_plus_window_len):
                l4, l3, l2, l1 = FIB_PLUS_FTWs[j:j+4][::-1]
                l4, l3, l2, l1 = timedelta(minutes=l4), timedelta(minutes=l3), timedelta(minutes=l2), timedelta(minutes=l1)

                left_slope = compute_sensor_activation(sensor2id, timeframed_dataset, t_star+l1, t_star+l2, weight='left')
                flat_part = compute_sensor_activation(sensor2id, timeframed_dataset, t_star+l2, t_star+l3, weight='flat')
                right_slope = compute_sensor_activation(sensor2id, timeframed_dataset, t_star+l3, t_star+l4, weight='right')

                features[i][ftw_window_len + j] = np.maximum(np.maximum(left_slope, flat_part), right_slope)
    print(features.shape, activities.shape)

    feature_out = args.output_path.rsplit('.', 1)[0] + '_feature.npy'
    activity_out = args.output_path.rsplit('.', 1)[0] + '_activity.npy'
    times_out = args.output_path.rsplit('.', 1)[0] + '_times.npy'

    np.save(feature_out, features)
    np.save(activity_out, activities)
    np.save(times_out, times)

    return features, activities

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', dest='input_file', action='store', default='', required=True, help='deep model')
    p.add_argument('--delta', action='store', dest='delta', default='10', required=True, help='deep model')
    p.add_argument('--output', dest='output_path', action='store', default='./outputs', required=False)
    p.add_argument('--window', dest='window', action='store', default='FIB_FTWs', required=False)
    # p.add_argument('--cv', dest='need_cv', action='store', default=False, help='whether to do cross validation')
    args = p.parse_args()
    computing_feature(args)