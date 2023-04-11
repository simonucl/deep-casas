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

activity_mapping_dict = {
    "hh101":{
        "Cook_Breakfast": "Cook",
        "Eat_Breakfast": "Eat",
        "Eat_Lunch": "Eat",
        "Eat_Dinner": "Eat",
        "Wash_Lunch_Dishes": "Wash_Dishes",
        "Toilet": "Personal_Hygiene",
        "Read": "Relax",
        "Work_At_Table": "Work",
    },
    "hh102":{
        "Eat": "Other_Activity",
        "Cook_Breakfast": "Cook",
        "Cook_Lunch": "Cook",
        "Cook_Dinner": "Cook",
        "Eat_Breakfast": "Eat",
        "Eat_Lunch": "Eat",
        "Eat_Dinner": "Eat",
        "Wash_Breakfast_Dishes": "Wash_Dishes",
        "Wash_Lunch_Dishes": "Wash_Dishes",
        "Wash_Dinner_Dishes": "Wash_Dishes",
        "Groom" : "Bathe",
        "Work_At_Table": "Work"
    },
    "hh103":{
        "Eat": "Other_Activity",
        "Cook_Breakfast": "Cook",
        "Cook_Lunch": "Cook",
        "Cook_Dinner": "Cook",
        "Eat_Breakfast": "Eat",
        "Eat_Lunch": "Eat",
        "Eat_Dinner": "Eat",
        "Wash_Breakfast_Dishes": "Wash_Dishes",
        "Wash_Lunch_Dishes": "Wash_Dishes",
        "Wash_Dinner_Dishes": "Wash_Dishes",
        "Work_At_Table": "Work",
    },
        "hh104":{
        "Cook_Breakfast": "Cook",
        "Cook_Lunch": "Cook",
        "Cook_Dinner": "Cook",
        "Eat_Breakfast": "Eat",
        "Eat_Lunch": "Eat",
        "Eat_Dinner": "Eat",
        "Wash_Breakfast_Dishes": "Wash_Dishes",
        "Wash_Lunch_Dishes": "Wash_Dishes",
        "Wash_Dinner_Dishes": "Wash_Dishes",
        "Groom" : "Bathe",
        "Work_At_Table": "Work",
        "Work_At_Computer": "Work",
    },        
    "hh105":{
        "Cook_Breakfast": "Cook",
        "Cook_Lunch": "Cook",
        "Cook_Dinner": "Cook",
        "Eat_Breakfast": "Eat",
        "Eat_Lunch": "Eat",
        "Eat_Dinner": "Eat",
        "Wash_Breakfast_Dishes": "Wash_Dishes",
        "Wash_Lunch_Dishes": "Wash_Dishes",
        "Wash_Dinner_Dishes": "Wash_Dishes",
        "Groom" : "Bathe",
        "Work_At_Table": "Work",
    },
    "hh106":{
        "Cook_Breakfast": "Cook",
        "Cook_Lunch": "Cook",
        "Cook_Dinner": "Cook",
        "Eat_Breakfast": "Eat",
        "Eat_Lunch": "Eat",
        "Eat_Dinner": "Eat",
        "Wash_Breakfast_Dishes": "Wash_Dishes",
        "Wash_Lunch_Dishes": "Wash_Dishes",
        "Wash_Dinner_Dishes": "Wash_Dishes",
        "Groom" : "Bathe",
        "Work_At_Table": "Work",
        "Work_At_Computer": "Work",
    },
        "hh107":{
        "Cook_Breakfast": "Cook",
        "Cook_Lunch": "Cook",
        "Cook_Dinner": "Cook",
        "Eat_Breakfast": "Eat",
        "Eat_Lunch": "Eat",
        "Eat_Dinner": "Eat",
        "Wash_Breakfast_Dishes": "Wash_Dishes",
        "Wash_Lunch_Dishes": "Wash_Dishes",
        "Wash_Dinner_Dishes": "Wash_Dishes",
        "Groom" : "Bathe",
        "Work_At_Table": "Work",
        "Work_At_Computer": "Work",
    },
    "hh108":{
        "Cook_Lunch": "Cook",
        "Cook_Dinner": "Cook",
        "Eat_Breakfast": "Eat",
        "Eat_Lunch": "Eat",
        "Wash_Breakfast_Dishes": "Wash_Dishes",
        "Wash_Lunch_Dishes": "Wash_Dishes",
        "Wash_Dinner_Dishes": "Wash_Dishes",
        "Work_At_Table": "Work",
        "Work_At_Computer": "Work",
        "Toliet": "Personal_Hygiene",
    },
    "hh109":{
        "Cook_Lunch": "Cook",
        "Cook_Dinner": "Cook",
        "Cook_Breakfast": "Cook",
        "Eat_Breakfast": "Eat",
        "Eat_Lunch": "Eat",
        "Eat_Dinner": "Eat",
        "Wash_Breakfast_Dishes": "Wash_Dishes",
        "Wash_Lunch_Dishes": "Wash_Dishes",
        "Wash_Dinner_Dishes": "Wash_Dishes",
        "Work_At_Table": "Work",
    },
    "hh110":{
        "Cook_Breakfast": "Cook",
        "Eat_Breakfast": "Eat",
        "Eat_Dinner": "Eat",
        "Wash_Breakfast_Dishes": "Wash_Dishes",
        "Work_At_Table": "Work",
        "Toilet": "Bathe",
    },
    "hh111":{
    "Cook_Breakfast": "Cook",
    "Cook_Lunch": "Cook",
        "Eat_Breakfast": "Eat",
        "Eat_Dinner": "Eat",
        "Wash_Breakfast_Dishes": "Wash_Dishes",
        "Wash_Lunch_Dishes": "Wash_Dishes",
        "Wash_Dinner_Dishes": "Wash_Dishes",
        "Work_At_Desk": "Work",
        "Work_On_Computer": "Work",
},
    "hh112":{
        "Cook_Breakfast": "Cook",
        "Cook_Lunch": "Cook",
        "Cook_Dinner": "Cook",
        "Wash_Breakfast_Dishes": "Wash_Dishes",
        "Wash_Lunch_Dishes": "Wash_Dishes",
        "Wash_Dinner_Dishes": "Wash_Dishes",
        "Work_At_Table": "Work",
        "Take_Medicine": "Personal_Hygiene",
        "Toilet": "Personal_Hygiene",
        "Groom": "Personal_Hygiene",
    },

    "hh113":{
        "Cook_Breakfast": "Cook",
        "Cook_Lunch": "Cook",
        "Cook_Dinner": "Cook",
        "Eat_Breakfast": "Eat",
        "Eat_Lunch": "Eat",
        "Eat_Dinner": "Eat",
        "Wash_Breakfast_Dishes": "Wash_Dishes",
        "Wash_Lunch_Dishes": "Wash_Dishes",
        "Wash_Dinner_Dishes": "Wash_Dishes",
        "Work_At_Table": "Work",
        "Work_At_Computer": "Work",
    },
    "hh114":{
            "Cook_Breakfast": "Cook",
        "Cook_Lunch": "Cook",
        "Cook_Dinner": "Cook",
        "Eat_Breakfast": "Eat",
        "Eat_Lunch": "Eat",
        "Eat_Dinner": "Eat",
        "Wash_Breakfast_Dishes": "Wash_Dishes",
        "Wash_Lunch_Dishes": "Wash_Dishes",
        "Wash_Dinner_Dishes": "Wash_Dishes",
        "Work_At_Table": "Work",
    },
    "hh115":{
        "Cook_Breakfast": "Cook",
        "Cook_Lunch": "Cook",
        "Cook_Dinner": "Cook",
        "Eat_Breakfast": "Eat",
        "Eat_Lunch": "Eat",
        "Eat_Dinner": "Eat",
        "Wash_Breakfast_Dishes": "Wash_Dishes",
        "Wash_Lunch_Dishes": "Wash_Dishes",
        "Wash_Dinner_Dishes": "Wash_Dishes",
        "Work_At_Table": "Work",
        "Work_At_Computer": "Work",
    },

    "hh116":{
        "Cook_Lunch": "Cook",
        "Cook_Dinner": "Cook",
        "Eat_Lunch": "Eat",
        "Eat_Dinner": "Eat",
        "Wash_Breakfast_Dishes": "Wash_Dishes",
        "Wash_Lunch_Dishes": "Wash_Dishes",
        "Wash_Dinner_Dishes": "Wash_Dishes",
        "Work_On_Table": "Work",
        "Groom": "Personal_Hygiene",
        # "Toilet": "Bed_Toilet_Transition",
    },
    "hh117":{
        "Cook_Breakfast": "Cook",
        "Cook_Lunch": "Cook",
        "Cook_Dinner": "Cook",
        "Eat_Breakfast": "Eat",
        "Eat_Lunch": "Eat",
        "Eat_Dinner": "Eat",
        "Wash_Breakfast_Dishes": "Wash_Dishes",
        "Wash_Lunch_Dishes": "Wash_Dishes",
        "Wash_Dinner_Dishes": "Wash_Dishes",
        "Work_At_Table": "Work",
        "Work_At_Computer": "Work",
    },
    "hh118":{
        "Cook_Lunch": "Cook",
        "Cook_Dinner": "Cook",
        "Cok_Breakfast": "Cook",
        "Eat_Lunch": "Eat",
        "Eat_Dinner": "Eat",
        "Eat_Breakfast": "Eat",
        "Eat": "Other",
        "Wash_Breakfast_Dishes": "Wash_Dishes",
        "Wash_Lunch_Dishes": "Wash_Dishes",
        "Wash_Dinner_Dishes": "Wash_Dishes",
        "Work_On_Table": "Work",
        "Work_On_Computer": "Work",
    },
    "hh119":{
        "Cook_Dinner": "Cook",
        "Eat_Dinner": "Eat",
        "Eat_Lunch": "Eat",
        "Wash_Dinner_Dishes": "Wash_Dishes",
    },
    "hh120": {
        "Cook_Breakfast": "Cook",
        "Cook_Lunch": "Cook",
        "Cook_Dinner": "Cook",
        "Eat_Breakfast": "Eat",
        "Eat_Lunch": "Eat",
        "Eat_Dinner": "Eat",
        "Wash_Breakfast_Dishes": "Wash_Dishes",
        "Wash_Lunch_Dishes": "Wash_Dishes",
        "Wash_Dinner_Dishes": "Wash_Dishes",
    },
    "hh121": {
    "Cook_Breakfast": "Cook",
    "Cook_Lunch": "Cook",
    "Cook_Dinner": "Cook",
    "Eat_Breakfast": "Eat",
    "Eat_Lunch": "Eat",
    "Eat_Dinner": "Eat",
    "Wash_Breakfast_Dishes": "Wash_Dishes",
    "Wash_Lunch_Dishes": "Wash_Dishes",
    "Wash_Dinner_Dishes": "Wash_Dishes",
    "Work_At_Table": "Work",
    },
    "hh122": {
        "Eat_Breakfast": "Eat",
        "Eat_Lunch": "Eat",
        "Eat_Dinner": "Eat",
        "Wash_Breakfast_Dishes": "Wash_Dishes",
        "Wash_Lunch_Dishes": "Wash_Dishes",
        "Wash_Dinner_Dishes": "Wash_Dishes",
        "Work_On_Computer": "Work",
        "Work_At_Table": "Work",
    },
    "hh123":{
        "Cook_Breakfast": "Cook",
        "Cook_Lunch": "Cook",
        "Cook_Dinner": "Cook",
        "Eat_Breakfast": "Eat",
        "Eat_Lunch": "Eat",
        "Eat_Dinner": "Eat",
        "Wash_Breakfast_Dishes": "Wash_Dishes",
        "Wash_Lunch_Dishes": "Wash_Dishes",
        "Wash_Dinner_Dishes": "Wash_Dishes",
        "Groom": "Personal_Hygiene",
        "Work_At_Table": "Work",
    },
    "hh124":{
    },
    "hh125":{
    "Eat_Breakfast": "Eat",
    "Eat_Lunch": "Eat",
    "Eat_Dinner": "Eat",
    "Toilet": "Personal_Hygiene",
    },
    "hh126":{
        "Cook_Breakfast": "Cook",
        "Cook_Lunch": "Cook",
        "Cook_Dinner": "Cook",
        "Eat_Breakfast": "Eat",
        "Eat_Lunch": "Eat",
        "Eat_Dinner": "Eat",
        "Wash_Breakfast_Dishes": "Wash_Dishes",
        "Wash_Lunch_Dishes": "Wash_Dishes",
        "Wash_Dinner_Dishes": "Wash_Dishes",
        "Groom": "Bathe",
        "Work_On_Computer": "Work",
    },
    "hh127":
    {
        "Cook_Breakfast": "Cook",
        "Cook_Lunch": "Cook",
        "Cook_Dinner": "Cook",
        "Eat": "Other",
        "Eat_Lunch": "Eat",
        "Eat_Dinner": "Eat",
        "Wash_Breakfast_Dishes": "Wash_Dishes",
        "Wash_Lunch_Dishes": "Wash_Dishes",
        "Wash_Dinner_Dishes": "Wash_Dishes",
        "Work_On_Computer": "Work",
    },
    "hh128":
    {
        "Eat_Breakfast": "Eat",
        "Eat_Lunch": "Eat",
        "Eat_Dinner": "Eat",
        "Work_At_Table": "Work",
    },
    "hh129":
    {   
        "Cook_Breakfast": "Cook",
        "Cook_Lunch": "Cook",
        "Cook_Dinner": "Cook",
        "Eat_Breakfast": "Eat",
        "Eat_Lunch": "Eat",
        "Eat_Dinner": "Eat",
        "Wash_Breakfast_Dishes": "Wash_Dishes",
        "Wash_Lunch_Dishes": "Wash_Dishes",
        "Wash_Dinner_Dishes": "Wash_Dishes",
        "Work_At_Table": "Work",
        "Work_At_Desk": "Work",
        "Toilet": "Personal_Hygiene",
        "Groom": "Personal_Hygiene",
    },
    
    "hh130":
    {
    "Cook_Breakfast": "Cook",
    "Cook_Lunch": "Cook",
    "Cook_Dinner": "Cook",
    "Eat_Breakfast": "Eat",
    "Eat_Lunch": "Eat",
    "Eat_Dinner": "Eat",
    "Wash_Breakfast_Dishes": "Wash_Dishes",
    "Wash_Lunch_Dishes": "Wash_Dishes",
    "Wash_Dinner_Dishes": "Wash_Dishes",
    "Work_On_Computer": "Work",
    }
                                                                   }

dropping_labels = ['Work_On_Computer', 'Take_Medicine', 'Work_At_Desk',
        'Go_To_Sleep', 'Wake_Up', 'Exercise', 'Nap', 'Laundry', 'r1.Sleep',
       'r1.Cook_Breakfast', 'r2.Personal_Hygiene', 'r2.Eat_Breakfast',
       'r2.Dress']

FTWs = [720, 360, 180, 60, 30, 15, 5, 3, 2, 1, 0, 0]
FIB_FTWs = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89][::-1]
FIB_FTWs = FTWs
FIB_PLUS_FTWs = [0, 1, 1, 2, 3, 5]

ESTWs = [12*(i) for i in range(13)]
SESTWs = [20*(i) for i in range(8)]

# ftw_window = 9

def get_time_window(dataset, start_time, delta, offset):
    return dataset[start_time+offset*delta: start_time+(offset+1)*delta]

def is_triggered_sensor(sensor2id, tup):
    timestamp, sensor, message, _ = tup
    if message in ['ON', 'OPEN']:
        return timestamp, sensor2id[sensor]
    try:
        message = int(message)
        if message == 100:
            return timestamp, sensor2id[sensor + 'ON']
        else:
            return timestamp, sensor2id[sensor + 'OFF']
    except:
        pass
    return timestamp, -1

def compute_sensor_activation(sensor2id, dataset, start_time, end_time, weight):
    sensor_activation = np.zeros(len(sensor2id))
    sensors_detail = list(dataset[start_time: end_time][['Sensor ID', 'Message', 'Activity']].itertuples(name=None))
    # print(sensors_detail)
    for j in sensors_detail:
        if j[-1] == 'Other_Activity':
            continue
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

    # ann_dataset['Activity'] = ann_dataset['Activity'].apply(lambda x: activity_mapping[x] if x in activity_mapping else x)

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
    light_sensors = list(timeframed_dataset[timeframed_dataset['Sensor Type'] == 'Control4-Light']['Sensor ID'].unique())
    light_sensors = [x + 'ON' for x in light_sensors] + [x + 'OFF' for x in light_sensors]
    sensors_list = list(set(sensors_list).union(set(light_sensors)))
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
    if args.merged:
        anchor_labels = ['Bathe', 'Enter_Home', 'Wash_Dishes', 'Relax', 'Work', 'Sleep', 'Leave_Home', 'Cook', 'Eat', 'Personal_Hygiene', 'Bed_Toilet_Transition']
    else:
        anchor_labels = None
        
    ann_dataset = pd.read_csv(args.input_file, sep='\t')

    dataset_name = args.input_file.rsplit('/', 1)[1].split('.')[0]

    raw_columns = ['Date & Time', 'Sensor ID', 'Room-level', 'Sensor location', 'Message', 'Sensor Type']
    ann_columns = raw_columns + ['Activity']

    ann_dataset.columns = ann_columns

    if args.merged:
        activity_mapping = activity_mapping_dict[dataset_name]

        ann_dataset['Activity'] = ann_dataset['Activity'].apply(lambda x: activity_mapping[x] if x in activity_mapping else x)

    # filter out the activities that are not in the anchor labels
    if args.merged:
        ann_dataset = ann_dataset[ann_dataset['Activity'].isin(anchor_labels)]

    if dropping_labels:
        ann_dataset = ann_dataset[~ann_dataset['Activity'].isin(dropping_labels)]

    activity2id = {}
    count = 0
    for act in ann_dataset['Activity'].unique():
        print(act)
        if act != 'Other_Activity':
            if anchor_labels:
                if act in anchor_labels:
                    activity2id[act] = count
                    count += 1
            else:
                activity2id[act] = count
                count += 1
    activity2id['Other_Activity'] = count
    
    print(activity2id)
    feature_out = args.output_path.rsplit('.', 1)[0] + '_mapping.json'
    with open(feature_out, 'w') as f:
        json.dump(activity2id, f)

    
    ann_dataset['Date & Time'] = pd.to_datetime(ann_dataset['Date & Time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    # filter out the rows with invalid date
    ann_dataset = ann_dataset[~ann_dataset['Date & Time'].isnull()]
    
    start_time, end_time = ann_dataset['Date & Time'].min(), ann_dataset['Date & Time'].max()
    timeframed_dataset = ann_dataset.set_index(['Date & Time'])

    delta = timedelta(minutes=int(args.delta))

    number_of_time_window = int(np.ceil((end_time - start_time) / delta))
    activities = np.zeros((number_of_time_window, len(activity2id)-1))
    times = np.zeros((number_of_time_window, 1))

    for i in trange(number_of_time_window):
        activity_within_range = timeframed_dataset[start_time+i*delta: start_time+(i+1)*delta]['Activity'].unique()
        times[i] = i * float(args.delta)
        for j in activity_within_range:
            if j == "Other_Activity":
                # activities[i][activity2id[j]] = 1
                continue
            elif anchor_labels and (j not in anchor_labels):
                continue
            else:
                activities[i][activity2id[j]] = 1

    activity_out = args.output_path.rsplit('.', 1)[0] + '_activity.npy'

    np.save(activity_out, activities)


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
    p.add_argument('--merged', dest='merged', action='store', default=False, required=False)
    # p.add_argument('--cv', dest='need_cv', action='store', default=False, help='whether to do cross validation')
    args = p.parse_args()
    computing_feature(args)