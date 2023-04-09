#!/usr/bin/env python3

import datetime
import os
import re
from collections import Counter
from datetime import datetime

import numpy as np
from keras.preprocessing import sequence

offset = 20
max_lenght = 2000

mappingActivities = {
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
    }
                                                                   }
anchor_labels = ['Bathe', 'Enter_Home', 'Wash_Dishes', 'Relax', 'Work', 'Sleep', 'Leave_Home', 'Cook', 'Eat', 'Personal_Hygiene', 'Bed_Toilet_Transition', "Other_Activity"]
dropping_labels = ['Work_On_Computer', 'Work', 'Take_Medicine', 'Work_At_Desk',
        'Go_To_Sleep', 'Wake_Up', 'Exercise', 'Nap', 'Laundry', 'r1.Sleep',
       'r1.Cook_Breakfast', 'r2.Personal_Hygiene', 'r2.Eat_Breakfast',
       'r2.Dress']

# datasets = ["./hh_dataset/ann_dataset/hh101.ann.txt", "./hh_dataset/ann_dataset/hh102.ann.txt"]
datasets = [f"./hh_dataset/hh{str(i)}/hh{str(i)}.ann.txt" for i in range(101, 131) if i != 124]

datasetsNames = [i.split('/')[-1].split('.')[0] for i in datasets]


def load_dataset(filename):
    # dateset fields
    timestamps = []
    sensors = []
    values = []
    activities = []

    activity = ''  # empty

    with open(filename, 'rb') as features:
        database = features.readlines()
        for i, line in enumerate(database):  # each line
            f_info = line.decode().split('\t')  # find fields
            try:
                # TODO Think weather to include the light sensor
                if 'M' == f_info[1][0] or 'D' == f_info[1][0] or 'Control4-Light' == f_info[5]:
                    # choose only M D T sensors, avoiding unexpected errors
                    # if not ('.' in str(np.array(f_info[0])) + str(np.array(f_info[1]))):
                    #     # Avoid errors at the timestamp
                    #     f_info[1] = f_info[1] + '.000000'
                    timestamps.append(datetime.strptime(str(np.array(f_info[0])),
                                                        "%Y-%m-%d %H:%M:%S.%f"))
                    sensors.append(str(np.array(f_info[1])))
                    values.append(str(np.array(f_info[4])))

                    if len(f_info) == 6:  # if activity does not exist
                        activities.append(activity)
                    else:  # if activity exists
                        des = f_info[-1].strip()
                        # if 'begin' in des:
                        #     activity = re.sub('begin', '', des)
                        #     if activity[-1] == ' ':  # if white space at the end
                        #         activity = activity[:-1]  # delete white space
                        #     activities.append(activity)
                        # if 'end' in des:
                        #     activities.append(activity)
                        #     activity = ''
                        activities.append(des)

            except (IndexError, ValueError) as e:
                print(e)
                print(i, line)
    features.close()
    # dictionaries: assigning keys to values
    temperature = []
    for element in values:
        try:
            temperature.append(float(element))
        except ValueError:
            pass
    # Create index for sensors and activities
    sensorsList = sorted(set(sensors))
    dictSensors = {}
    for i, sensor in enumerate(sensorsList):
        dictSensors[sensor] = i
    activityList = sorted(set(activities))
    dictActivities = {}
    for i, activity in enumerate(activityList):
        dictActivities[activity] = i
    valueList = sorted(set(values))
    dictValues = {}
    for i, v in enumerate(valueList):
        dictValues[v] = i
    dictObs = {}
    count = 0
    for key in dictSensors.keys():
        if "M" or "AD" in key:
            dictObs[key + "OFF"] = count
            count += 1
            dictObs[key + "ON"] = count
            count += 1
        if "D" in key:
            dictObs[key + "CLOSE"] = count
            count += 1
            dictObs[key + "OPEN"] = count
            count += 1
        if "LS" in key:
            dictObs[key + "OFF"] = count
            count += 1
            dictObs[key + "ON"] = count
            count += 1

    XX = []
    YY = []
    X = []
    Y = []
    TT = []
    T = []
    # XX: create dictionary for sensors in embedded number from the dictObs dict
    # YY: The corresponding acitivity index
    for kk, s in enumerate(sensors):
        if "L" in s:
            try:
                if int(values[kk]) > 50:
                    XX.append(dictObs[s + 'ON'])
                else:
                    XX.append(dictObs[s + 'OFF'])
            except ValueError:
                continue
        else:
            if kk >= len(values):
                print(kk)
                continue
            if (s + str(values[kk])) not in dictObs.keys():
                continue
            XX.append(dictObs[s + str(values[kk])])
        YY.append(dictActivities[activities[kk]])
        TT.append(timestamps[kk])

    inverse_dictActivities = {v: k for k, v in dictActivities.items()}

    x = []
    t = []
    # x: the list containing the corresponding sensor sequence for activities at i
    # y: the list containing activities sequence
    for i, y in enumerate(YY):
        if i == 0:
            if (inverse_dictActivities[y] not in dropping_labels):
                Y.append(y)
                x = [XX[i]]
                t = [TT[i]]
        if i > 0:
            if y == YY[i - 1]:
                x.append(XX[i])
                t.append(TT[i])
            else:
                if (inverse_dictActivities[y] not in dropping_labels):
                    Y.append(y)
                    X.append(x)
                    T.append((t[0], t[-1]))
                    x = [XX[i]]
                    t = [TT[i]]
                else:
                    x = []
                    t = [TT[i]]
        if i == len(YY) - 1:
            Y.append(y)
            X.append(x)
            T.append((t[0], t[-1]))
    if len(Y) == len(X) + 1:
        Y = Y[:-1]
    assert len(X) == len(Y), f"X: {len(X)}, Y: {len(Y)}"
    assert len(X) == len(T), f"X: {len(X)}, T: {len(T)}"
    assert len(Y) == len(T), f"Y: {len(Y)}, T: {len(T)}"

    print(dictActivities)
    return X, Y, dictActivities, T, dictObs


def convertActivities(X, Y, dictActivities, mapping):
    Yf = Y.copy()
    Xf = X.copy()
    activities = {}
    count = 0
    for i, y in enumerate(Y):
        # convertact = [key for key, value in dictActivities.items() if value == y][0]
        # Yf[i] = activitiesList.index(convertact)
        # activities[convertact] = Yf[i]
        convertact = [key for key, value in dictActivities.items() if value == y][0]
        activity = (mapping[convertact]) if (convertact in mapping) else convertact
        if activity not in activities:
            activities[activity] = count
            count += 1
        Yf[i] = activities[activity]

    inverted_activities = {v: k for k, v in activities.items()}

    assert len(Xf) == len(Yf), f"Xf: {len(Xf)}, Yf: {len(Yf)}"

    Xf = [x for i, x in enumerate(Xf) if inverted_activities[Yf[i]] in anchor_labels]
    Yf = [y for i, y in enumerate(Yf) if inverted_activities[y] in anchor_labels]

    count = 0
    new_actvities = {}
    for i, y in enumerate(Yf):
        activity = inverted_activities[y]
        if activity not in new_actvities:
            new_actvities[activity] = count
            count += 1
        Yf[i] = new_actvities[activity]
    
    activities = new_actvities

    return Xf, Yf, activities


if __name__ == '__main__':
    for filename in datasets:
        datasetName = filename.split("/")[-1].split('.')[0]
        print('Loading ' + datasetName + ' dataset ...')
        X, Y, dictActivities, T, dictOps = load_dataset(filename)

        X, Y, dictActivities = convertActivities(X, Y, dictActivities,
                                                 mapping=mappingActivities[datasetName])

        print(dictActivities)
        print(sorted(dictActivities, key=dictActivities.get))
        print("n° instances post-filtering:\t" + str(len(X)))

        print(Counter(Y))

        X = np.array(X, dtype=object)
        Y = np.array(Y, dtype=object)

        assert len(set(Y)) == len(dictActivities)
        X = sequence.pad_sequences(X, maxlen=max_lenght, dtype='int32')
        if not os.path.exists('npy'):
            os.makedirs('npy')

        np.save('./npy/' + datasetName + '-x.npy', X)
        np.save('./npy/' + datasetName + '-y.npy', Y)
        np.save('./npy/' + datasetName + '-labels.npy', dictActivities)
        np.save('./npy/' + datasetName + '-timestamps.npy', T)


def getData(datasetName):
    np_load_old = np.load

    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

    X = np.load('./npy/' + datasetName + '-x.npy')
    Y = np.load('./npy/' + datasetName + '-y.npy')
    dictActivities = np.load('./npy/' + datasetName + '-labels.npy').item()
    T = np.load('./npy/' + datasetName + '-timestamps.npy')

    # restore np.load for future normal usage
    np.load = np_load_old

    return X, Y, dictActivities, T
