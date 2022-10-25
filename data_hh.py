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

# Map the activity to offset
activitiesList = ['Bathe','Bed_Toilet_Transition','Cook','Cook_Breakfast','Cook_Dinner',
 'Cook_Lunch','Dress','Drink', 'Eat', 'Eat_Breakfast', 'Eat_Dinner',
 'Eat_Lunch', 'Enter_Home', 'Entertain_Guests', 'Evening_Meds', 'Go_To_Sleep',
 'Groom', 'Leave_Home', 'Morning_Meds', 'Other_Activity', 'Personal_Hygiene',
 'Phone', 'Read', 'Relax', 'Sleep', 'Sleep_Out_Of_Bed', 'Step_Out', 'Toilet',
 'Wake_Up', 'Wash_Breakfast_Dishes', 'Wash_Dinner_Dishes', 'Wash_Dishes',
 'Wash_Lunch_Dishes', 'Watch_TV', 'Work_At_Table']

# cookActivities = {"cairo": {"Other": offset,
#                             "Work": offset + 1,
#                             "Take_medicine": offset + 2,
#                             "Sleep": offset + 3,
#                             "Leave_Home": offset + 4,
#                             "Eat": offset + 5,
#                             "Bed_to_toilet": offset + 6,
#                             "Bathing": offset + 7,
#                             "Enter_home": offset + 8,
#                             "Personal_hygiene": offset + 9,
#                             "Relax": offset + 10,
#                             "Cook": offset + 11},
#                   "kyoto7": {"Other": offset,
#                              "Work": offset + 1,
#                              "Sleep": offset + 2,
#                              "Relax": offset + 3,
#                              "Personal_hygiene": offset + 4,
#                              "Cook": offset + 5,
#                              "Bed_to_toilet": offset + 6,
#                              "Bathing": offset + 7,
#                              "Eat": offset + 8,
#                              "Take_medicine": offset + 9,
#                              "Enter_home": offset + 10,
#                              "Leave_home": offset + 11},
#                   "kyoto8": {"Other": offset,
#                              "Bathing": offset + 1,
#                              "Cook": offset + 2,
#                              "Sleep": offset + 3,
#                              "Work": offset + 4,
#                              "Bed_to_toilet": offset + 5,
#                              "Personal_hygiene": offset + 6,
#                              "Relax": offset + 7,
#                              "Eat": offset + 8,
#                              "Take_medicine": offset + 9,
#                              "Enter_home": offset + 10,
#                              "Leave_home": offset + 11}
#     ,
#                   "kyoto11": {"Other": offset,
#                               "Work": offset + 1,
#                               "Sleep": offset + 2,
#                               "Relax": offset + 3,
#                               "Personal_hygiene": offset + 4,
#                               "Leave_Home": offset + 5,
#                               "Enter_home": offset + 6,
#                               "Eat": offset + 7,
#                               "Cook": offset + 8,
#                               "Bed_to_toilet": offset + 9,
#                               "Bathing": offset + 10,
#                               "Take_medicine": offset + 11},
#                   "milan": {"Other": offset,
#                             "Work": offset + 1,
#                             "Take_medicine": offset + 2,
#                             "Sleep": offset + 3,
#                             "Relax": offset + 4,
#                             "Leave_Home": offset + 5,
#                             "Eat": offset + 6,
#                             "Cook": offset + 7,
#                             "Bed_to_toilet": offset + 8,
#                             "Bathing": offset + 9,
#                             "Enter_home": offset + 10,
#                             "Personal_hygiene": offset + 11},
#                   }

# datasets = ["./hh_dataset/ann_dataset/hh101.ann.txt", "./hh_dataset/ann_dataset/hh102.ann.txt"]
datasets = ["./hh_dataset/ann_dataset/hh102.ann.txt"]

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
                if 'M' == f_info[1][0] or 'D' == f_info[1][0] or 'T' == f_info[1][0]:
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

            except IndexError:
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
        if "T" in key:
            for temp in range(0, int((max(temperature) - min(temperature)) * 2) + 1):
                dictObs[key + str(float(temp / 2.0) + min(temperature))] = count + temp

    XX = []
    YY = []
    X = []
    Y = []
    # XX: create dictionary for sensors in embedded number from the dictObs dict
    # YY: The corresponding acitivity index
    for kk, s in enumerate(sensors):
        if "T" in s:
            XX.append(dictObs[s + str(round(float(values[kk]), 1))])
        else:
            XX.append(dictObs[s + str(values[kk])])
        YY.append(dictActivities[activities[kk]])

    x = []
    # x: the list containing the corresponding sensor sequence for activities at i
    # y: the list containing activities sequence
    for i, y in enumerate(YY):
        if i == 0:
            Y.append(y)
            x = [XX[i]]
        if i > 0:
            if y == YY[i - 1]:
                x.append(XX[i])
            else:
                Y.append(y)
                X.append(x)
                x = [XX[i]]
        if i == len(YY) - 1:
            if y != YY[i - 1]:
                Y.append(y)
            X.append(x)
    return X, Y, dictActivities


def convertActivities(X, Y, dictActivities, activitiesList):
    Yf = Y.copy()
    Xf = X.copy()
    activities = {}
    for i, y in enumerate(Y):
        convertact = [key for key, value in dictActivities.items() if value == y][0]
        Yf[i] = activitiesList.index(convertact)
        activities[convertact] = Yf[i]

    return Xf, Yf, activities


if __name__ == '__main__':
    for filename in datasets:
        datasetName = filename.split("/")[-1].split('.')[0]
        print('Loading ' + datasetName + ' dataset ...')
        X, Y, dictActivities = load_dataset(filename)

        # X, Y, dictActivities = convertActivities(X, Y,
        #                                          dictActivities,
        #                                          activitiesList)

        print(sorted(dictActivities, key=dictActivities.get))
        print("n° instances post-filtering:\t" + str(len(X)))

        print(Counter(Y))

        X = np.array(X, dtype=object)
        Y = np.array(Y, dtype=object)

        X = sequence.pad_sequences(X, maxlen=max_lenght, dtype='int32')
        if not os.path.exists('npy'):
            os.makedirs('npy')

        np.save('./npy/' + datasetName + '-x.npy', X)
        np.save('./npy/' + datasetName + '-y.npy', Y)
        np.save('./npy/' + datasetName + '-labels.npy', dictActivities)


def getData(datasetName):
    np_load_old = np.load

    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

    X = np.load('./npy/' + datasetName + '-x.npy')
    Y = np.load('./npy/' + datasetName + '-y.npy')
    dictActivities = np.load('./npy/' + datasetName + '-labels.npy').item()

    # restore np.load for future normal usage
    np.load = np_load_old

    return X, Y, dictActivities
