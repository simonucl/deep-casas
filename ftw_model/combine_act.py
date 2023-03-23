from collections import Counter, defaultdict
from datetime import datetime, timedelta
import argparse

import numpy as np
from keras.preprocessing import sequence
import sys
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
import sklearn
import os
import multiprocessing as mp

import sys
sys.path.append('../')
from data_hh import load_dataset
import textdistance

def read_hh_dataset(dataset_path):

    ann_dataset = pd.read_csv(dataset_path, sep='\t')

    raw_columns = ['Date & Time', 'Sensor ID', 'Room-level', 'Sensor location', 'Message', 'Sensor Type']
    ann_columns = raw_columns + ['Activity']

    ann_dataset.columns = ann_columns
    # ann_dataset['Activity'] = ann_dataset['Activity'].apply(lambda x: activity_mapping[x] if x in activity_mapping else x)

    ann_dataset['Date & Time'] = pd.to_datetime(ann_dataset['Date & Time'], format='%Y-%m-%d %H:%M:%S')
    start_time, end_time = ann_dataset['Date & Time'].min(), ann_dataset['Date & Time'].max()
    timeframed_dataset = ann_dataset.set_index(['Date & Time'])

    activity2id = {}
    count = 0
    for act in ann_dataset['Activity'].unique():
        if act != 'Other_Activity':
            activity2id[act] = count
            count += 1
    activity2id['Other_Activity'] = count
    
    return timeframed_dataset, start_time, end_time, activity2id

def similarity(i, j, X_activities):
    return i, j, np.mean([textdistance.levenshtein.normalized_similarity(x, y) for x in X_activities[i] for y in X_activities[j]])

# Write a main function taking in arguments from the command line 
def main(args):
    # Load the dataset
    dataset_no = args.hh_dataset
    dataset_path = '../hh_dataset/hh' + dataset_no + '/hh' + dataset_no + '.ann.txt'
    print('Dataset loaded')
    
    # Load the data
    X, Y, dictActivities = load_dataset(dataset_path)
    print('Data loaded')

    output_path = '../similarity_matrix/hh_' + dataset_no + '/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    y_act = np.array([y for y in dictActivities if y != 'Other_Activity'])

    X_activities = [[x for i, x in enumerate(X) if Y[i] == dictActivities[y]] for y in y_act]
    print([len(x) for x in X_activities])

    X_activities = [[y for y in x if len(y) > np.quantile([len(y) for y in x], 0.2 if len(x) < 600 else 0.5)] for x in X_activities]

    print([len(x) for x in X_activities])
    # Is that any zero length activity?
    print([len(x) for x in X_activities if len(x) == 0])


    # Compute levenshtein distance matrix for activities
    # sim_dist_matrix = np.zeros((len(y_act), len(y_act)))
    # for i in trange(len(y_act)):
    #     for j in range(i, len(y_act)):
    #         # Mean of the distances between the words of the two activities
    #         sim_dist_matrix[i,j] = np.mean([textdistance.levenshtein.distance(x, y) for x in X_activities[i] for y in X_activities[j]])
    #         sim_dist_matrix[j,i] = sim_dist_matrix[i,j]

    # # Save the similarity matrix in pandas dataframe
    # df = pd.DataFrame(sim_dist_matrix, columns=y_act, index=y_act)
    # df.to_csv(output_path + 'sim_dist_matrix.csv')

    # print('Similarity matrix computed')

    
    # Compute level of similarity between activities
    sim_matrix = np.zeros((len(y_act), len(y_act)))
    mp_args = [(i, j, X_activities) for i in range(len(y_act)) for j in range(i, len(y_act))]
    
    with mp.Pool(processes=mp.cpu_count()-5) as pool:
        print ('Computing similarity matrix')

        results = pool.starmap(similarity, mp_args)
        for i, j, sim in results:
            sim_matrix[i,j] = sim
            sim_matrix[j,i] = sim

    # for i in trange(len(y_act)):
    #     for j in range(i, len(y_act)):
    #         # Mean of the distances between the words of the two activities
    #         sim_matrix[i,j] = np.mean([textdistance.levenshtein.normalized_similarity(x, y) for x in X_activities[i] for y in X_activities[j]])
    #         sim_matrix[j,i] = sim_matrix[i,j]
    # Save the similarity matrix in pandas dataframe

    df = pd.DataFrame(sim_matrix, columns=y_act, index=y_act)
    df.to_csv(output_path + 'sim_matrix.csv')

    # Compute normalized distance matrix
    # norm_dist_matrix = np.zeros((len(y_act), len(y_act)))
    # for i in trange(len(y_act)):
    #     for j in range(i, len(y_act)):
    #         # Mean of the distances between the words of the two activities
    #         norm_dist_matrix[i,j] = np.mean([textdistance.levenshtein.normalized_distance(x, y) for x in X_activities[i] for y in X_activities[j]])
    #         norm_dist_matrix[j,i] = norm_dist_matrix[i,j]

    # # Save the similarity matrix in pandas dataframe
    # df = pd.DataFrame(norm_dist_matrix, columns=y_act, index=y_act)
    # df.to_csv(output_path + 'norm_dist_matrix.csv')

# Define behaviour when called from command line
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hh_dataset', type=str, default='1', help='hh dataset number')
    args = parser.parse_args()
    main(args)
    
    