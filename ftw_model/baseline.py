import os
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
import sklearn
import argparse

import sys
sys.path.append('../')
from ftw_model.extract_feature import computing_feature_wo

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
# FTWs = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144][::-1]
ftw_window = 10

def read_hh_dataset(dataset_path):

    ann_dataset = pd.read_csv(dataset_path, sep='\t')

    raw_columns = ['Date & Time', 'Sensor ID', 'Room-level', 'Sensor location', 'Message', 'Sensor Type']
    ann_columns = raw_columns + ['Activity']

    ann_dataset.columns = ann_columns
    ann_dataset['Activity'] = ann_dataset['Activity'].apply(lambda x: activity_mapping[x] if x in activity_mapping else x)

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
    
def preprocess_features(features, method) -> np.ndarray:
    if method not in ['mean', 'mean_std', 'mean_with_weight', 'mean_std_max_min']:
        raise ValueError('Please double check the method parameter')

    preprocessed_features = np.array([])
    if method == 'mean':
        return features.mean(axis=1)
    elif method == 'mean_std':
        preprocess_features = np.array(features.mean(axis=1))
        print(preprocess_features.shape)
        preprocess_features = np.concatenate([preprocess_features, features.std(axis=1)], axis=1)
        print(preprocess_features.shape)
        return preprocess_features
    elif method == 'mean_with_weight':
        out = []
        for instance in features:
            out.append(np.sum([vector*((i+1)/10) for i, vector in enumerate(instance)], axis=0))
        return np.array(out)
    return np.array([])

def multi_label_random_forest(X_train, X_test, Y_train, Y_test, activity2id, out):
    rfc = RandomForestClassifier(class_weight='balanced')
    rfc.fit(X_train, Y_train)
    y_pred = rfc.predict(X_test)
    # classification_report(Y_test, y_pred)
    result_df = pd.DataFrame(classification_report(Y_test, y_pred, output_dict=True, digits=4)).rename(columns={str(i): act for i, act in enumerate(list(activity2id.keys())[:-1])}).T
    result_df = result_df.round(4)
    result_df.to_csv(out, sep='\t')

def ensemble_random_forest(X_train, X_test, Y_train, Y_test, activity2id, out):
    output = open(out, 'w+')
    Y_train = Y_train.T
    Y_test = Y_test.T
    tsv = TimeSeriesSplit(n_splits=3)
    # X_train = preprocess_features(X_train, 'mean_with_weight')
    # X_test = preprocess_features(X_test, 'mean_with_weight')

    results_dict = []
    for activity, index in activity2id.items():
        if index == len(activity2id) - 1:
            break
        Y_train_label = Y_train[index]
        Y_test_label = Y_test[index]

        rfc = RandomForestClassifier(class_weight='balanced')
        rfc.fit(X_train, Y_train_label)
        y_pred = rfc.predict(X_test)
        report = classification_report(Y_test_label, y_pred, output_dict=True)
        results_dict.append({'activity': activity, 
                    'positive_P': report['1.0']['precision'],
                    'positive_R': report['1.0']['recall'],
                    'positive"F1': report['1.0']['f1-score'],
                    "support": report['1.0']['support'],
                    'macro_P': report['macro avg']['precision'],
                    'macro_R': report['macro avg']['recall'],
                    'macro_F1': report['macro avg']['f1-score'],
                    'accuracy': report['accuracy']})
# result_file.write(str(best_classification_report) + '\n')
        pd.DataFrame(results_dict).round(4).to_csv(out, sep='\t')
        
    output.close()

def knn(X_train, X_test, Y_train, Y_test, activity2id, out):
    neigh = KNeighborsClassifier(n_neighbors=16)
    neigh.fit(X_train, Y_train)
    y_pred = neigh.predict(X_test)

    result_df = pd.DataFrame(classification_report(Y_test, y_pred, output_dict=True)).rename(columns={str(i): act for i, act in enumerate(list(activity2id.keys())[:-1])}).T
    result_df = result_df.round(4)
    result_df.to_csv(out, sep='\t')

def main(args):
    dataset = args.dataset
    out = os.path.join(args.out, args.model + '_' + dataset + '.txt')
    timeframed_dataset, start_time, end_time, activity2id = read_hh_dataset(f'../hh_dataset/{dataset}/{dataset}.ann.txt')

    features = np.load(f'../hh_dataset/hh_npy/{dataset}_feature.npy')
    activities = np.load(f'../hh_dataset/hh_npy/{dataset}_activity.npy')

    # Splitting the data into training and validation
    tsv = TimeSeriesSplit(n_splits=3)
    processed_features = preprocess_features(features, 'mean_with_weight')
    X = processed_features
    Y = activities
    train, test = list(tsv.split(X, Y))[-1]
    X_train, Y_train, X_test, Y_test = X[train], Y[train], X[test], Y[test]

    if args.model == 'knn':
        knn(X_train, X_test, Y_train, Y_test, activity2id, out)
    elif args.model == 'random_forest':
        multi_label_random_forest(X_train, X_test, Y_train, Y_test, activity2id, out)
    elif args.model == 'ensemble_rf':
        ensemble_random_forest(X_train, X_test, Y_train, Y_test, activity2id, out)
    else:
        raise ValueError ('Wrong input')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', dest='model', action='store', default='', help='deep model')
    p.add_argument('--dataset', action='store', default='', required=True, help='deep model')
    p.add_argument('--out', action='store', default='', required=True, help='deep model')
    # p.add_argument('--feature_encoding', action='store', default='mean_with_weight', required=True, help='deep model')

    args = p.parse_args()
    main(args)