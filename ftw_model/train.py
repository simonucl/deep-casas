import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import time
import math
import sys
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
import time

from numpy.typing import NDArray
from over_sample import get_minority_instace, MLSMOTE
from model import LSTM, EarlyStopper, LSTM_1d, Multi_out_LSTM, FocalLoss, Multi
import json
import warnings
warnings.filterwarnings('ignore')
import os

import argparse
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold, TimeSeriesSplit, train_test_split
from tqdm import tqdm, trange

seed=42

class ActivityDataset(Dataset):
  """
  This is a custom dataset class. It can get more complex than this, but simplified so you can understand what's happening here without
  getting bogged down by the preprocessing
  """
  def __init__(self, X, Y, t):
    self.X = X
    self.Y = Y
    self.t = t
    if len(self.X) != len(self.Y):
      raise Exception("The length of X does not match the length of Y")

  def __len__(self):
    return len(self.X)

  def __getitem__(self, index):
    # note that this isn't randomly selecting. It's a simple get a single item that represents an x and y
    _x = self.X[index]
    _y = self.Y[index]
    _t = self.t[index]
    return _x, _y, _t
  
def preprocess_features(features, method) -> np.ndarray:
    if method not in ['mean', 'mean_std', 'mean_with_weight', 'mean_std_max_min', 'mean_with_exp_decay']:
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
            weights = [1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 8, 7]
            # out.append(np.sum([vector*((i+1)/len(instance)) for i, vector in enumerate(instance)], axis=0))
            out.append(np.sum([vector*(weights[i]/len(instance)) for i, vector in enumerate(instance)], axis=0))
        return np.array(out)
    elif method == 'mean_with_exp_decay':
        out = []
        lamd = 0.25
        denom = np.sum([np.exp(-lamd * i) for i in range(10)])
        for instance in features:
            out.append(np.sum([vector*(np.exp(-lamd * i) / denom) for i, vector in enumerate(instance)], axis=0))
        return np.array(out)
    return np.array([])

def calculate_metrics(pred, target, threshold=0.3):
    pred = np.array(pred > threshold, dtype=float)
    return {'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro'),
            'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro'),
            'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro'),
            'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro'),
            'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro'),
            'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro'),
            'samples/precision': precision_score(y_true=target, y_pred=pred, average='samples'),
            'samples/recall': recall_score(y_true=target, y_pred=pred, average='samples'),
            'samples/f1': f1_score(y_true=target, y_pred=pred, average='samples'),
            }

def main(args):
    assert args.model in ['LSTM', 'BiLSTM']

    model = args.model

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    features = np.load(args.features)
    activities = np.load(args.activities)
    times = np.load(args.features.rsplit('_', 1)[0] + '_times.npy')

    mapping_file = args.features.rsplit('_', 1)[0] + '_mapping.json'
    with open(mapping_file, 'r') as f:
        activitiy_mapping = json.load(f)
    ensemble_activities = activities.T

    if args.feature_encoding != '1d_cnn' and args.feature_encoding != 'multi-label':
        processed_features = preprocess_features(features, args.feature_encoding)
    else:
        processed_features = features

    input_size = processed_features.shape[-1]

    n_hidden = 512
    n_categories = len(activitiy_mapping) - 1
    n_layer = 3

    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1)

    n_iters = 100
    print_every = n_iters // 10
    plot_every =  n_iters // 10
    batch_size = 32

    tsv = TimeSeriesSplit(n_splits=3)
    kfold = KFold(n_splits=3)
    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []
    accur = []

    # early stopping
    result_file = './result_new1/' + model + '-' + '(' + args.features.rsplit('/')[-1].rsplit('_', 1)[0] + ')' + '_' + args.feature_encoding + '_overlook' + str(args.file_ext) + ('_' + args.delta if args.delta else '') + '_' + str(args.batch)
    if (not os.path.exists(result_file)) or (not os.path.isdir(result_file)):
        os.makedirs(result_file)
    results_dict = []
    early_stopper = EarlyStopper(patience=5 * print_every, min_delta=0)

    if args.feature_encoding == '1d_cnn':
        cnn_weights = []

    for fold, split in enumerate(list(kfold.split(processed_features, activities))):

        results_dict = []

        n_categories = len(activitiy_mapping) - 1

        ensemble_model_results = torch.tensor([])

        new_result_file = os.path.join(result_file, str(fold + 1))
        os.makedirs(new_result_file, exist_ok=True)
        if args.feature_encoding == 'multi-label':

            total_classes = np.sum(activities.T, axis=1)

            medians = [total_classes[k]/len(activities) for k in range(0,n_categories)]
            median_all = np.mean(medians)
            my_freqs = median_all/(np.float64(medians)+(10E-14))

            print(my_freqs.shape)
            rnn = Multi_out_LSTM(processed_features.shape[1], input_size, n_hidden, n_categories, n_layer, (model == 'BiLSTM'))
            print(rnn.to(device))
            learning_rate = 0.0001
            optimizer = optim.Adam(rnn.parameters(),lr=learning_rate, weight_decay=1e-5)
            train, test = list(tsv.split(processed_features, activities))[-1]
            X_train, X_test, y_train, y_test = processed_features[train], processed_features[test], activities[train], activities[test]
            t_train, t_test = times[train], times[test]

            # X_train, X_test, y_train, y_test = train_test_split(processed_features, activities, train_size=0.8, test_size=0.2)

            X_train = np.array(X_train)
            X_test = np.array(X_test)
            y_train = np.array(y_train)
            y_test = np.array(y_test)
            t_train = np.array(t_train)
            t_test = np.array(t_test)

            # print('X_train shape:', processed_features[train].shape)
            # print('y_train shape:', activities[train].shape)
            print('X_train shape:', X_train.shape)
            print('y_train shape:', y_train.shape)
            print('Is CUDA available:', torch.cuda.is_available())

            pred = None

            # result_file.write(activity + '\n')

            # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(my_freqs).to(device))
            criterion = FocalLoss(torch.tensor(my_freqs).to(device))
            best_f1_score = 0
            best_classification_report = None
            best_epoch = 0

            # train_features, train_labels = torch.tensor(processed_features[train], dtype=torch.float32), torch.tensor(activities[train], dtype=torch.float32)
            # dev_features, dev_labels = torch.tensor(processed_features[test], dtype=torch.float32), torch.tensor(activities[test], dtype=torch.float32)
            train_features, train_labels, train_times = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32), torch.tensor(t_train, dtype=torch.float32)
            dev_features, dev_labels, dev_times = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32), torch.tensor(t_test, dtype=torch.float32)
            

            for iter in trange(1, n_iters + 1):
                train_dataset = DataLoader(ActivityDataset(train_features, train_labels, train_times), batch_size=args.batch, shuffle=False)
                dev_dataset = DataLoader(ActivityDataset(dev_features, dev_labels, dev_times), batch_size=len(y_test), shuffle=False)

                rnn.train()
                training_loss = 0
                for features, labels, train_time in train_dataset:
                # train_labels, train_tensor = torch.tensor(train_activities, dtype=torch.float32), torch.tensor(train_features, dtype=torch.float32)
                # dev_labels, dev_tensor = torch.tensor(dev_activities, dtype=torch.float32), torch.tensor(dev_features, dtype=torch.float32)

                    features = features.to(device).float()
                    labels = labels.to(device).float()

                # dev_tensor = dev_tensor.to(device)
                # dev_labels = dev_labels.to(device).unsqueeze(1)
                            
                    output, logits = rnn(features)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    training_loss += (loss.cpu())

                print(training_loss / len(train_dataset))

                validation_loss = None
                #scheduler.step()
                
                for dev_tensor, dev_activities in dev_dataset:
                    dev_tensor = dev_tensor.to(device)
                    dev_labels = dev_activities.to(device)
                    if len(dev_labels.shape) < 2:
                        dev_labels = dev_labels.unsqueeze(1)

                    rnn.eval()
                    with torch.no_grad():
                        prediction, logits = rnn(dev_tensor)
                        
                        pred = prediction.cpu().detach().numpy().round()

                        validation_loss = criterion(logits, dev_labels)
                        result = calculate_metrics(pred, dev_activities.cpu().numpy())
                        
                        #backprop
                        if result['macro/f1'] > best_f1_score:
                            best_f1_score = result['macro/f1']
                            best_classification_report = result
                            best_epoch = iter
                            best_validation_loss = validation_loss

                    if iter%print_every == 0:
                        accur.append(result)
                        
                        # result_file.write("epoch {}\tloss : {}\t accuracy : {}\n".format(iter,loss,acc))
                        print("epoch {}\tloss : {}\t accuracy : {}".format(iter,validation_loss,result))
                        # print(classification_report(dev_activities, pred, digits=4))
                
                current_loss += validation_loss.item()
                # if early_stopper.early_stop(validation_loss):
                #     print("epoch {}\tloss : {}\t accuracy : {}".format(best_epoch,best_validation_loss,best_classification_report))
                #     # result_file.write("Best epoch at " + str(best_epoch) + ". Early stopping at epoch " + str(iter))
                #     break

                if iter % plot_every == 0:
                    all_losses.append(current_loss / plot_every)
                    current_loss = 0
            # result_file.write(str(best_classification_report) + '\n')
            pd.DataFrame(best_classification_report).round(4).to_csv(result_file + 'multi-label.tsv', sep='\t')
        else:
            n_categories = 1
            # train, test = list(tsv.split(processed_features, activities))[-1]

            train, test = split
            X_train, X_test, y_train, y_test = processed_features[train], processed_features[test], activities[train], activities[test]
            t_train, t_test = times[train], times[test]

            # X_train, X_test, y_train, y_test = train_test_split(processed_features, activities, train_size=0.8, test_size=0.2)

            X_train = np.array(X_train)
            X_test = np.array(X_test)
            y_train = np.array(y_train)
            y_test = np.array(y_test)
            t_train = np.array(t_train)
            t_test = np.array(t_test)

            # x_res, y_res = MLSMOTE(*get_minority_instace(pd.DataFrame(X_train), pd.DataFrame(y_train)), 500)
            # X_train = np.concatenate([X_train, x_res])
            # y_train = np.concatenate([y_train, y_res])

            val_losses = []
            models = []
            ensemble_model_results = [None for _ in range(len(activitiy_mapping))]
            # Track the time it takes to train the model
            start = time.time()

            for i, mapping in enumerate(tqdm(list(activitiy_mapping.items())[:-1], total=len(activitiy_mapping) - 1)):

                activity, index = mapping
                
                if args.feature_encoding == '1d_cnn':
                    rnn = LSTM_1d(processed_features.shape[1], input_size, n_hidden, n_categories, n_layer, (model == 'BiLSTM'))
                else:
                    rnn = LSTM(input_size,n_hidden,n_categories,n_layer, (model == 'BiLSTM'))
                early_stopper = EarlyStopper(patience=5 * print_every, min_delta=0)

                best_model = deepcopy(rnn)

                print(rnn.to(device))

                # ones = np.count_nonzero(y_train[:, index])
                # nfreq = torch.tensor().to(device)
                # print(nfreq.shape)
                criterion = nn.BCEWithLogitsLoss()
                learning_rate = 0.0005
                optimizer = optim.Adam(rnn.parameters(),lr=learning_rate, weight_decay=1e-5)
                
                y_train_act = y_train[:, index]
                y_test_act = y_test[:, index]

                print('X_train shape:', X_train.shape)
                print('y_train shape:', y_train_act.shape)
                print('T_train shape:', t_train.shape)

                print('Is CUDA available:', torch.cuda.is_available())

                pred = None

                # result_file.write(activity + '\n')
                ensemble_model_results[i] = np.zeros(len(test))

                best_f1_score = 0
                best_classification_report = None
                best_epoch = 0

                val_loss = []
                train_features, train_labels, train_times = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train_act, dtype=torch.float32), torch.tensor(t_train, dtype=torch.int64)
                dev_features, dev_labels, dev_times = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test_act, dtype=torch.float32), torch.tensor(t_test, dtype=torch.int64)
                for iter in trange(1, n_iters + 1):
                    train_dataset = DataLoader(ActivityDataset(train_features, train_labels, train_times), batch_size=args.batch, shuffle=False)
                    dev_dataset = DataLoader(ActivityDataset(dev_features, dev_labels, dev_times), batch_size=len(test), shuffle=False)

                    rnn.train()
                    training_loss = 0

                    for features, labels, train_time in train_dataset:
                    # train_labels, train_tensor = torch.tensor(train_activities, dtype=torch.float32), torch.tensor(train_features, dtype=torch.float32)
                    # dev_labels, dev_tensor = torch.tensor(dev_activities, dtype=torch.float32), torch.tensor(dev_features, dtype=torch.float32)

                        features = features.to(device).float()
                        labels = labels.to(device).unsqueeze(1).float()
                        train_time = train_time.to(device).float()

                    # dev_tensor = dev_tensor.to(device)
                    # dev_labels = dev_labels.to(device).unsqueeze(1)
                                
                        output, logits = rnn(features, train_time)
                        loss = criterion(logits, labels)
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                        training_loss += loss
                    # print(training_loss / len(train_dataset))
                    validation_loss = None
                    #scheduler.step()
                    for dev_tensor, dev_activities, dev_time in dev_dataset:
                        dev_tensor = dev_tensor.to(device)
                        dev_labels = dev_activities.to(device)
                        dev_time = dev_time.to(device).float()

                        if len(dev_labels.shape) < 2:
                            dev_labels = dev_labels.unsqueeze(1)

                        rnn.eval()
                        with torch.no_grad():
                            prediction, logits = rnn(dev_tensor, dev_time)
                            pred =  prediction.reshape(-1).cpu().detach().numpy().round()

                            validation_loss = criterion(logits, dev_labels)
                            acc = f1_score(dev_activities.cpu(), pred)

                            val_loss.append(validation_loss.item())
                            #backprop
                            if acc > best_f1_score:
                                best_f1_score = acc
                                best_classification_report = classification_report(dev_activities.cpu(), pred, digits=4, output_dict=True)
                                best_epoch = iter
                                ensemble_model_results[i] = pred
                                best_model = deepcopy(rnn)

                        # print(rnn.softconv.weight)
                        # print(rnn.softconv.bias)
                        if iter%print_every == 0:
                            accur.append(acc)
                            # result_file.write("epoch {}\tloss : {}\t accuracy : {}\n".format(iter,loss,acc))
                            print("epoch {}\tloss : {}\t F1-score : {}\t Best-F1 : {} ".format(iter,validation_loss,acc, best_f1_score))
                            # print(classification_report(dev_activities, pred, digits=4))
                    
                    current_loss += validation_loss.item()
                    # if early_stopper.early_stop(validation_loss):
                    #     for param in best_model.parameters():
                    #         param.requires_grad = False
                    #     models.append(best_model)

                    #     # result_file.write("Best epoch at " + str(best_epoch) + ". Early stopping at epoch " + str(iter))
                    #     break

                    if iter % plot_every == 0:
                        all_losses.append(current_loss / plot_every)
                        current_loss = 0
                val_losses.append(val_loss)
                if args.feature_encoding == '1d_cnn':
                    cnn_weights.append(rnn.softconv.weight.data.cpu().numpy())
                if best_classification_report is None:
                    results_dict.append({'activity': activity, 
                                        'positive_P': 0,
                                        'positive_R': 0,
                                        'positive"F1': 0,
                                        "support": 0, 
                                        'macro_P': 0,
                                        'macro_R': 0, 
                                        'macro_F1': 0,
                                        'accuracy': 0})
                else:
                    results_dict.append({'activity': activity, 
                                    'positive_P': best_classification_report['1.0']['precision'],
                                    'positive_R': best_classification_report['1.0']['recall'],
                                    'positive"F1': best_classification_report['1.0']['f1-score'],
                                    "support": best_classification_report['1.0']['support'],
                                    'macro_P': best_classification_report['macro avg']['precision'],
                                    'macro_R': best_classification_report['macro avg']['recall'],
                                    'macro_F1': best_classification_report['macro avg']['f1-score'],
                                    'accuracy': best_classification_report['accuracy']})
            # result_file.write(str(best_classification_report) + '\n')

        end = time.time()
        with open(os.path.join(new_result_file, 'time.txt'), 'w') as f:
            f.write(str(end - start))

        pd.DataFrame(results_dict).round(4).to_csv(os.path.join(new_result_file, 'report.txt'), sep='\t')
        np.save(os.path.join(new_result_file, 'predictions.npy'), np.array(ensemble_model_results))
        np.save(os.path.join(new_result_file, 'activities.npy'), y_test)
        np.save(os.path.join(new_result_file, 'losses.npy'), np.array(val_losses))
        pd.DataFrame(calculate_metrics(pd.DataFrame(map(lambda x : list(x), np.array(ensemble_model_results)[:y_test.shape[1]])).T.to_numpy(), y_test), index=[0]).to_csv(os.path.join(new_result_file, 'report1.txt'), sep='\t')
        if args.feature_encoding == '1d_cnn':
            np.save(os.path.join(new_result_file, 'cnn_weight.npy'), np.array(cnn_weights))
    return models
    
    def multi_label_train(X_train, X_test, y_train, y_test, models):
        batch_size = 16
        n_iters = 100
        total_classes = np.sum(activities.T, axis=1)

        medians = [total_classes[k]/len(activities) for k in range(0,n_categories)]
        median_all = np.mean(medians)
        my_freqs = median_all/(np.float64(medians)+(10E-14))

        model = Multi(len(models), len(models), models)
        learning_rate = 0.0001
        optimizer = optim.Adam(model.parameters(),lr=learning_rate)
        criterion = FocalLoss(torch.tensor(my_freqs).to(device))

        train_features, train_labels = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
        dev_features, dev_labels = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)
        
        for iter in trange(1, n_iters + 1):
            train_dataset = DataLoader(ActivityDataset(train_features, train_labels), batch_size=batch_size, shuffle=False)
            dev_dataset = DataLoader(ActivityDataset(dev_features, dev_labels), batch_size=len(y_test), shuffle=False)
            model.train()
            training_loss = 0
            best_f1_score = 0
            best_classification_report = None

            for features, labels in train_dataset:
            # train_labels, train_tensor = torch.tensor(train_activities, dtype=torch.float32), torch.tensor(train_features, dtype=torch.float32)
            # dev_labels, dev_tensor = torch.tensor(dev_activities, dtype=torch.float32), torch.tensor(dev_features, dtype=torch.float32)

                features = features.to(device).float()
                labels = labels.to(device).float()

                output, logits = model(features)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                validation_loss = None
                #scheduler.step()
                
                current_loss = 0
                
                for dev_tensor, dev_activities in dev_dataset:
                    dev_tensor = dev_tensor.to(device)
                    dev_labels = dev_activities.to(device)
                    if len(dev_labels.shape) < 2:
                        dev_labels = dev_labels.unsqueeze(1)

                    model.eval()
                    with torch.no_grad():
                        prediction, logits = model(dev_tensor)
                        
                        pred = prediction.cpu().detach().numpy().round()

                        validation_loss = criterion(logits, dev_labels)
                        result = calculate_metrics(pred, dev_activities.cpu().numpy())
                        
                        #backprop
                        if result['macro/f1'] > best_f1_score:
                            best_f1_score = result['macro/f1']
                            best_classification_report = result
                            best_epoch = iter
                            best_validation_loss = validation_loss

                    if iter%print_every == 0:
                        accur.append(result)
                        
                        # result_file.write("epoch {}\tloss : {}\t accuracy : {}\n".format(iter,loss,acc))
                        print("epoch {}\tloss : {}\t accuracy : {}".format(iter,validation_loss,result))
                        # print(classification_report(dev_activities, pred, digits=4))
                
                current_loss += validation_loss.item()
                # if early_stopper.early_stop(validation_loss):
                #     print("epoch {}\tloss : {}\t accuracy : {}".format(best_epoch,best_validation_loss,best_classification_report))
                #     # result_file.write("Best epoch at " + str(best_epoch) + ". Early stopping at epoch " + str(iter))
                #     break

                if iter % plot_every == 0:
                    all_losses.append(current_loss / plot_every)
                    current_loss = 0
            # result_file.write(str(best_classification_report) + '\n')
            pd.DataFrame(best_classification_report).round(4).to_csv(result_file + 'multi-label.tsv', sep='\t')
        
        # ensemble_model_results = multi_label_train(X_train, X_test, y_train, y_test, models)



if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', dest='model', action='store', default='', help='deep model')
    p.add_argument('--features', action='store', default='', required=True, help='deep model')
    p.add_argument('--activities', action='store', default='', required=True, help='deep model')
    p.add_argument('--feature_encoding', action='store', default='mean_with_weight', required=True, help='deep model')
    p.add_argument('--delta', action='store', default='20', required=False, help='deep model')
    p.add_argument('--file_ext', action='store', default='', required=False)
    p.add_argument('--batch', action='store', type=int, default=32, required=False)

    # p.add_argument('--cv', dest='need_cv', action='store', default=False, help='whether to do cross validation')
    args = p.parse_args()
    main(args)