#!/usr/bin/env python3

import argparse
import csv
from datetime import datetime

import numpy as np
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.utils import compute_class_weight

import data_hh
import models
import sys
import os
import pandas as pd
import time

# fix random seed for reproducibility
seed = 7
units = 128
epochs = 50
dropout = 0.2

if __name__ == '__main__':
    """The entry point"""
    # set and parse the arguments list
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description='')
    p.add_argument('--v', dest='model', action='store', default='', help='deep model')
    args = p.parse_args()

    print(data_hh.datasetsNames)
    for dataset in data_hh.datasetsNames:

        X, Y, dictActivities, T = data_hh.getData(dataset)

        Y = Y.astype('int') 

        cvaccuracy = []
        cvscores = []
        modelname = ''
        tsv = TimeSeriesSplit(n_splits=3)
        kfold = StratifiedKFold(n_splits=3)
        k = 0
        for train, test in kfold.split(X, Y):
            start_time = time.time()

            print('X_train shape:', X[train].shape)
            print('y_train shape:', Y[train].shape)

            print(dictActivities)
            args_model = str(args.model)

            if 'Ensemble' in args_model:
                input_dim = np.array([X[train], X[train]]).shape
                X_train_input = [X[train], X[train]]
                X_test_input = [X[test], X[test]]
            else:
                input_dim = len(X[train])
                X_train_input = X[train]
                X_test_input = X[test]
            no_activities = len(dictActivities)

            print(no_activities)
            target_names = sorted(dictActivities, key=dictActivities.get)
            print(target_names)

            if args_model == 'LSTM':
                model = models.get_LSTM(input_dim, units, data_hh.max_lenght, no_activities, dropout=dropout)
            elif args_model == 'biLSTM':
                model = models.get_biLSTM(input_dim, units, data_hh.max_lenght, no_activities, dropout=dropout)
            elif args_model == 'Ensemble2LSTM':
                model = models.get_Ensemble2LSTM(input_dim, units, data_hh.max_lenght, no_activities, dropout=dropout)
            elif args_model == 'CascadeEnsembleLSTM':
                model = models.get_CascadeEnsembleLSTM(input_dim, units, data_hh.max_lenght, no_activities, dropout=dropout)
            elif args_model == 'CascadeLSTM':
                model = models.get_CascadeLSTM(input_dim, units, data_hh.max_lenght, no_activities, dropout=dropout)
            else:
                print('Please get the model name '
                      '(eg. --v [LSTM | biLSTM | Ensemble2LSTM | CascadeEnsembleLSTM | CascadeLSTM])')
                exit(-1)

            model = models.compileModel(model)
            # sys.exit(1)
            modelname = model.name

            checkpoint_filepath = './tmp/checkpoint1/'
            output_dir = checkpoint_filepath + model.name + '-' + dataset + '_merged/'
            # check if the directory exists
            os.makedirs(output_dir, exist_ok=True)

            output_dir = output_dir + 'fold' + str(k + 1) + '/'
            os.makedirs(output_dir, exist_ok=True)

            # currenttime = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
            # checkpoint_filepath += model.name + '-' + str(currenttime) + '/'
            # os.mkdir(checkpoint_filepath)

            csv_logger = CSVLogger(
                output_dir + model.name + '-' + dataset + '-fold' + str(k + 1) + '.csv')
            model_checkpoint = ModelCheckpoint(
                output_dir + model.name + '-' + dataset + '-fold' + str(k + 1) + '.hdf5',
                monitor='val_accuracy',
                mode='max',
                save_best_only=True)
            early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.00001)

            # train the model
            print('Begin training ...')
            class_weight = compute_class_weight(class_weight='balanced', classes=np.unique(Y),
                                                y=Y)  # use as optional argument in the fit function

            model.fit(X_train_input, Y[train], validation_split=0.2, epochs=epochs, batch_size=64, verbose=1,
                      callbacks=[early_stopping_callback, csv_logger, model_checkpoint])

            # evaluate the model
            print('Begin testing ...')
            scores = model.evaluate(X_test_input, Y[test], batch_size=64, verbose=1)
            print('%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))

            print('Report:')
            # target_names = sorted(dictActivities, key=dictActivities.get)

            predictions = model.predict(X_test_input, batch_size=64)
            print(predictions.shape)
            classes = np.argmax(predictions, axis=1)


            # TODO modify the predict result
            print(classification_report(list(Y[test]), classes, labels=list(range(len(target_names))),  target_names=target_names))
            print('Confusion matrix:')
            labels = list(dictActivities.values())
            print(confusion_matrix(list(Y[test]), classes, labels=labels))

            # savethe confusion matrix
            pd.DataFrame(confusion_matrix(list(Y[test]), classes, labels=labels)).to_csv(
                output_dir + 'confusion_matrix.csv')
            
            # save the classification report
            pd.DataFrame(classification_report(list(Y[test]), classes, labels=list(range(len(target_names))),  target_names=target_names,
                                                  output_dict=True)).transpose().to_csv(
                output_dir + 'classification_report.csv')
            
            # save the predictions using npy
            np.save(output_dir + 'predictions.npy', predictions)
            # save the gold labels using npy
            np.save(output_dir + 'gold_labels.npy', Y[test])

            # Save the predictions timestamp
            np.save(output_dir + 'predictions_timestamp.npy', T[test])

            end_time = time.time()
            # measure the time
            with open(output_dir + 'time.txt', 'w') as f:
                f.write(str(end_time - start_time) + ' seconds\n')

            cvaccuracy.append(scores[1] * 100)
            cvscores.append(scores)

            k += 1

        print('{:.2f}% (+/- {:.2f}%)'.format(np.mean(cvaccuracy), np.std(cvaccuracy)))

        currenttime = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
        csvfile = 'cv-scores-' + modelname + '-' + dataset + '-' + str(currenttime) + '.csv'

        with open(csvfile, "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            for val in cvscores:
                writer.writerow([",".join(str(el) for el in val)])
