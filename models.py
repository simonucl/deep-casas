#!/usr/bin/env python3

from ast import Mod
from pyexpat import model
from unicodedata import name
from keras.layers import Dense, LSTM, Bidirectional, concatenate, Input, Conv1D, Dropout, MaxPool1D, Flatten
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras import Model


def get_LSTM(input_dim, output_dim, max_lenght, no_activities, dropout=0.0):
    model = Sequential(name='LSTM')
    # model.add(Input(shape=(input_dim,)))
    model.add(Embedding(input_dim, output_dim, input_length=max_lenght, mask_zero=True))
    model.add(LSTM(output_dim, dropout=dropout))
    model.add(Dense(no_activities, activation='softmax'))
    return model


def get_biLSTM(input_dim, output_dim, max_lenght, no_activities, dropout=0.0):
    model = Sequential(name='biLSTM')
    model.add(Embedding(input_dim, output_dim, input_length=max_lenght, mask_zero=True))
    model.add(Bidirectional(LSTM(output_dim, dropout=dropout)))
    model.add(Dense(no_activities, activation='softmax'))
    return model


def get_Ensemble2LSTM(input_dim, output_dim, max_lenght, no_activities, dropout=0.0):
    in_layer1 = Input(shape=(input_dim[2],))
    model1 = Embedding(input_dim[1], output_dim, input_length=max_lenght, mask_zero=True)(in_layer1)
    model1 = Bidirectional(LSTM(output_dim, dropout=dropout))(model1)

    in_layer2 = Input(shape=(input_dim[2],))
    model2 = (Embedding(input_dim[1], output_dim, input_length=max_lenght, mask_zero=True))(in_layer2)
    model2 = LSTM(output_dim, dropout=dropout)(model2)

    model = concatenate([model1, model2])

    model = (Dense(no_activities, activation='softmax')) (model)

    return Model(inputs=[in_layer1, in_layer2], outputs=model, name='Ensemble2LSTM')


def get_CascadeEnsembleLSTM(input_dim, output_dim, max_lenght, no_activities, dropout=0.0):
    n_timesteps, n_features = input_dim[1], input_dim[2]
    in_layer1 = Input(shape=(n_features, ))
    model1 = (Embedding(n_timesteps, output_dim, input_length=max_lenght, mask_zero=True))(in_layer1)
    model1 = (Bidirectional(LSTM(output_dim, return_sequences=True, dropout=dropout)))(model1)

    in_layer2 = Input(shape=(n_features, ))
    model2 = (Embedding(input_dim, output_dim, input_length=max_lenght, mask_zero=True))(in_layer2)
    model2 = (LSTM(output_dim, return_sequences=True, dropout=dropout))(model2)

    model = concatenate([model1, model2])
    model = (LSTM(output_dim, dropout=dropout))(model)
    model = (Dense(no_activities, activation='softmax'))(model)
    return Model(inputs=[in_layer1, in_layer2], outputs=model, name='CascadeEnsembleLSTM')


def get_CascadeLSTM(input_dim, output_dim, max_lenght, no_activities, dropout=0.0):
    model = Sequential(name='CascadeLSTM')
    model.add(Embedding(input_dim, output_dim, input_length=max_lenght, mask_zero=True))
    model.add(Bidirectional(LSTM(output_dim, return_sequences=True, dropout=dropout)))
    model.add(LSTM(output_dim, dropout=dropout))
    model.add(Dense(no_activities, activation='softmax'))
    return model

def get_CNN_biLSTM(input_dim, output_dim, max_lenght, no_activities, dropout=0.0, seed=7):
    n_timesteps, n_features = input_dim[1], input_dim[2]
    in_layer1 = Input(shape=(n_features, ))
    model1 = (Conv1D(filters = 64, kernel_size = 3, activation='relu', input_shape=(n_timesteps, n_features)))(in_layer1)
    model1 = (Conv1D(filters = 32, kernel_size = 3, activation='relu', input_shape=(n_timesteps, n_features)))(model1)
    model1 = Dropout(.5, seed=seed)(model1)
    model1 = MaxPool1D(pool_size=2)(model1)
    model1 = Flatten()(model1)

    
    in_layer2 = Input(shape=(n_features, ))
    model2 = (Conv1D(filters = 64, kernel_size = 7, activation='relu', input_shape=(n_timesteps, n_features)))(in_layer2)
    model2 = (Conv1D(filters = 32, kernel_size = 7, activation='relu', input_shape=(n_timesteps, n_features)))(model2)
    model2 = Dropout(.5, seed=seed)(model2)
    model2 = MaxPool1D(pool_size=2)(model2)
    model2 = Flatten()(model2)

    in_layer3 = Input(shape=(n_features, ))
    model3 = (Conv1D(filters = 64, kernel_size = 11, activation='relu', input_shape=(n_timesteps, n_features)))(in_layer3)
    model3 = (Conv1D(filters = 32, kernel_size = 11, activation='relu', input_shape=(n_timesteps, n_features)))(model3)
    model3 = Dropout(.5, seed=seed)(model3)
    model3 = MaxPool1D(pool_size=2)(model3)
    model3 = Flatten()(model3)

    model = concatenate([model1, model2, model3])
    model = (Bidirectional(LSTM(64, return_sequences=True)))(model)
    model = (Bidirectional(LSTM(32, return_sequences=True)))(model)

    model = (Dense(no_activities, activation='softmax'))(model)
    return Model(inputs=[in_layer1, in_layer2, in_layer3], outputs=model, name='CNN-BiLSTM')

def compileModel(model):
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model
