import torch
import numpy as np
import pandas as pd
import torch.nn as nn

class LSTM(nn.Module):
    
    def __init__(self,input_dim,hidden_dim,output_dim,layer_num, is_bidirectional):
        super(LSTM,self).__init__()
        self.hidden_dim = hidden_dim // 2 if is_bidirectional else hidden_dim
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_dim,self.hidden_dim,layer_num,batch_first=False, bidirectional=is_bidirectional)
        self.fc = nn.Linear(hidden_dim,output_dim)
        self.sigmoid = nn.Sigmoid()
        # self.bn = nn.BatchNorm1d(32)
        
    def forward(self,inputs):
        # x = self.bn(inputs)
        lstm_out,(hn,cn) = self.lstm(inputs)
        out = self.fc(lstm_out)
        # print(out.shape)
        return self.sigmoid(out)

class LSTM_1d(nn.Module):
    
    def __init__(self,cnn_dim, input_dim,hidden_dim,output_dim,layer_num, is_bidirectional):
        super(LSTM_1d,self).__init__()
        self.softconv = nn.Conv1d(cnn_dim, 1, kernel_size=1)
        self.hidden_dim = hidden_dim // 2 if is_bidirectional else hidden_dim
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_dim,self.hidden_dim,layer_num,batch_first=False, bidirectional=is_bidirectional)
        self.fc = nn.Linear(hidden_dim,output_dim)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(10)
        
    def forward(self,inputs):
        x = self.bn(inputs)
        feature_1d = self.softconv(inputs).squeeze()
        lstm_out,(hn,cn) = self.lstm(feature_1d)
        out = self.fc(lstm_out)
        # print(out.shape)
        return self.sigmoid(out)
    
class Multi_out_LSTM(nn.Module):

    def __init__(self, cnn_dim, input_dim,hidden_dim,output_dim,layer_num, is_bidirectional):
        super(Multi_out_LSTM, self).__init__()
        self.bn = nn.BatchNorm1d(10)
        self.softconv = nn.Conv1d(cnn_dim, 1, kernel_size=1)
        self.hidden_dim = hidden_dim // 2 if is_bidirectional else hidden_dim
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_dim,self.hidden_dim,layer_num,batch_first=False, bidirectional=is_bidirectional)
        self.out = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=hidden_dim, out_features=output_dim)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,inputs):
        x = self.bn(inputs)
        feature_1d = self.softconv(inputs).squeeze()

        lstm_out,(hn,cn) = self.lstm(feature_1d)
        logits = self.out(lstm_out)
        # print(out.shape)
        return self.sigmoid(logits), logits

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        # if validation_loss < self.min_validation_loss:
        #     self.min_validation_loss = validation_loss
        #     self.counter = 0
        # elif validation_loss > (self.min_validation_loss + self.min_delta):
        #     self.counter += 1
        #     if self.counter >= self.patience:
        #         return True
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        # self.min_validation_loss = validation_loss
        return False