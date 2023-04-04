import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.distributions.weibull import Weibull
from torch import Tensor
from typing import Optional, Sequence
from torch.nn import functional as F
from time2vec import SineActivation, CosineActivation, TimeEncoding

class LSTM(nn.Module):
    
    def __init__(self,input_dim,hidden_dim,output_dim,layer_num, is_bidirectional):
        super(LSTM,self).__init__()
        self.hidden_dim = hidden_dim // 2 if is_bidirectional else hidden_dim
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_dim,self.hidden_dim,layer_num,batch_first=False, bidirectional=is_bidirectional)
        self.time2vec = SineActivation(1, input_dim)

        self.fc = nn.Linear(hidden_dim,output_dim)
        self.sigmoid = nn.Sigmoid()
        # self.bn = nn.BatchNorm1d(32)
        
    def forward(self,inputs, times):
        # x = self.bn(inputs)
        times_vec = self.time2vec(times)
        inputs = torch.add(inputs, 0.002 * times_vec)
        lstm_out,(hn,cn) = self.lstm(inputs)
        out = self.fc(lstm_out)
        # print(out.shape)
        return self.sigmoid(out), out

# Exp: 0.002 lr
class LSTM_1d(nn.Module):
    
    def __init__(self,cnn_dim, input_dim,hidden_dim,output_dim,layer_num, is_bidirectional):
        super(LSTM_1d,self).__init__()
        # (4456, 10, 56) -> (4456, 56)
        self.softconv = nn.Conv1d(cnn_dim, 1, kernel_size=1) # 10
        # self.dropout = nn.Dropout(0.2)
        self.time2vec = SineActivation(1, input_dim)
        # print(self.softconv.bias.shape)
        self.softconv.weight = nn.Parameter(Weibull(torch.tensor([1.0]), torch.tensor([3.0])).sample((1, cnn_dim)))
        # self.softconv.bias = torch.nn.Parameter(torch.zeros(1))
        # nn.init.normal_(self.softconv, 0, 1)
        self.hidden_dim = hidden_dim // 2 if is_bidirectional else hidden_dim
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_dim,self.hidden_dim,layer_num,batch_first=True, bidirectional=is_bidirectional, dropout=0.4)
        self.fc = nn.Linear(hidden_dim,output_dim)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(10)
        
    def forward(self, inputs, times):
        # x = self.bn(inputs)
        
        feature_1d = self.softconv(inputs).squeeze()
        times_vec = self.time2vec(times)
        # print(feature_1d)
        # print(times_vec)
        feature_1d = torch.add(feature_1d, 0.002 * times_vec)

        lstm_out,(hn,cn) = self.lstm(feature_1d)
        out = self.fc(lstm_out)
        # print(out.shape)
        return self.sigmoid(out), out
    
# Exp: No time2vec
class LSTM_1d_2(nn.Module):
        
    def __init__(self,cnn_dim, input_dim,hidden_dim,output_dim,layer_num, is_bidirectional):
        super(LSTM_1d_2,self).__init__()
        # (4456, 10, 56) -> (4456, 56)
        self.softconv = nn.Conv1d(cnn_dim, 1, kernel_size=1) # 10
        # self.time2vec = SineActivation(1, input_dim)
        # print(self.softconv.bias.shape)
        self.softconv.weight = nn.Parameter(Weibull(torch.tensor([1.0]), torch.tensor([3.0])).sample((1, cnn_dim)))
        # self.softconv.bias = torch.nn.Parameter(torch.zeros(1))
        # nn.init.normal_(self.softconv, 0, 1)
        self.hidden_dim = hidden_dim // 2 if is_bidirectional else hidden_dim
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_dim,self.hidden_dim,layer_num,batch_first=False, bidirectional=is_bidirectional, dropout=0.4)
        self.fc = nn.Linear(hidden_dim,output_dim)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(10)
        
    def forward(self, inputs, times):
        # x = self.bn(inputs)
        feature_1d = self.softconv(inputs).squeeze()
        # times_vec = self.time2vec(times)
        # print(feature_1d)
        # print(times_vec)
        # feature_1d = torch.add(feature_1d, 0.002 * times_vec)

        lstm_out,(hn,cn) = self.lstm(feature_1d)
        out = self.fc(lstm_out)
        # print(out.shape)
        return self.sigmoid(out), out
        
# 0.005 weight
class LSTM_1d_3(nn.Module):
                
    def __init__(self,cnn_dim, input_dim,hidden_dim,output_dim,layer_num, is_bidirectional):
        super(LSTM_1d_3,self).__init__()
        # (4456, 10, 56) -> (4456, 56)
        self.softconv = nn.Conv1d(cnn_dim, 1, kernel_size=1) # 10
        self.time2vec = SineActivation(1, input_dim)
        self.time2vec1 = SineActivation(1, input_dim, pattern=86400)

        # print(self.softconv.bias.shape)
        self.softconv.weight = nn.Parameter(Weibull(torch.tensor([1.0]), torch.tensor([3.0])).sample((1, cnn_dim)))
        # self.softconv.bias = torch.nn.Parameter(torch.zeros(1))
        # nn.init.normal_(self.softconv, 0, 1)
        self.hidden_dim = hidden_dim // 2 if is_bidirectional else hidden_dim
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_dim,self.hidden_dim,layer_num,batch_first=False, bidirectional=is_bidirectional, dropout=0.4)
        self.fc = nn.Linear(hidden_dim,output_dim)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(10)
        
    def forward(self, inputs, times):
        # x = self.bn(inputs)
        feature_1d = self.softconv(inputs).squeeze()
        times_vec1 = self.time2vec(times)
        times_vec2 = self.time2vec1(times)

        # print(feature_1d)
        # print(times_vec)
        feature_1d = torch.add(feature_1d, 0.005 * times_vec1)

        lstm_out,(hn,cn) = self.lstm(feature_1d)
        out = self.fc(lstm_out)
        # print(out.shape)
        return self.sigmoid(out), out
    
# Day and week pattern
class LSTM_1d_4(nn.Module):
                
    def __init__(self,cnn_dim, input_dim,hidden_dim,output_dim,layer_num, is_bidirectional):
        super(LSTM_1d_4,self).__init__()
        # (4456, 10, 56) -> (4456, 56)
        self.softconv = nn.Conv1d(cnn_dim, 1, kernel_size=1) # 10
        self.time2vec = SineActivation(1, input_dim)
        self.time2vec1 = SineActivation(1, input_dim, pattern=86400)

        # print(self.softconv.bias.shape)
        self.softconv.weight = nn.Parameter(Weibull(torch.tensor([1.0]), torch.tensor([3.0])).sample((1, cnn_dim)))
        # self.softconv.bias = torch.nn.Parameter(torch.zeros(1))
        # nn.init.normal_(self.softconv, 0, 1)
        self.hidden_dim = hidden_dim // 2 if is_bidirectional else hidden_dim
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_dim,self.hidden_dim,layer_num,batch_first=False, bidirectional=is_bidirectional, dropout=0.4)
        self.fc = nn.Linear(hidden_dim,output_dim)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(10)
        
    def forward(self, inputs, times):
        # x = self.bn(inputs)
        feature_1d = self.softconv(inputs).squeeze()
        times_vec1 = self.time2vec(times)
        times_vec2 = self.time2vec1(times)

        # print(feature_1d)
        # print(times_vec)
        feature_1d = torch.add(feature_1d, 0.002 * times_vec1)
        feature_1d = torch.add(feature_1d, 0.0001 * times_vec2)

        lstm_out,(hn,cn) = self.lstm(feature_1d)
        out = self.fc(lstm_out)
        # print(out.shape)
        return self.sigmoid(out), out
    
# 0.001 weight
class LSTM_1d_5(nn.Module):
                
    def __init__(self,cnn_dim, input_dim,hidden_dim,output_dim,layer_num, is_bidirectional):
        super(LSTM_1d_5,self).__init__()
        # (4456, 10, 56) -> (4456, 56)
        self.softconv = nn.Conv1d(cnn_dim, 1, kernel_size=1) # 10
        self.time2vec = SineActivation(1, input_dim)
        self.time2vec1 = SineActivation(1, input_dim, pattern=86400)

        # print(self.softconv.bias.shape)
        self.softconv.weight = nn.Parameter(Weibull(torch.tensor([1.0]), torch.tensor([3.0])).sample((1, cnn_dim)))
        # self.softconv.bias = torch.nn.Parameter(torch.zeros(1))
        # nn.init.normal_(self.softconv, 0, 1)
        self.hidden_dim = hidden_dim // 2 if is_bidirectional else hidden_dim
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_dim,self.hidden_dim,layer_num,batch_first=False, bidirectional=is_bidirectional, dropout=0.4)
        self.fc = nn.Linear(hidden_dim,output_dim)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(10)
        
    def forward(self, inputs, times):
        # x = self.bn(inputs)
        feature_1d = self.softconv(inputs).squeeze()
        times_vec1 = self.time2vec(times)
        times_vec2 = self.time2vec1(times)

        # print(feature_1d)
        # print(times_vec)
        feature_1d = torch.add(feature_1d, 0.001 * times_vec1)

        lstm_out,(hn,cn) = self.lstm(feature_1d)
        out = self.fc(lstm_out)
        # print(out.shape)
        return self.sigmoid(out), out
    
# 0.01 weight
class LSTM_1d_6(nn.Module):
                
    def __init__(self,cnn_dim, input_dim,hidden_dim,output_dim,layer_num, is_bidirectional):
        super(LSTM_1d_6,self).__init__()
        # (4456, 10, 56) -> (4456, 56)
        self.softconv = nn.Conv1d(cnn_dim, 1, kernel_size=1) # 10
        self.time2vec = SineActivation(1, input_dim)
        self.time2vec1 = SineActivation(1, input_dim, pattern=86400)

        # print(self.softconv.bias.shape)
        self.softconv.weight = nn.Parameter(Weibull(torch.tensor([1.0]), torch.tensor([3.0])).sample((1, cnn_dim)))
        # self.softconv.bias = torch.nn.Parameter(torch.zeros(1))
        # nn.init.normal_(self.softconv, 0, 1)
        self.hidden_dim = hidden_dim // 2 if is_bidirectional else hidden_dim
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_dim,self.hidden_dim,layer_num,batch_first=False, bidirectional=is_bidirectional, dropout=0.4)
        self.fc = nn.Linear(hidden_dim,output_dim)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(10)
        
    def forward(self, inputs, times):
        # x = self.bn(inputs)
        feature_1d = self.softconv(inputs).squeeze()
        times_vec1 = self.time2vec(times)
        times_vec2 = self.time2vec1(times)

        # print(feature_1d)
        # print(times_vec)
        feature_1d = torch.add(feature_1d, 0.01 * times_vec1)

        lstm_out,(hn,cn) = self.lstm(feature_1d)
        out = self.fc(lstm_out)
        # print(out.shape)
        return self.sigmoid(out), out
    
# time-encoding 0.001
class LSTM_1d_7(nn.Module):
                
    def __init__(self,cnn_dim, input_dim,hidden_dim,output_dim,layer_num, is_bidirectional):
        super(LSTM_1d_7,self).__init__()
        # (4456, 10, 56) -> (4456, 56)
        self.softconv = nn.Conv1d(cnn_dim, 1, kernel_size=1) # 10
        self.time2vec = TimeEncoding(1, input_dim)

        # print(self.softconv.bias.shape)
        self.softconv.weight = nn.Parameter(Weibull(torch.tensor([1.0]), torch.tensor([3.0])).sample((1, cnn_dim)))
        # self.softconv.bias = torch.nn.Parameter(torch.zeros(1))
        # nn.init.normal_(self.softconv, 0, 1)
        self.hidden_dim = hidden_dim // 2 if is_bidirectional else hidden_dim
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_dim,self.hidden_dim,layer_num,batch_first=False, bidirectional=is_bidirectional, dropout=0.4)
        self.fc = nn.Linear(hidden_dim,output_dim)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(10)
        
    def forward(self, inputs, times):
        # x = self.bn(inputs)
        feature_1d = self.softconv(inputs).squeeze()
        times_vec1 = self.time2vec(times)

        # print(feature_1d)
        # print(times_vec)
        feature_1d = torch.add(feature_1d, 0.001 * times_vec1)

        lstm_out,(hn,cn) = self.lstm(feature_1d)
        out = self.fc(lstm_out)
        # print(out.shape)
        return self.sigmoid(out), out
    
# time-encoding 0.002
class LSTM_1d_8(nn.Module):
                
    def __init__(self,cnn_dim, input_dim,hidden_dim,output_dim,layer_num, is_bidirectional):
        super(LSTM_1d_8,self).__init__()
        # (4456, 10, 56) -> (4456, 56)
        self.softconv = nn.Conv1d(cnn_dim, 1, kernel_size=1) # 10
        self.time2vec = TimeEncoding(1, input_dim)

        # print(self.softconv.bias.shape)
        self.softconv.weight = nn.Parameter(Weibull(torch.tensor([1.0]), torch.tensor([3.0])).sample((1, cnn_dim)))
        # self.softconv.bias = torch.nn.Parameter(torch.zeros(1))
        # nn.init.normal_(self.softconv, 0, 1)
        self.hidden_dim = hidden_dim // 2 if is_bidirectional else hidden_dim
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_dim,self.hidden_dim,layer_num,batch_first=False, bidirectional=is_bidirectional, dropout=0.4)
        self.fc = nn.Linear(hidden_dim,output_dim)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(10)
        
    def forward(self, inputs, times):
        # x = self.bn(inputs)
        feature_1d = self.softconv(inputs).squeeze()
        times_vec1 = self.time2vec(times)

        # print(feature_1d)
        # print(times_vec)
        feature_1d = torch.add(feature_1d, 0.002 * times_vec1)

        lstm_out,(hn,cn) = self.lstm(feature_1d)
        out = self.fc(lstm_out)
        # print(out.shape)
        return self.sigmoid(out), out

# time-encoding 0.005
class LSTM_1d_9(nn.Module):
                
    def __init__(self,cnn_dim, input_dim,hidden_dim,output_dim,layer_num, is_bidirectional):
        super(LSTM_1d_9,self).__init__()
        # (4456, 10, 56) -> (4456, 56)
        self.softconv = nn.Conv1d(cnn_dim, 1, kernel_size=1) # 10
        self.time2vec = TimeEncoding(1, input_dim)

        # print(self.softconv.bias.shape)
        self.softconv.weight = nn.Parameter(Weibull(torch.tensor([1.0]), torch.tensor([3.0])).sample((1, cnn_dim)))
        # self.softconv.bias = torch.nn.Parameter(torch.zeros(1))
        # nn.init.normal_(self.softconv, 0, 1)
        self.hidden_dim = hidden_dim // 2 if is_bidirectional else hidden_dim
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_dim,self.hidden_dim,layer_num,batch_first=False, bidirectional=is_bidirectional, dropout=0.4)
        self.fc = nn.Linear(hidden_dim,output_dim)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(10)
        
    def forward(self, inputs, times):
        # x = self.bn(inputs)
        feature_1d = self.softconv(inputs).squeeze()
        times_vec1 = self.time2vec(times)

        # print(feature_1d)
        # print(times_vec)
        feature_1d = torch.add(feature_1d, 0.005 * times_vec1)

        lstm_out,(hn,cn) = self.lstm(feature_1d)
        out = self.fc(lstm_out)
        # print(out.shape)
        return self.sigmoid(out), out
    
# time-encoding 0.01
class LSTM_1d_10(nn.Module):
                
    def __init__(self,cnn_dim, input_dim,hidden_dim,output_dim,layer_num, is_bidirectional):
        super(LSTM_1d_10,self).__init__()
        # (4456, 10, 56) -> (4456, 56)
        self.softconv = nn.Conv1d(cnn_dim, 1, kernel_size=1) # 10
        self.time2vec = TimeEncoding(1, input_dim)

        # print(self.softconv.bias.shape)
        self.softconv.weight = nn.Parameter(Weibull(torch.tensor([1.0]), torch.tensor([3.0])).sample((1, cnn_dim)))
        # self.softconv.bias = torch.nn.Parameter(torch.zeros(1))
        # nn.init.normal_(self.softconv, 0, 1)
        self.hidden_dim = hidden_dim // 2 if is_bidirectional else hidden_dim
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_dim,self.hidden_dim,layer_num,batch_first=False, bidirectional=is_bidirectional, dropout=0.4)
        self.fc = nn.Linear(hidden_dim,output_dim)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(10)
        
    def forward(self, inputs, times):
        # x = self.bn(inputs)
        feature_1d = self.softconv(inputs).squeeze()
        times_vec1 = self.time2vec(times)

        # print(feature_1d)
        # print(times_vec)
        feature_1d = torch.add(feature_1d, 0.01 * times_vec1)

        lstm_out,(hn,cn) = self.lstm(feature_1d)
        out = self.fc(lstm_out)
        # print(out.shape)
        return self.sigmoid(out), out
    
class Joint_learning(nn.Module):
    def __init__(self, cnn_dim, input_dim, hidden_dim, output_dim, fusion_dim):
        super(Joint_learning, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, 2, batch_first=True, dropout=0.4)
        self.lstm_dense = nn.Linear(hidden_dim, fusion_dim)

        self.cnn1 = nn.Conv1d(input_dim, 64, kernel_size=3, stride=1)
        self.cnn2 = nn.Conv1d(64, 64, kernel_size=3, stride=1)
        self.cnn_dense = nn.Linear(64, fusion_dim)

        self.shared_fc = nn.Linear(fusion_dim, fusion_dim)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(fusion_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        # x = self.bn(inputs)
        # flatten inputs
        print(inputs.shape)
        inputs = inputs.view(inputs.shape[0], -1)
        lstm_out, (hn, cn) = self.lstm(inputs)
        lstm_out = self.lstm_dense(lstm_out)

        # stack itself 3 times
        inputs = torch.stack([inputs, inputs, inputs], dim=-1)
        print(inputs.shape)

        cnn_out = self.cnn1(inputs)
        print(cnn_out.shape)
        cnn_out = self.cnn2(cnn_out)
        cnn_out = self.cnn_dense(cnn_out)

        out = torch.add(lstm_out, cnn_out)
        out = self.relu(self.shared_fc(out))
        out = self.fc(out)

        return self.softmax(out), out
    
class Multi_out_LSTM(nn.Module):

    def __init__(self, cnn_dim, input_dim,hidden_dim,output_dim,layer_num, is_bidirectional):
        super(Multi_out_LSTM, self).__init__()
        self.bn = nn.BatchNorm1d(cnn_dim)
        self.softconvs = []
        for i in range(output_dim):
            softconv = nn.Conv1d(cnn_dim, 1, kernel_size=1)
            softconv.weight = nn.Parameter(Weibull(torch.tensor([1.0]), torch.tensor([3.0])).sample((1, cnn_dim)))
            self.softconvs.append(self.softconvs)
        self.softconv = nn.Conv1d(cnn_dim, 1, kernel_size=1)
        self.softconv.weight = nn.Parameter(Weibull(torch.tensor([1.0]), torch.tensor([3.0])).sample((1, cnn_dim)))
        self.time2vec = SineActivation(1, input_dim)

        self.hidden_dim = hidden_dim // 2 if is_bidirectional else hidden_dim
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_dim,self.hidden_dim,layer_num,batch_first=False, bidirectional=is_bidirectional, dropout=0.4)
        self.out = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features=hidden_dim, out_features=output_dim)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,inputs, times):
        x = self.bn(inputs)
        feature_1d = self.softconv(inputs).squeeze()

        times_vec = self.time2vec(times)
        feature_1d = torch.add(feature_1d, 0.002 * times_vec)

        lstm_out,(hn,cn) = self.lstm(feature_1d)
        logits = self.out(lstm_out)
        # print(out.shape)
        return self.sigmoid(logits), logits

class Multi(nn.Module):

    def __init__(self, models, input_dim,output_dim):
        super(Multi, self).__init__()
        self.models = models
        self.out = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=input_dim, out_features=output_dim)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inputs):
        hidden = torch.Tensor([])
        for model in self.models:
            hidden = torch.concat([hidden, model(inputs)])
        logits = self.out(hidden)
        # print(out.shape)
        return self.sigmoid(logits), logits
    
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_f1_score = -np.inf

    def early_stop(self, f1_score):
        # if validation_loss < self.min_validation_loss:
        #     self.min_validation_loss = validation_loss
        #     self.counter = 0
        # elif validation_loss > (self.min_validation_loss + self.min_delta):
        #     self.counter += 1
        #     if self.counter >= self.patience:
        #         return True
        if f1_score > self.max_f1_score:
            self.max_f1_score = f1_score
            self.counter = 0
        elif f1_score > (self.max_f1_score + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        # self.min_validation_loss = validation_loss
        return False
    
class FocalLoss(nn.Module):
    def __init__(self, nfreqs, gamma=2, alpha=0.25, reduction = "mean"):
        super(FocalLoss, self).__init__()
        self._gamma = gamma
        self._alpha = alpha
        self._red = reduction
        self.nfreqs = nfreqs

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

        Args:
            inputs (Tensor): A float tensor of arbitrary shape.
                    The predictions for each example.
            targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha (float): Weighting factor in range (0,1) to balance
                    positive vs negative examples or -1 for ignore. Default: ``0.25``.
            gamma (float): Exponent of the modulating factor (1 - p_t) to
                    balance easy vs hard examples. Default: ``2``.
            reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                    ``'none'``: No reduction will be applied to the output.
                    ``'mean'``: The output will be averaged.
                    ``'sum'``: The output will be summed. Default: ``'none'``.
        Returns:
            Loss tensor with the reduction option applied.
        """
        # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py

        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=self.nfreqs, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self._gamma)

        if self._alpha >= 0:
            alpha_t = self._alpha * targets + (1 - self._alpha) * (1 - targets)
            loss = alpha_t * loss

        # Check reduction option and return loss accordingly
        if self._red == "none":
            pass
        elif self._red == "mean":
            loss = loss.mean()
        elif self._red == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{self._red} \n Supported reduction modes: 'none', 'mean', 'sum'"
            )
        return loss
