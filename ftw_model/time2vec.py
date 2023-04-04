import torch
from torch import nn
import numpy as np
import math

def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        #print(w.shape, t1.shape, b.shape)
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    #print(v1.shape)
    return torch.cat([v1, v2], -1)

class SineActivation(nn.Module):
    def __init__(self, in_features, out_features, pattern=1440):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        w = torch.Tensor(in_features, out_features-1)
        w0 = torch.Tensor(in_features, 1)
        torch.Tensor.fill_(w, torch.pi * 2 / pattern)
        torch.Tensor.fill_(w0, torch.pi * 2 / pattern)

        self.w0 = nn.parameter.Parameter(w0)
        self.b0 = nn.parameter.Parameter(torch.zeros(1))
        self.w = nn.parameter.Parameter(w)
        self.b = nn.parameter.Parameter(torch.zeros(out_features-1))
        self.f = torch.sin

    def forward(self, tau):
        # tau.shape = (1, in_features)
        # w.shape = (in_features, out_features-1)
        # w0.shape = (in_features, 1)
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)
    
class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(out_features-1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)
    
class TimeEncoding(nn.Module):
    def __init__(self, in_features, out_features, pattern=1440):
        super(TimeEncoding, self).__init__()
        self.out_features = out_features
        self.pattern = pattern

    def forward(self, tau):
        p = [torch.sin(tau / self.pattern ** (2 * i / self.out_features)) if i % 2 == 0 else torch.cos(tau / self.pattern ** (2 * i / self.out_features)) for i in range(self.out_features)]
        p = torch.concat(p, dim=-1)
        return p