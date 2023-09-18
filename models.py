import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel=(5, 5), stride=1, padding=1):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel, stride=stride, padding=padding, groups=nin)
        nn.init.kaiming_normal_(self.depthwise.weight)  # Parameter Initialization
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)
        nn.init.kaiming_normal_(self.pointwise.weight)  # Parameter Initialization
    def forward(self, x):
        out = self.depthwise(x)
        del x
        out = self.pointwise(out)
        return out

class EmoticonNet(nn.Module):
    def __init__(self, device='cpu', input_image=torch.zeros(1, 7680, 1, 9, 9), kernel=(5, 5), stride=1, padding=1, n_classes=9, n_units=128, formula="Expectation"):
        """
        EmoticonNet
            - Input: Electroencephalogram Signals 
                    [Batch Size, Sequence, Image Size]
            - Return: Probabilities for each class 
                    [Batch Size, n_classes]
        """
        super(EmoticonNet, self).__init__()

        self.device = device
        self.formula = formula
        n_channel = input_image.shape[2]
        
        # EmoticonNet 2D ConvNet
        self.ENet2D = nn.Sequential(
            depthwise_separable_conv(n_channel, 32, kernel, stride=stride, padding=padding),
            nn.BatchNorm2d(32),
            nn.ELU(),
            depthwise_separable_conv(32, 64, kernel, stride=stride, padding=padding),
            nn.BatchNorm2d(64),
            nn.ELU(),
            depthwise_separable_conv(64, 128, kernel, stride=stride, padding=padding),
            nn.BatchNorm2d(128),
            nn.ELU()
        ).to(self.device)

        # EmoticonNet Temporal 1D Convolution
        self.ENet1D = nn.Conv1d(3*3*128, 128, input_image.shape[1], stride=stride, padding=0).to(self.device)
        # nn.init.kaiming_normal_(self.ENet1D.weight)  # Parameter Initialization

        # FC layers
        self.fc = nn.Linear(128, 128).to(self.device)
        self.fc1 = nn.Linear(128, 128).to(self.device)
        self.fc2 = nn.Linear(128, n_classes).to(self.device)
        self.max = nn.Softmax().to(self.device)
        self.sigmoid = nn.Sigmoid().to(self.device)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.fill_(0)

    
    def forward(self, x):
        tmp = x.reshape(x.shape[0]*x.shape[1], x.shape[2], x.shape[3], x.shape[4]).to(self.device)
        tmp = self.ENet2D(tmp).to(self.device)
        tmp = tmp.reshape(x.shape[0], x.shape[1], 128, 3, 3)
        temp_conv = F.elu(self.ENet1D(tmp.reshape(x.shape[0], 3*3*128, x.shape[1])))
        temp_conv = temp_conv.reshape(temp_conv.shape[0], -1)
        del tmp, x

        out = temp_conv/torch.max(temp_conv)
        out1 = F.sigmoid(self.fc(out))
        out = torch.mul(out, out1)
        out = F.elu(self.fc1(out))
        out = self.fc2(out)
        if self.formula in ["Expextation", "Bernoulli"]:
            out = self.max(out)
        else:
            out = self.sigmoid(out)

        del temp_conv, out1

        return out

if __name__=="__main__":
    print("[MODELS]")

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)*8 / 1e+9
    
    model = EmoticonNet()
    print(count_parameters(model), "GB")