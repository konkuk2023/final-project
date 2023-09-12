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

class Calculator():
    def __init__(self, cal_type, penalty=1):
        self.cal_type = cal_type
        self.penalty = penalty
    def calculate(self, probs):
        if self.cal_type == "Expectation":
            return self.expectation(probs)
        elif self.cal_type == "Bernoulli":
            return self.bernoulli(probs)
    def expectation(self, probs):
        return torch.sum(torch.tensor(range(1,10)).to(probs.device)*probs, 1)
    def bernoulli(self, probs):
        return torch.sum(torch.tensor(range(1,10)).to(probs.device)*probs - self.penalty*(1-probs)*probs, 1)

class Custom_Loss(nn.Module):
    def __init__(self, device='cpu', alpha=1, dist_type="None", n_classes=9):
        super(Custom_Loss, self).__init__()
        self.device = device
        self.dist_type = dist_type
        self.n_classes = n_classes
        self.alpha = alpha
        # self.ce = nn.CrossEntropyLoss()
        # self.mse = nn.MSELoss()
    def forward(self, pred_probs, true_class, pred_points, true_points):
        """
            pred_probs: [B x n_classes]
            true_classes: [B x 1]
            pred_points: [B x 1]
            pred_points: [B x 1]
        """
        MSE = nn.MSELoss()
        CE = nn.CrossEntropyLoss()

        mse = MSE(pred_points, true_points).to(self.device)
        ce = torch.tensor(0, dtype=torch.float64).to(self.device)

        # No Distance
        if self.dist_type == "None":
            weight = torch.tensor([1]*pred_probs.shape[0]).to(self.device)
            for index, pred in enumerate(pred_probs):
                ce += weight[index] * CE(pred.unsqueeze(0), true_class[index].unsqueeze(0))
        # 1(:Offset) + Abs. Distance
        elif self.dist_type == "Absolute":
            weight = torch.abs(torch.subtract(torch.argmax(pred_probs, 1), true_class)) + 1
            for index, pred in enumerate(pred_probs):
                ce += weight[index] * CE(pred.unsqueeze(0), true_class[index].unsqueeze(0))
        # 1(:Offset) + Sqr. Distance
        elif self.dist_type == "Square":
            weight = torch.subtract(torch.argmax(pred_probs, 1), true_class) ** 2 + 1
            for index, pred in enumerate(pred_probs):
                ce += weight[index] * CE(pred.unsqueeze(0), true_class[index].unsqueeze(0))
        
        ce = ce / pred_probs.shape[0]

        loss = mse + self.alpha*ce

        return loss, mse, ce
        
class Custom_Loss2(nn.Module):
    def __init__(self, device='cpu', alpha=1, dist_type="None", n_classes=9):
        super(Custom_Loss2, self).__init__()
        self.device = device
        self.dist_type = dist_type
        self.n_classes = n_classes
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
    def forward(self, pred_probs, true_class, pred_points, true_points):
        """
            pred_probs: [B x n_classes]
            true_classes: [B x 1]
            pred_points: [B x 1]
            pred_points: [B x 1]
        """
        mse = self.mse(pred_points, true_points).to(self.device)
        ce = torch.tensor(0, dtype=torch.float64).to(self.device)
        # ce = -1 * (torch.log(1 - pred_probs)+1e-6) * (1-nn.functional.one_hot(true_class, num_classes=self.n_classes))
        # ce = torch.nan_to_num(ce, posinf=1e10, neginf=-1e10, nan= 0)

        true_class = true_class.long()
        true_points = true_points.long()

        # No Distance
        if self.dist_type == "None":
            weight = torch.tensor([1]*pred_probs.shape[0]).to(self.device)
            for index, pred in enumerate(pred_probs):
                ce += weight[index] * self.ce(pred.unsqueeze(0), true_class[index].unsqueeze(0))
        # 1(:Offset) + Abs. Distance
        elif self.dist_type == "Absolute":
            weight = torch.abs(torch.subtract(torch.argmax(pred_probs, 1), true_class)) + 1
            for index, pred in enumerate(pred_probs):
                ce += weight[index] * self.ce(pred.unsqueeze(0), true_class[index].unsqueeze(0))
        # 1(:Offset) + Sqr. Distance
        elif self.dist_type == "Square":
            weight = torch.subtract(torch.argmax(pred_probs, 1), true_class) ** 2 + 1
            for index, pred in enumerate(pred_probs):
                ce += weight[index] * self.ce(pred.unsqueeze(0), true_class[index].unsqueeze(0))
        
        ce = ce / pred_probs.shape[0]

        loss = mse + self.alpha*ce

        return loss.float(), mse, ce
        
if __name__=="__main__":
    print("[LOSS FUNCTIONS]")

    loss = Custom_Loss(dist_type="None", n_classes=9)

    cal = Calculator(cal_type="Expectation")

    pred_probs = torch.tensor([[0.01, 0.01, 0.02, 0.02, 0.01, 0.9, 0.01, 0.01, 0.01],
                         [0.01, 0.01, 0.02, 0.02, 0.01, 0.01, 0.9, 0.01, 0.01],
                         [0.01, 0.01, 0.02, 0.02, 0.01, 0.01, 0.01, 0.9, 0.01],
                         [0.01, 0.01, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.9]])
    true_class = torch.tensor([5, 6, 7, 8])
    
    pred_points = cal.calculate(pred_probs)
    true_points = torch.tensor([6, 7, 8, 9])
    
    
    result = loss(pred_probs, true_class, pred_points, true_points)

    print(result)