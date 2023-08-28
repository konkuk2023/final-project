import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset

from options import INPUT_IMAGE, KERNEL, STRIDE, PADDING, N_CLASSES, N_UNITS, MOMENTUM, \
                    W_BASEMEAN_DATA_PATH, WO_BASEMEAN_DATA_PATH, DE_W_BASEMEAN_DATA_PATH, DE_WO_BASEMEAN_DATA_PATH, \
                    LABEL_PATH, WEIGHTS_PATH

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

def get_5fold(label_path, file_length=10):
    num_division = 60 // file_length
    label = pd.read_csv(label_path)
    label = pd.concat([label for _ in range(num_division)]).sort_values(by="Unnamed: 0").reset_index(drop=True)
    trial = label.iloc[:,0].str.split("_trial").str[1].astype('int64')
    test_indexes = [
        ([idx for idx in range(len(trial)) if trial[idx] not in range(1, 9)], [idx for idx in range(len(trial)) if trial[idx] in range(1, 9)]),
        ([idx for idx in range(len(trial)) if trial[idx] not in range(9, 17)], [idx for idx in range(len(trial)) if trial[idx] in range(9, 17)]),
        ([idx for idx in range(len(trial)) if trial[idx] not in range(17, 25)], [idx for idx in range(len(trial)) if trial[idx] in range(17, 25)]),
        ([idx for idx in range(len(trial)) if trial[idx] not in range(25, 33)], [idx for idx in range(len(trial)) if trial[idx] in range(25, 33)]),
        ([idx for idx in range(len(trial)) if trial[idx] not in range(33, 41)], [idx for idx in range(len(trial)) if trial[idx] in range(33, 41)])]
    
    return test_indexes

class DEAP_Full(Dataset):
    def __init__(self, data_dir, label_path, target="valence", n_classes=9):
        self.data_dir = data_dir
        self.label = pd.read_csv(label_path)
        self.target = target
        self.denominator = 9 / n_classes

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        name = self.label.iloc[idx, 0]
        data_path = os.path.join(self.data_dir, name)+'_win_128.pt'
        eeg_signal = torch.load(data_path)

        # if self.target == "valence":
        #     valence = self.label.iloc[idx, 1]
        #     return name, eeg_signal.unsqueeze(1).type("torch.FloatTensor"), valence
        # elif self.target == "arousal":
        #     arousal = self.label.iloc[idx, 2]
        #     return name, eeg_signal.unsqueeze(1).type("torch.FloatTensor"), arousal

        score = self.label.loc[idx, self.target]

        label_cls = np.around(score/self.denominator) - 1
        

        return name, eeg_signal.unsqueeze(1).type("torch.FloatTensor"), score, label_cls.astype(np.int32)

class DEAP(Dataset):
    def __init__(self, data_dir, label_path, target="valence", n_classes=9, file_length=10, feature_type="EEG"):
        self.data_dir = data_dir
        self.label = pd.read_csv(label_path)
        self.target = target
        self.feature_type = feature_type
        self.denominator = 12 // n_classes  # 9->1 / 5->2 / 3->4

        self.file_length = file_length
        self.num_division = 60 // file_length

    def __len__(self):
        return len(self.label)*self.num_division

    def __getitem__(self, idx):

        if self.feature_type=="EEG":
            sample_rate = 128
        elif self.feature_type=="DE":
            sample_rate = 1

        trial_idx = idx // self.num_division
        data_idx = (idx % self.num_division) * self.file_length * sample_rate

        name = self.label.iloc[trial_idx, 0]
        data_path = os.path.join(self.data_dir, name)+'_win_128.pt'
        eeg_signal = torch.load(data_path)[data_idx:data_idx+self.file_length*sample_rate]

        score = self.label.loc[trial_idx, self.target]

        if self.denominator == 1:
            label_cls = np.around(score/self.denominator) - 1
        elif self.denominator == 2:
            label_cls = np.floor(score/self.denominator)
        elif self.denominator == 4:
            label_cls = np.floor((score+1)/self.denominator)
        
        return name, eeg_signal.unsqueeze(1).type("torch.FloatTensor"), score, label_cls.astype(np.int32)




if __name__=="__main__":
    print("[DATA LOADER]")

    # dataset = DEAP(W_BASEMEAN_DATA_PATH, LABEL_PATH, target="arousal", n_classes=3, file_length=10)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1) 
    dataset = DEAP(DE_W_BASEMEAN_DATA_PATH, LABEL_PATH, target="arousal", n_classes=3, feature_type="DE", file_length=10)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1) 

    for name, eeg, score, label_cls in dataloader:
        print(f'-----{name}')
        print(">>", eeg.shape)
        print(">>", score)
        print(">>", label_cls)
    
    print(len(dataloader))
    print(dataloader.__len__)