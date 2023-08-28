import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset

from options import INPUT_IMAGE, KERNEL, STRIDE, PADDING, N_CLASSES, N_UNITS, MOMENTUM, \
                    W_BASEMEAN_DATA_PATH, WO_BASEMEAN_DATA_PATH, LABEL_PATH, WEIGHTS_PATH

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

def get_5fold(label_path):
    label = pd.read_csv(label_path)
    trial = label.iloc[:,0].str.split("_trial").str[1].astype('int64')
    test_indexes = [
        ([idx for idx in range(len(trial)) if trial[idx] not in range(1, 9)], [idx for idx in range(len(trial)) if trial[idx] in range(1, 9)]),
        ([idx for idx in range(len(trial)) if trial[idx] not in range(9, 17)], [idx for idx in range(len(trial)) if trial[idx] in range(9, 17)]),
        ([idx for idx in range(len(trial)) if trial[idx] not in range(17, 25)], [idx for idx in range(len(trial)) if trial[idx] in range(17, 25)]),
        ([idx for idx in range(len(trial)) if trial[idx] not in range(25, 33)], [idx for idx in range(len(trial)) if trial[idx] in range(25, 33)]),
        ([idx for idx in range(len(trial)) if trial[idx] not in range(33, 41)], [idx for idx in range(len(trial)) if trial[idx] in range(33, 41)])]
    
    return test_indexes


class DEAP(Dataset):
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

if __name__=="__main__":
    print("[DATA LOADER]")

    dataset = DEAP(W_BASEMEAN_DATA_PATH, LABEL_PATH, target="arousal", n_classes=3)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4) 

    for name, eeg, score, label_cls in dataloader:
        print(f'-----{name}')
        print(">>", eeg.shape)
        print(">>", score)
        print(">>", label_cls)