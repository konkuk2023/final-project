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
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

"""
    Method
    - SD: subject dependent
    - CSV: cross subject - video
    - CSI: cross subject - interval
    - US: unseen subject
"""
def get_5fold(label_path, file_length=10, method="CSI", subject=None):
    num_division = 60 // file_length    # 60 // 6 = 10
    label = pd.read_csv(label_path)
    label = pd.concat([label for _ in range(num_division)]).sort_values(by="Unnamed: 0").reset_index(drop=True)
    trial = label.iloc[:,0].str.split("_trial").str[1].astype('int64')
    sub = label.iloc[:,0].str.split("_trial").str[0].str.replace("s", "").astype('int64')
    
    if method=="SD":
        print(f"Experiment(5-fold) only for {subject} Subject will be conducted!!")
        test_indexes = [
        ([idx for idx, s in zip(range(len(trial)), sub) if trial[idx] not in range(1, 9) and s==subject], [idx for idx, s in zip(range(len(trial)), sub) if trial[idx] in range(1, 9) and s==subject]),
        ([idx for idx, s in zip(range(len(trial)), sub) if trial[idx] not in range(9, 17) and s==subject], [idx for idx, s in zip(range(len(trial)), sub) if trial[idx] in range(9, 17) and s==subject]),
        ([idx for idx, s in zip(range(len(trial)), sub) if trial[idx] not in range(17, 25) and s==subject], [idx for idx, s in zip(range(len(trial)), sub) if trial[idx] in range(17, 25) and s==subject]),
        ([idx for idx, s in zip(range(len(trial)), sub) if trial[idx] not in range(25, 33) and s==subject], [idx for idx, s in zip(range(len(trial)), sub) if trial[idx] in range(25, 33) and s==subject]),
        ([idx for idx, s in zip(range(len(trial)), sub) if trial[idx] not in range(33, 41) and s==subject], [idx for idx, s in zip(range(len(trial)), sub) if trial[idx] in range(33, 41) and s==subject])]
    
    elif method=="CSV":
        print("Cross-Subject(Video unit) Experiment(5-fold) will be conducted!!")
        test_indexes = [
        ([idx for idx in range(len(trial)) if trial[idx] not in range(1, 9)], [idx for idx in range(len(trial)) if trial[idx] in range(1, 9)]),
        ([idx for idx in range(len(trial)) if trial[idx] not in range(9, 17)], [idx for idx in range(len(trial)) if trial[idx] in range(9, 17)]),
        ([idx for idx in range(len(trial)) if trial[idx] not in range(17, 25)], [idx for idx in range(len(trial)) if trial[idx] in range(17, 25)]),
        ([idx for idx in range(len(trial)) if trial[idx] not in range(25, 33)], [idx for idx in range(len(trial)) if trial[idx] in range(25, 33)]),
        ([idx for idx in range(len(trial)) if trial[idx] not in range(33, 41)], [idx for idx in range(len(trial)) if trial[idx] in range(33, 41)])]

    elif method=="CSI":
        print("Cross-Subject(Interval unit) Experiment(5-fold) will be conducted!!")
        test_indexes = [
        ([idx for idx in range(len(trial)) if idx % 5 != 0], [idx for idx in range(len(trial)) if idx % 5 == 0]),
        ([idx for idx in range(len(trial)) if idx % 5 != 1], [idx for idx in range(len(trial)) if idx % 5 == 1]),
        ([idx for idx in range(len(trial)) if idx % 5 != 2], [idx for idx in range(len(trial)) if idx % 5 == 2]),
        ([idx for idx in range(len(trial)) if idx % 5 != 3], [idx for idx in range(len(trial)) if idx % 5 == 3]),
        ([idx for idx in range(len(trial)) if idx % 5 != 4], [idx for idx in range(len(trial)) if idx % 5 == 4])]

    elif method=="US":
        print("Unseen-Subject Experiment(5-fold) will be conducted!!")
        test_indexes = [
        ([idx for idx in range(len(sub)) if sub[idx] not in range(1, 7)], [idx for idx in range(len(sub)) if sub[idx] in range(1, 7)]),
        ([idx for idx in range(len(sub)) if sub[idx] not in range(7, 13)], [idx for idx in range(len(sub)) if sub[idx] in range(7, 13)]),
        ([idx for idx in range(len(sub)) if sub[idx] not in range(13, 19)], [idx for idx in range(len(sub)) if sub[idx] in range(13, 19)]),
        ([idx for idx in range(len(sub)) if sub[idx] not in range(19, 26)], [idx for idx in range(len(sub)) if sub[idx] in range(19, 26)]),
        ([idx for idx in range(len(sub)) if sub[idx] not in range(26, 33)], [idx for idx in range(len(sub)) if sub[idx] in range(26, 33)])]
    
    return test_indexes

def get_10fold(label_path, file_length=10, method="CSI", subject=None):
    num_division = 60 // file_length
    label = pd.read_csv(label_path)
    label = pd.concat([label for _ in range(num_division)]).sort_values(by="Unnamed: 0").reset_index(drop=True)
    trial = label.iloc[:,0].str.split("_trial").str[1].astype('int64')
    sub = label.iloc[:,0].str.split("_trial").str[0].str.replace("s", "").astype('int64')
    
    if method=="SD":
        print(f"Experiment(10-fold) only for {subject} Subject will be conducted!!")
        test_indexes = [
        ([idx for idx, s in zip(range(len(trial)), sub) if trial[idx] not in range(1, 5) and s==subject], [idx for idx, s in zip(range(len(trial)), sub) if trial[idx] in range(1, 5) and s==subject]),
        ([idx for idx, s in zip(range(len(trial)), sub) if trial[idx] not in range(5, 9) and s==subject], [idx for idx, s in zip(range(len(trial)), sub) if trial[idx] in range(5, 9) and s==subject]),
        ([idx for idx, s in zip(range(len(trial)), sub) if trial[idx] not in range(9, 13) and s==subject], [idx for idx, s in zip(range(len(trial)), sub) if trial[idx] in range(9, 13) and s==subject]),
        ([idx for idx, s in zip(range(len(trial)), sub) if trial[idx] not in range(13, 17) and s==subject], [idx for idx, s in zip(range(len(trial)), sub) if trial[idx] in range(13, 17) and s==subject]),
        ([idx for idx, s in zip(range(len(trial)), sub) if trial[idx] not in range(17, 21) and s==subject], [idx for idx, s in zip(range(len(trial)), sub) if trial[idx] in range(17, 21) and s==subject]),
        ([idx for idx, s in zip(range(len(trial)), sub) if trial[idx] not in range(21, 25) and s==subject], [idx for idx, s in zip(range(len(trial)), sub) if trial[idx] in range(21, 25) and s==subject]),
        ([idx for idx, s in zip(range(len(trial)), sub) if trial[idx] not in range(25, 29) and s==subject], [idx for idx, s in zip(range(len(trial)), sub) if trial[idx] in range(25, 29) and s==subject]),
        ([idx for idx, s in zip(range(len(trial)), sub) if trial[idx] not in range(29, 33) and s==subject], [idx for idx, s in zip(range(len(trial)), sub) if trial[idx] in range(29, 33) and s==subject]),
        ([idx for idx, s in zip(range(len(trial)), sub) if trial[idx] not in range(33, 37) and s==subject], [idx for idx, s in zip(range(len(trial)), sub) if trial[idx] in range(33, 37) and s==subject]),
        ([idx for idx, s in zip(range(len(trial)), sub) if trial[idx] not in range(37, 41) and s==subject], [idx for idx, s in zip(range(len(trial)), sub) if trial[idx] in range(37, 41) and s==subject])]

    elif method=="CSV":
        print("Cross-Subject(Video unit) Experiment(5-fold) will be conducted!!")
        test_indexes = [
        ([idx for idx in range(len(trial)) if trial[idx] not in range(1, 5)], [idx for idx in range(len(trial)) if trial[idx] in range(1, 5)]),
        ([idx for idx in range(len(trial)) if trial[idx] not in range(5, 9)], [idx for idx in range(len(trial)) if trial[idx] in range(5, 9)]),
        ([idx for idx in range(len(trial)) if trial[idx] not in range(9, 13)], [idx for idx in range(len(trial)) if trial[idx] in range(9, 13)]),
        ([idx for idx in range(len(trial)) if trial[idx] not in range(13, 17)], [idx for idx in range(len(trial)) if trial[idx] in range(13, 17)]),
        ([idx for idx in range(len(trial)) if trial[idx] not in range(17, 21)], [idx for idx in range(len(trial)) if trial[idx] in range(17, 21)]),
        ([idx for idx in range(len(trial)) if trial[idx] not in range(21, 25)], [idx for idx in range(len(trial)) if trial[idx] in range(21, 25)]),
        ([idx for idx in range(len(trial)) if trial[idx] not in range(25, 29)], [idx for idx in range(len(trial)) if trial[idx] in range(25, 29)]),
        ([idx for idx in range(len(trial)) if trial[idx] not in range(29, 33)], [idx for idx in range(len(trial)) if trial[idx] in range(29, 33)]),
        ([idx for idx in range(len(trial)) if trial[idx] not in range(33, 37)], [idx for idx in range(len(trial)) if trial[idx] in range(33, 37)]),
        ([idx for idx in range(len(trial)) if trial[idx] not in range(37, 41)], [idx for idx in range(len(trial)) if trial[idx] in range(37, 41)])]
    
    elif method=="CSI":
        print("Cross-Subject(Interval unit) Experiment(5-fold) will be conducted!!")
        test_indexes = [
        ([idx for idx in range(len(trial)) if idx % 10 != 0], [idx for idx in range(len(trial)) if idx % 10 == 0]),
        ([idx for idx in range(len(trial)) if idx % 10 != 1], [idx for idx in range(len(trial)) if idx % 10 == 1]),
        ([idx for idx in range(len(trial)) if idx % 10 != 2], [idx for idx in range(len(trial)) if idx % 10 == 2]),
        ([idx for idx in range(len(trial)) if idx % 10 != 3], [idx for idx in range(len(trial)) if idx % 10 == 3]),
        ([idx for idx in range(len(trial)) if idx % 10 != 4], [idx for idx in range(len(trial)) if idx % 10 == 4]),
        ([idx for idx in range(len(trial)) if idx % 10 != 5], [idx for idx in range(len(trial)) if idx % 10 == 5]),
        ([idx for idx in range(len(trial)) if idx % 10 != 6], [idx for idx in range(len(trial)) if idx % 10 == 6]),
        ([idx for idx in range(len(trial)) if idx % 10 != 7], [idx for idx in range(len(trial)) if idx % 10 == 7]),
        ([idx for idx in range(len(trial)) if idx % 10 != 8], [idx for idx in range(len(trial)) if idx % 10 == 8]),
        ([idx for idx in range(len(trial)) if idx % 10 != 9], [idx for idx in range(len(trial)) if idx % 10 == 9])]
    
    elif method=="US":
        print("Unseen-Subject Experiment(10-fold) will be conducted!!")
        test_indexes = [
        ([idx for idx in range(len(sub)) if sub[idx] not in range(1, 4)], [idx for idx in range(len(sub)) if sub[idx] in range(1, 4)]),
        ([idx for idx in range(len(sub)) if sub[idx] not in range(4, 7)], [idx for idx in range(len(sub)) if sub[idx] in range(4, 7)]),
        ([idx for idx in range(len(sub)) if sub[idx] not in range(7, 10)], [idx for idx in range(len(sub)) if sub[idx] in range(7, 10)]),
        ([idx for idx in range(len(sub)) if sub[idx] not in range(10, 13)], [idx for idx in range(len(sub)) if sub[idx] in range(10, 13)]),
        ([idx for idx in range(len(sub)) if sub[idx] not in range(13, 16)], [idx for idx in range(len(sub)) if sub[idx] in range(13, 16)]),
        ([idx for idx in range(len(sub)) if sub[idx] not in range(16, 19)], [idx for idx in range(len(sub)) if sub[idx] in range(16, 19)]),
        ([idx for idx in range(len(sub)) if sub[idx] not in range(19, 22)], [idx for idx in range(len(sub)) if sub[idx] in range(19, 22)]),
        ([idx for idx in range(len(sub)) if sub[idx] not in range(22, 25)], [idx for idx in range(len(sub)) if sub[idx] in range(22, 25)]),
        ([idx for idx in range(len(sub)) if sub[idx] not in range(25, 29)], [idx for idx in range(len(sub)) if sub[idx] in range(25, 29)]),
        ([idx for idx in range(len(sub)) if sub[idx] not in range(29, 33)], [idx for idx in range(len(sub)) if sub[idx] in range(29, 33)])]

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
        self.denominator = 12 // n_classes  # 9->1 / 5->2 / 3->4

        if feature_type=="EEG":
            self.sample_rate = 128
        elif feature_type=="DE":
            self.sample_rate = 1

        self.file_length = file_length
        self.num_division = 60 // file_length

    def __len__(self):
        return len(self.label)*self.num_division

    def __getitem__(self, idx):

        trial_idx = idx // self.num_division    # Trial index
        data_idx = (idx % self.num_division) * self.file_length * self.sample_rate # Time index to select interval

        name = self.label.iloc[trial_idx, 0]
        data_path = os.path.join(self.data_dir, name)+'_win_128.pt'
        eeg_signal = torch.load(data_path)[data_idx:data_idx+self.file_length*self.sample_rate]

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
    dataset = DEAP(W_BASEMEAN_DATA_PATH, LABEL_PATH, target="arousal", n_classes=9, feature_type="EEG", file_length=1)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1) 

    # # for name, eeg, score, label_cls in dataloader:
    
    # #     print(f'-----{name}')
    # #     print(">>", eeg.shape)
    # #     print(">>", score)
    # #     print(">>", label_cls)
    
    # print(len(dataloader))
    # print(dataloader.__len__)

    samples = get_10fold(label_path=LABEL_PATH, file_length=60, method="CSI", subject=None)
    for train_idx, test_idx in tqdm(samples):
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)

        print("---------------------")
        print(train_idx)
        print(test_idx)

        trainloader = torch.utils.data.DataLoader(dataset, num_workers=8, batch_size=1, sampler=train_subsampler) 
        testloader = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=1, sampler=test_subsampler) 

        print(len(trainloader))
        print(len(testloader))