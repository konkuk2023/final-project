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
        print("Cross-Subject(Video unit) Experiment(10-fold) will be conducted!!")
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
        print("Cross-Subject(Interval unit) Experiment(10-fold) will be conducted!!")
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

    """
        [1s length]
            - #. of Datas: 1280 x 60 = 76800
        
        [6s length]
            - #. of Datas: 1280 x 10 = 12800
    
    """
    FOLDS = 10
    METHOD = "CSI"
    LENGTH = 6

    SUBJECT = None
    FEATURE = "DE"
    TARGET = "valence"

    NUM_DATA = int(1280 * (60/LENGTH))

    print(f'- FILE LENGTH: {LENGTH}s')
    print(f'- #. of DATAs: {NUM_DATA}\n')

    full_set = set(range(NUM_DATA)) # Full Set
    empty_set = set()               # Empty Set

    dataset = DEAP(W_BASEMEAN_DATA_PATH, LABEL_PATH, target=TARGET, n_classes=9, feature_type=FEATURE, file_length=LENGTH)
    if FOLDS == 5:
        samples = get_5fold(label_path=LABEL_PATH, file_length=LENGTH, method=METHOD, subject=SUBJECT)
    else:
        samples = get_10fold(label_path=LABEL_PATH, file_length=LENGTH, method=METHOD, subject=SUBJECT)

    for fold, (train_idx, test_idx) in tqdm(enumerate(samples)):
        print(f'\n>> Fold {fold+1}')

        trainset = set(train_idx)
        testset = set(test_idx)

        train_subjects = []
        test_subjects = []

        train_videos = []
        test_videos = []

        train_splits = []
        test_splits = []

        # Test 1. Intersection Test
        if (trainset & testset) == empty_set:
            print("\t++ TEST 1) Success: The intersection is an Empty Set")
        else:
            print("\t++ TEST 1) Failed: The intersection is not an Empty Set")
        
        # Test 2. Union Test
        if (trainset | testset) == full_set:
            print("\t++ TEST 2) Success: The union is the Full Set")
        else:
            print("\t++ TEST 2) Failed: The union is not the Full Set")
        
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)
        trainloader = torch.utils.data.DataLoader(dataset, num_workers=8, batch_size=1, sampler=train_subsampler) 
        testloader = torch.utils.data.DataLoader(dataset, num_workers=8, batch_size=1, sampler=test_subsampler) 

        # Test 3 & 4
        for name, _, _, _ in tqdm(trainloader, desc="@TrainSet Checking", leave=False):
            subject, video = name[0].split("_")
            
            train_subjects.append(subject)
            train_subjects = list(set(train_subjects)) # Eliminate Duplication

            train_videos.append(video)
            train_videos = list(set(train_videos)) # Eliminate Duplication

        for name, _, _, _ in tqdm(testloader, desc="@TestSet Checking", leave=False):
            subject, video = name[0].split("_")
            
            test_subjects.append(subject)
            test_subjects = list(set(test_subjects)) # Eliminate Duplication

            test_videos.append(video)
            test_videos = list(set(test_videos)) # Eliminate Duplication

        # Test 3. # of Subjects
        print(f'\n\t++ TEST 3-1) #. of Subjects in the Train Set: {len(train_subjects)}')
        print(f'\t++ TEST 3-2) #. of Subjects in the Test Set: {len(test_subjects)}')
        print(f'\t++ TEST 3-3) Intersection: {len(set(train_subjects) & set(test_subjects))}')
        print(f'\t++ TEST 3-4) Union: {len(set(train_subjects) | set(test_subjects))}')

        # Test 4. # of Videos
        print(f'\n\t++ TEST 4-1) #. of Videos in the Train Set: {len(train_videos)}')
        print(f'\t++ TEST 4-2) #. of Videos in the Test Set: {len(test_videos)}')
        print(f'\t++ TEST 4-3) Intersection: {len(set(train_videos) & set(test_videos))}')
        print(f'\t++ TEST 4-4) Union: {len(set(train_videos) | set(test_videos))}')

        # Test 5. # of Splits
        for idx in tqdm(train_idx, desc="@TrainSet Checking", leave=False):
            internal_subject = idx % int(60/LENGTH)

            train_splits.append(internal_subject)
            train_splits = list(set(train_splits))

        for idx in tqdm(test_idx, desc="@TestSet Checking", leave=False):
            internal_subject = idx % int(60/LENGTH)
            
            test_splits.append(internal_subject)
            test_splits = list(set(test_splits))

        print(f'\n\t++ TEST 5-1) #. of Splits in the Train Set: {len(train_splits)}')
        print(f'\t++ TEST 5-2) #. of Splits in the Test Set: {len(test_splits)}')
        print(f'\t++ TEST 5-3) Intersection: {len(set(train_splits) & set(test_splits))}')
        print(f'\t++ TEST 5-4) Union: {len(set(train_splits) | set(test_splits))}')
