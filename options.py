import torch

INPUT_IMAGE=torch.zeros(1, 7680, 1, 9, 9)
KERNEL=(5, 5)
STRIDE=1
PADDING=1
N_CLASSES=9
N_UNITS=128

MOMENTUM = (0.5, 0.9)

W_BASEMEAN_DATA_PATH = "/workspace/G_Project/DEAP_Dataset/Feature_Extraction/Base_EEG/basemean_yes_pt"
WO_BASEMEAN_DATA_PATH = "/workspace/G_Project/DEAP_Dataset/Feature_Extraction/Base_EEG/basemean_no_pt"
DE_W_BASEMEAN_DATA_PATH = "/workspace/G_Project/DEAP_Dataset/Feature_Extraction/DE_feature/basemean_yes_pt"
DE_WO_BASEMEAN_DATA_PATH = "/workspace/G_Project/DEAP_Dataset/Feature_Extraction/DE_feature/basemean_no_pt"
LABEL_PATH = "/workspace/G_Project/DEAP_Dataset/EEG_label/EEG_label.csv"
WEIGHTS_PATH = "/workspace/G_Project/weights"
SCOPE1 = "https://spreadsheets.google.com/feeds"
SCOPE2 = "https://www.googleapis.com/auth/drive"