import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import Dataset

from options import INPUT_IMAGE, KERNEL, STRIDE, PADDING, N_CLASSES, N_UNITS, MOMENTUM, \
                    W_BASEMEAN_DATA_PATH, WO_BASEMEAN_DATA_PATH, DE_W_BASEMEAN_DATA_PATH, DE_WO_BASEMEAN_DATA_PATH, \
                    LABEL_PATH, WEIGHTS_PATH, SCOPE1, SCOPE2
from key_options import KEY_FILE, SHEET_URL # Not opened in GitHub
from models import EmoticonNet
from dataloader import get_5fold, get_10fold, DEAP
from loss import Custom_Loss, Custom_Loss2

import os
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import warnings
warnings.filterwarnings('ignore')

class Calculator():
    """
        MMSE Score Calculator
            Expectation Formula is the base formula for calculating MMSE score.
            You can choose whether to apply Bernoulli Penalty on the calculation.
            - Expectation Formula
            - Bernoulli Formula: Apply Bernoulli Penalty on Expectation Formula
    """
    def __init__(self, cal_type, penalty=1, n_classes=9):
        self.cal_type = cal_type
        self.penalty = penalty
        self.interval_length = 12 // n_classes
    def calculate(self, probs):
        if self.cal_type == "Expectation":
            return self.expectation(probs)
        elif self.cal_type == "Bernoulli":
            return self.bernoulli(probs)
    def expectation(self, probs):
        return torch.sum(torch.tensor(range(1,10,self.interval_length)).to(probs.device)*probs, 1)
    def bernoulli(self, probs):
        return torch.sum(torch.tensor(range(1,10,self.interval_length)).to(probs.device)*probs - self.penalty*(1-probs)*probs, 1)
    
def optimizer_to(optim, device):
    """
        Reference: https://github.com/pytorch/pytorch/issues/8741
        You can put the optimizer on GPU or CPU.
    """
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def str2bool(v):
    """
        Reference: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
        Argparse gets all of the input as a string type or a numeruc type.
        So you have to change the string type to boolean type if you need.
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def create_directory(directory):
    """
        Create a directory to save the weights and logs.
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

def save_google_spread(config, results, accuracies, epochs):
    """
        Save the RMSE results of 5-fold cross validation on the google spread sheet.
    """
    sec2sheet = {
        60: "Full Length",
        6: "6s Length",
        1: "1s Length"
    }

    base_offset = 2
    method_offset = {
        "CSV": 0,
        "CSI": 36
    }
    intervals_offset = {
        9: 0,
        5: 12,
        3: 24
    }
    target_offset = {
        "valence": 0,
        "arousal": 6
    }
    formula_offset = {
        "Expectation": 0,
        "Bernoulli": 3
    }
    weight_offset = {
        "None": 0,
        "Absolute": 1, 
        "Square": 2
    }

    scope = [SCOPE1, SCOPE2]
    credentials = ServiceAccountCredentials.from_json_keyfile_name(KEY_FILE, scope)
    gc = gspread.authorize(credentials)
    doc = gc.open_by_url(SHEET_URL)
    worksheet = doc.worksheet(sec2sheet[config["file_length"]])

    offset = base_offset + method_offset[config["method"]] + intervals_offset[config["n_classes"]] + target_offset[config["target"]] \
            + formula_offset[config["formula"]] + weight_offset[config["weight"]]
    offset = str(offset)

    mean_rmse = np.mean(results)
    worksheet.update_acell("G"+offset, np.round(mean, 4))
    worksheet.update_acell("I"+offset, np.round(results[0], 4))
    worksheet.update_acell("K"+offset, np.round(results[1], 4))
    worksheet.update_acell("M"+offset, np.round(results[2], 4))
    worksheet.update_acell("O"+offset, np.round(results[3], 4))
    worksheet.update_acell("Q"+offset, np.round(results[4], 4))

    mean_acc = np.mean(accuracies)
    worksheet.update_acell("H"+offset, np.round(mean, 4))
    worksheet.update_acell("J"+offset, np.round(accuracies[0], 4))
    worksheet.update_acell("L"+offset, np.round(accuracies[1], 4))
    worksheet.update_acell("N"+offset, np.round(accuracies[2], 4))
    worksheet.update_acell("P"+offset, np.round(accuracies[3], 4))
    worksheet.update_acell("R"+offset, np.round(accuracies[4], 4))

    worksheet.update_acell("T"+offset, epochs[0])
    worksheet.update_acell("U"+offset, epochs[1])
    worksheet.update_acell("V"+offset, epochs[2])
    worksheet.update_acell("W"+offset, epochs[3])
    worksheet.update_acell("X"+offset, epochs[4])

def save_google_spread_test_params(config, results, epochs, offset):
    """
        Save the RMSE results of 5-fold cross validation on the google spread sheet for testing hyperparameters.
    """
    scope = [SCOPE1, SCOPE2]
    credentials = ServiceAccountCredentials.from_json_keyfile_name(KEY_FILE, scope)
    gc = gspread.authorize(credentials)
    doc = gc.open_by_url(SHEET_URL)
    worksheet = doc.worksheet("Settings")

    offset = str(offset)

    mean = np.mean(results)
    worksheet.update_acell("G"+offset, np.round(mean, 4))
    worksheet.update_acell("H"+offset, np.round(results[0], 4))
    worksheet.update_acell("I"+offset, np.round(results[1], 4))
    worksheet.update_acell("J"+offset, np.round(results[2], 4))
    worksheet.update_acell("K"+offset, np.round(results[3], 4))
    worksheet.update_acell("L"+offset, np.round(results[4], 4))

    worksheet.update_acell("N"+offset, epochs[0])
    worksheet.update_acell("O"+offset, epochs[1])
    worksheet.update_acell("P"+offset, epochs[2])
    worksheet.update_acell("Q"+offset, epochs[3])
    worksheet.update_acell("R"+offset, epochs[4])

def train_cv(config):
    """
        Function to train a model which uses Expectation Formula or Bernoulli Formula by 5-fold cross validation.
    """
    # Basic Informations
    target = config["target"]
    formula = config["formula"]
    weight = config["weight"]
    alpha = config["alpha"]
    basemean = config["basemean"]

    torch.cuda.empty_cache()

    # Results of each fold
    best_epochs = []
    results = []
    accuracies = []
    
    if config["folds"]==5:
        kfold = get_5fold(LABEL_PATH, file_length=config["file_length"], method=config["method"], subject=config["subject"])
    elif config["folds"]==10:
        kfold = get_10fold(LABEL_PATH, file_length=config["file_length"], method=config["method"], subject=config["subject"])
    cal = Calculator(cal_type=config["formula"], n_classes=config["n_classes"])

    if config["feature_type"]=="EEG":
        input_length = int(7680 / (60//config["file_length"]))
        input_image = torch.zeros(1, input_length, 1, 9, 9)
        if config["basemean"]:
            dataset = DEAP(data_dir=W_BASEMEAN_DATA_PATH, label_path=LABEL_PATH, target=config["target"], file_length=config["file_length"], feature_type="EEG")
        else:
            dataset = DEAP(data_dir=WO_BASEMEAN_DATA_PATH, label_path=LABEL_PATH, target=config["target"], file_length=config["file_length"], feature_type="EEG")
    elif config["feature_type"]=="DE":
        input_length = int(60 / (60//config["file_length"]))
        input_image = torch.zeros(1, input_length, 1, 9, 9)
        if config["basemean"]:
            dataset = DEAP(data_dir=DE_W_BASEMEAN_DATA_PATH, label_path=LABEL_PATH, target=config["target"], file_length=config["file_length"], feature_type="DE")
        else:
            dataset = DEAP(data_dir=DE_WO_BASEMEAN_DATA_PATH, label_path=LABEL_PATH, target=config["target"], file_length=config["file_length"], feature_type="DE")

    # Devide folds
    for fold, (train_idx, test_idx) in enumerate(kfold):
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)

        trainloader = torch.utils.data.DataLoader(dataset, num_workers=2, batch_size=config["batch"], sampler=train_subsampler) 
        testloader = torch.utils.data.DataLoader(dataset, num_workers=2, batch_size=1, sampler=test_subsampler) 

        if config["test_params"]:
            save_path = os.path.join(WEIGHTS_PATH, config["target"], config["method"], 'TEST2', str(config["file_length"])+'s', config["output_dir"], f'fold_{fold+1}')
        else:
            save_path = os.path.join(WEIGHTS_PATH, config["target"], config["method"], config["feature_type"], str(config["file_length"])+'s', config["output_dir"], f'ALPHA_{alpha}', f'fold_{fold+1}')
        create_directory(save_path)

        f = open(os.path.join(save_path, 'output.csv'), 'w')
        f.write('epoch,loss,MSE,CE,RMSE,v_loss,v_MSE,v_CE,v_RMSE,ACC\n')

        print(f'\nTraining for fold {fold+1}')

        LOSS = Custom_Loss(device=config["device"], alpha=config["alpha"], dist_type=config["weight"])

        model = EmoticonNet(device=config["device"], n_classes=config["n_classes"], input_image=input_image, formula=config["formula"])
        if config["optimizer"] == "SGD":
            optimizer = SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9)
        elif config["optimizer"] == "Adam":
            optimizer = Adam(model.parameters(), lr=config["learning_rate"], betas=config["betas"])
        elif config["optimizer"] == "AdamW":
            optimizer = AdamW(model.parameters(), lr=config["learning_rate"], betas=config["betas"], weight_decay=config["weight_decay"])

        for epoch in range(config["epochs"]):
            print(f"[INFO] Target:{target}||Formula:{formula}||Loss:{weight}||Alpha:{alpha}||BaseMean:{basemean}||Fold:{fold+1}")
            torch.cuda.empty_cache()

            model.train()

            optimizer.zero_grad()
            train_mse = 0
            train_ce = 0
            train_loss = 0
            num_data = 0
            optimizer_to(optimizer, config['device'])
            for _, eeg, score, label_cls in tqdm(trainloader, desc=f"Epoch {epoch+1} / TRAIN", ncols=100, ascii=" =", leave=False):
                # Batch size
                num_batch = eeg.shape[0]
                num_data += num_batch

                true_points = torch.round(score).type("torch.FloatTensor").to(config["device"])
                pred_probs = model(eeg.to(config["device"]))
                pred_points = cal.calculate(pred_probs)
                true_class = label_cls.type("torch.LongTensor").to(config["device"])

                del score, eeg, label_cls
                torch.cuda.empty_cache()

                # Calculate Loss Function
                loss, mse, ce = LOSS(pred_probs, true_class, pred_points, true_points)

                del pred_probs, true_class, pred_points, true_points
                torch.cuda.empty_cache()

                train_mse += mse.item()*num_batch
                train_ce += ce.item()*num_batch
                train_loss += loss.item()*num_batch

                # Weights Update
                loss.backward(retain_graph = False)
                optimizer.step()
                optimizer.zero_grad()

                del loss, mse, ce, num_batch
                torch.cuda.empty_cache()

            optimizer_to(optimizer, 'cpu')
            
            train_mse /= num_data
            train_ce /= num_data
            train_loss /= num_data
            train_rmse = train_mse ** (1/2)

            print(f'[{epoch+1}/{config["epochs"]}] LOSS: {train_loss}, MSE: {train_mse}, CE: {train_ce}, RMSE: {train_rmse}')

            model.eval()
            with torch.no_grad():
                
                num_test_data = len(testloader)

                val_mse = 0
                val_ce = 0
                val_loss = 0
                num_data = 0

                acc = 0

                for _, eeg, score, label_cls in tqdm(testloader, desc=f"Epoch {epoch+1} / VALIDATE", ncols=100, ascii=" =", leave=False):

                    true_points = torch.round(score).type("torch.FloatTensor").to(config["device"])
                    pred_probs = model(eeg.to(config["device"]))
                    pred_points = cal.calculate(pred_probs)
                    true_class = label_cls.type("torch.LongTensor").to(config["device"])

                    pred_cls = 1 if pred_points >= 5 else 0
                    true_cls = 1 if true_points >= 5 else 0
                    if pred_cls==true_cls:
                        acc+=1

                    del score, eeg, label_cls
                    torch.cuda.empty_cache()

                    # Calculate Loss Function
                    loss, mse, ce = LOSS(pred_probs, true_class, pred_points, true_points)

                    del pred_probs, true_class, pred_points, true_points
                    torch.cuda.empty_cache()
                    
                    val_mse += mse.item()
                    val_ce += ce.item()
                    val_loss += loss.item()

                    del loss, mse, ce
                    torch.cuda.empty_cache()
                
                val_mse /= num_test_data
                val_ce /= num_test_data
                val_loss /= num_test_data
                val_rmse = val_mse ** (1/2)
                acc /= num_test_data

                if epoch==0:
                    best_rmse_val = val_rmse
                    best_acc = acc
                    results.append(best_rmse_val)
                    accuracies.append(best_acc)
                    best_epochs.append(epoch+1)
                    best_mse_val = val_mse
                    best_ce_val = val_ce
                    best_loss_val = val_loss
                    torch.save(model.state_dict(), os.path.join(save_path, f'best_model_{epoch+1}.pt'))
                else:
                    if val_rmse <= best_rmse_val:
                        best_rmse_val = val_rmse
                        best_acc = acc
                        results[-1] = best_rmse_val
                        accuracies[-1] = best_acc
                        best_epochs[-1] = epoch+1
                        best_mse_val = val_mse
                        best_ce_val = val_ce
                        best_loss_val = val_loss
                        torch.save(model.state_dict(), os.path.join(save_path, f'best_model_{epoch+1}.pt'))
                print(f'\t[Validation] LOSS: {val_loss} | MSE: {val_mse} | CE: {val_ce} | RMSE: {val_rmse} | ACC: {acc}\n\t[BEST] LOSS: {best_loss_val} | MSE: {best_mse_val} | CE: {best_ce_val} | RMSE: {best_rmse_val} | ACC: {best_acc}')
            f.write(f'{epoch+1},{train_loss},{train_mse},{train_ce},{train_rmse},{val_loss},{val_mse},{val_ce},{val_rmse},{acc}\n')
            del val_mse, val_ce, val_loss, val_rmse, train_mse, train_ce, train_loss, train_rmse, num_test_data, acc
            torch.cuda.empty_cache()
        f.close()   
        del optimizer, model, LOSS, trainloader, testloader, train_subsampler, test_subsampler
        torch.cuda.empty_cache()
        print(f"\nFold {fold+1}: {best_rmse_val}")
    print("\n[Results]")
    for fold, (result, accuracy) in enumerate(zip(results, accuracies)):
        print(f"Fold {fold+1}: {result:.4f}, {(accuracy*100):.3f} %")
    print(f"Mean RMSE of 5 folds CV: {np.mean(results):.4f}")
    print(f"Mean Accuracy of 5 folds CV: {(np.mean(accuracies)*100):.3f} %")

    if config["save_gspread"]:
        save_google_spread(config, results, accuracies, best_epochs)
        print("Completely save the results on google spread sheet!!")
    else:
        print("Don't save the results on google spread sheet.")
    
    if config["test_params"]:
        save_google_spread_test_params(config, results, best_epochs, config["gspread_offset"])
        print("[TEST PARAMS] Completely save the results on google spread sheet!!")

def train_mse(config):
    """
        Function to train a model which uses Mean Squared Error as a loss function by 5-fold cross validation.
    """
    # Basic Informations
    print("Train a model which uses only Mean Squared Error!")
    target = config["target"]
    basemean = config["basemean"]

    torch.cuda.empty_cache()

    # Results of each fold
    best_epochs = []
    results = []
    accuracies = []
    
    if config["folds"]==5:
        kfold = get_5fold(LABEL_PATH, file_length=config["file_length"], method=config["method"], subject=config["subject"])
    elif config["folds"]==10:
        kfold = get_10fold(LABEL_PATH, file_length=config["file_length"], method=config["method"], subject=config["subject"])
    SIGMOID = nn.Sigmoid()

    if config["feature_type"]=="EEG":
        input_length = int(7680 / (60//config["file_length"]))
        input_image = torch.zeros(1, input_length, 1, 9, 9)
        if config["basemean"]:
            dataset = DEAP(data_dir=W_BASEMEAN_DATA_PATH, label_path=LABEL_PATH, target=config["target"], file_length=config["file_length"], feature_type="EEG")
        else:
            dataset = DEAP(data_dir=WO_BASEMEAN_DATA_PATH, label_path=LABEL_PATH, target=config["target"], file_length=config["file_length"], feature_type="EEG")
    elif config["feature_type"]=="DE":
        input_length = int(60 / (60//config["file_length"]))
        input_image = torch.zeros(1, input_length, 1, 9, 9)
        if config["basemean"]:
            dataset = DEAP(data_dir=DE_W_BASEMEAN_DATA_PATH, label_path=LABEL_PATH, target=config["target"], file_length=config["file_length"], feature_type="DE")
        else:
            dataset = DEAP(data_dir=DE_WO_BASEMEAN_DATA_PATH, label_path=LABEL_PATH, target=config["target"], file_length=config["file_length"], feature_type="DE")

    # Devide folds
    for fold, (train_idx, test_idx) in enumerate(kfold):
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)

        trainloader = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=config["batch"], sampler=train_subsampler) 
        testloader = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=1, sampler=test_subsampler) 

        save_path = os.path.join(WEIGHTS_PATH, config["target"], config["method"], config["feature_type"], str(config["file_length"])+'s', config["output_dir"], f'fold_{fold+1}')
        create_directory(save_path)

        f = open(os.path.join(save_path, 'output.csv'), 'w')
        f.write('epoch,MSE,RMSE,v_MSE,v_RMSE\n')

        print(f'\nTraining for fold {fold+1}')

        LOSS = nn.MSELoss() # Mean Squared Error Loss
        model = EmoticonNet(device=config["device"], n_classes=config["n_classes"], input_image=input_image, formula=config["formula"])
        if config["optimizer"] == "SGD":
            optimizer = SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9)
        elif config["optimizer"] == "Adam":
            optimizer = Adam(model.parameters(), lr=config["learning_rate"], betas=config["betas"])
        elif config["optimizer"] == "AdamW":
            optimizer = AdamW(model.parameters(), lr=config["learning_rate"], betas=config["betas"], weight_decay=config["weight_decay"])

        for epoch in range(config["epochs"]):
            print(f"[INFO] Target:{target}||Loss:Mean Squared Error||BaseMean:{basemean}||Fold:{fold+1}")
            torch.cuda.empty_cache()

            model.train()

            optimizer.zero_grad()
            train_loss = 0
            num_data = 0
            optimizer_to(optimizer, config['device'])
            ## error
            for _, eeg, score, label_cls in tqdm(trainloader, desc=f"Epoch {epoch+1} / TRAIN", ncols=100, ascii=" =", leave=False):
                # Batch size
                num_batch = eeg.shape[0]
                num_data += num_batch

                true_points = torch.round(score).type("torch.FloatTensor").to(config["device"])
                pred_probs = model(eeg.to(config["device"]))
                pred_points = 8 * pred_probs + 1
                # true_class = label_cls.type("torch.LongTensor").to(config["device"])

                del score, eeg, label_cls
                torch.cuda.empty_cache()

                # Calculate Loss Function
                loss = LOSS(pred_points, true_points)

                del pred_probs, pred_points, true_points
                torch.cuda.empty_cache()

                train_loss += loss.item()*num_batch

                # Weights Update
                loss.backward(retain_graph = False)
                optimizer.step()
                optimizer.zero_grad()

                del loss, num_batch
                torch.cuda.empty_cache()

            optimizer_to(optimizer, 'cpu')
            
            train_loss /= num_data
            train_rmse = train_loss ** (1/2)

            print(f'[{epoch+1}/{config["epochs"]}] MSE: {train_loss}, RMSE: {train_rmse}')

            model.eval()
            with torch.no_grad():

                num_test_data = len(testloader)

                val_loss = 0
                num_data = 0

                acc = 0

                for _, eeg, score, label_cls in tqdm(testloader, desc=f"Epoch {epoch+1} / VALIDATE", ncols=100, ascii=" =", leave=False):


                    true_points = torch.round(score).type("torch.FloatTensor").to(config["device"])
                    pred_probs = model(eeg.to(config["device"]))
                    pred_points = 8 * pred_probs + 1
                    
                    pred_cls = 1 if pred_points >= 5 else 0
                    true_cls = 1 if true_points >= 5 else 0
                    if pred_cls==true_cls:
                        acc+=1

                    del score, eeg, label_cls
                    torch.cuda.empty_cache()

                    # Calculate Loss Function
                    loss = LOSS(pred_points, true_points)

                    del pred_probs, pred_points, true_points
                    torch.cuda.empty_cache()
                    
                    val_loss += loss.item()

                    del loss
                    torch.cuda.empty_cache()
                
                val_loss /= num_test_data
                val_rmse = val_loss ** (1/2)
                acc /= num_test_data

                if epoch==0:
                    best_rmse_val = val_rmse
                    results.append(best_rmse_val)
                    best_epochs.append(epoch+1)
                    accuracies.append(acc)
                    best_loss_val = val_loss
                    torch.save(model.state_dict(), os.path.join(save_path, f'best_model_{epoch+1}.pt'))
                else:
                    if val_rmse <= best_rmse_val:
                        best_rmse_val = val_rmse
                        results[-1] = best_rmse_val
                        best_epochs[-1] = epoch+1
                        accuracies[-1] = acc
                        best_loss_val = val_loss
                        torch.save(model.state_dict(), os.path.join(save_path, f'best_model_{epoch+1}.pt'))
                print(f'\t[Validation] LOSS: {val_loss} | RMSE: {val_rmse}\n\t[BEST] LOSS: {best_loss_val} | RMSE: {best_rmse_val}')
            f.write(f'{epoch+1},{train_loss},{train_rmse},{val_loss},{val_rmse}\n')
            del val_loss, val_rmse, train_loss, train_rmse
            torch.cuda.empty_cache()
        f.close()   
        del optimizer, model, LOSS, trainloader, testloader, train_subsampler, test_subsampler
        torch.cuda.empty_cache()
        print(f"\nFold {fold+1}: {best_rmse_val}")
    print(f"[Results]\nFold 1: {results[0]:.4f}\nFold 2: {results[1]:.4f}\nFold 3: {results[2]:.4f}\nFold 4: {results[3]:.4f}\nFold 5: {results[4]:.4f}")
    print(f"Mean RMSE of 5 folds CV: {np.mean(results):.4f}")

def test_cv(config):
    """
        Function to test a model which uses Expectation Formula or Bernoulli Formula by 5-fold cross validation.
    """
    # Basic Informations
    target = config["target"]
    formula = config["formula"]
    weight = config["weight"]
    alpha = config["alpha"]
    basemean = config["basemean"]

    torch.cuda.empty_cache()

    results = []
    accuracies = []
    weight_epochs = config["test_weights"].split(" ")
    
    if config["folds"]==5:
        kfold = get_5fold(LABEL_PATH, file_length=config["file_length"], method=config["method"], subject=config["subject"])
    elif config["folds"]==10:
        kfold = get_10fold(LABEL_PATH, file_length=config["file_length"], method=config["method"], subject=config["subject"])
    cal = Calculator(cal_type=config["formula"], n_classes=config["n_classes"])

    if config["feature_type"]=="EEG":
        input_length = int(7680 / (60//config["file_length"]))
        input_image = torch.zeros(1, input_length, 1, 9, 9)
        if config["basemean"]:
            dataset = DEAP(data_dir=W_BASEMEAN_DATA_PATH, label_path=LABEL_PATH, target=config["target"], file_length=config["file_length"], feature_type="EEG")
        else:
            dataset = DEAP(data_dir=WO_BASEMEAN_DATA_PATH, label_path=LABEL_PATH, target=config["target"], file_length=config["file_length"], feature_type="EEG")
    elif config["feature_type"]=="DE":
        input_length = int(60 / (60//config["file_length"]))
        input_image = torch.zeros(1, input_length, 1, 9, 9)
        if config["basemean"]:
            dataset = DEAP(data_dir=DE_W_BASEMEAN_DATA_PATH, label_path=LABEL_PATH, target=config["target"], file_length=config["file_length"], feature_type="DE")
        else:
            dataset = DEAP(data_dir=DE_WO_BASEMEAN_DATA_PATH, label_path=LABEL_PATH, target=config["target"], file_length=config["file_length"], feature_type="DE")

    # Devide folds
    for fold, (_, test_idx) in enumerate(kfold):
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx) 
        testloader = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=1, sampler=test_subsampler) 
        
        num_test_data = len(testloader)
        acc = 0

        print(f'\nTest for fold {fold+1}')

        LOSS = Custom_Loss(device=config["device"], alpha=config["alpha"], dist_type=config["weight"])
        model = EmoticonNet(device=config["device"], n_classes=config["n_classes"], input_image=input_image, formula=config["formula"])
        ep = weight_epochs[fold]
        if config["test_params"]:
            model.load_state_dict(torch.load(os.path.join(WEIGHTS_PATH, config["target"], config["method"], 'TEST2', str(config["file_length"])+'s', config["output_dir"], f'fold_{fold+1}', f'best_model_{ep}.pt')))
        else:
            model.load_state_dict(torch.load(os.path.join(WEIGHTS_PATH, config["target"], config["method"], config["feature_type"], str(config["file_length"])+'s', config["output_dir"], f'ALPHA_{alpha}', f'fold_{fold+1}', f'best_model_{ep}.pt')))

        print(f"[INFO] Target:{target}||Formula:{formula}||Loss:{weight}||Alpha:{alpha}||BaseMean:{basemean}||Fold:{fold+1}")
        torch.cuda.empty_cache()

        model.eval()
        with torch.no_grad():

            test_mse = 0
            test_ce = 0
            test_loss = 0
            num_data = 0

            for _, eeg, score, label_cls in tqdm(testloader, desc=f"TEST", ncols=100, ascii=" =", leave=False):

                true_points = torch.round(score).type("torch.FloatTensor").to(config["device"])
                pred_probs = model(eeg.to(config["device"]))
                pred_points = cal.calculate(pred_probs)
                true_class = label_cls.type("torch.LongTensor").to(config["device"])

                pred_cls = 1 if pred_points >= 5 else 0
                true_cls = 1 if true_points >= 5 else 0
                if pred_cls==true_cls:
                    acc+=1

                del score, eeg, label_cls
                torch.cuda.empty_cache()

                # Calculate Loss Function
                loss, mse, ce = LOSS(pred_probs, true_class, pred_points, true_points)
                del pred_probs, true_class, pred_points, true_points
                torch.cuda.empty_cache()
                
                test_mse += mse.item()
                test_ce += ce.item()
                test_loss += loss.item()

                del loss, mse, ce
                torch.cuda.empty_cache()
            
            test_mse /= num_test_data
            test_ce /= num_test_data
            test_loss /= num_test_data
            test_rmse = test_mse ** (1/2)
            acc /= num_test_data

            results.append(test_rmse)
            accuracies.append(acc)

            print(f'\t[Test] LOSS: {test_loss} | MSE: {test_mse} | CE: {test_ce} | RMSE: {test_rmse} | Accuracy: {acc}')

        del test_mse, test_ce, test_loss, test_rmse
        torch.cuda.empty_cache()
    
    print("\n[Results]")
    for fold, (result, accuracy) in enumerate(zip(results, accuracies)):
        print(f"Fold {fold+1}: {result:.4f}, {(accuracy*100):.3f} %")
    print(f"Mean RMSE of 5 folds CV: {np.mean(results):.4f}")
    print(f"Mean Accuracy of 5 folds CV: {(np.mean(accuracies)*100):.3f} %")

def test_mse(config):
    """
        Function to test a model which uses Mean Squared Error as a loss function by 5-fold cross validation.
    """
    # Basic Informations
    target = config["target"]
    basemean = config["basemean"]

    torch.cuda.empty_cache()

    results = []
    accuracies = []
    weight_epochs = config["test_weight"].split(" ")
    
    if config["folds"]==5:
        kfold = get_5fold(LABEL_PATH, file_length=config["file_length"], method=config["method"], subject=config["subject"])
    elif config["folds"]==10:
        kfold = get_10fold(LABEL_PATH, file_length=config["file_length"], method=config["method"], subject=config["subject"])
    SIGMOID = nn.Sigmoid()

    if config["feature_type"]=="EEG":
        input_length = int(7680 / (60//config["file_length"]))
        input_image = torch.zeros(1, input_length, 1, 9, 9)
        if config["basemean"]:
            dataset = DEAP(data_dir=W_BASEMEAN_DATA_PATH, label_path=LABEL_PATH, target=config["target"], file_length=config["file_length"], feature_type="EEG")
        else:
            dataset = DEAP(data_dir=WO_BASEMEAN_DATA_PATH, label_path=LABEL_PATH, target=config["target"], file_length=config["file_length"], feature_type="EEG")
    elif config["feature_type"]=="DE":
        input_length = int(60 / (60//config["file_length"]))
        input_image = torch.zeros(1, input_length, 1, 9, 9)
        if config["basemean"]:
            dataset = DEAP(data_dir=DE_W_BASEMEAN_DATA_PATH, label_path=LABEL_PATH, target=config["target"], file_length=config["file_length"], feature_type="DE")
        else:
            dataset = DEAP(data_dir=DE_WO_BASEMEAN_DATA_PATH, label_path=LABEL_PATH, target=config["target"], file_length=config["file_length"], feature_type="DE")

    # Devide folds
    for fold, (_, test_idx) in enumerate(kfold):
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)
        testloader = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=1, sampler=test_subsampler) 

        num_test_data = len(testloader)
        acc = 0

        save_path = os.path.join(WEIGHTS_PATH, config["target"], config["method"], config["feature_type"], str(config["file_length"])+'s', config["output_dir"], f'fold_{fold+1}')
        create_directory(save_path)

        f = open(os.path.join(save_path, 'output.csv'), 'w')
        f.write('epoch,MSE,RMSE,v_MSE,v_RMSE\n')

        print(f'\nTraining for fold {fold+1}')

        LOSS = nn.MSELoss() # Mean Squared Error Loss
        model = EmoticonNet(device=config["device"], n_classes=config["n_classes"], input_image=input_image, formula=config["formula"])
        ep = weight_epochs[fold]
        model.load_state_dict(torch.load(os.path.join(WEIGHTS_PATH, config["output_dir"], f'fold_{fold+1}', f'best_model_{ep}.pt')))

        print(f"[INFO] Target:{target}||Loss:Mean Squared Error||BaseMean:{basemean}||Fold:{fold+1}")
        torch.cuda.empty_cache()

        model.eval()
        with torch.no_grad():

            test_loss = 0

            for _, eeg, score, label_cls in tqdm(testloader, desc=f"Epoch {epoch+1} / VALIDATE", ncols=100, ascii=" =", leave=False):

                true_points = torch.round(score).type("torch.FloatTensor").to(config["device"])
                pred_probs = model(eeg.to(config["device"]))
                pred_points = 8 * pred_probs + 1

                pred_cls = 1 if pred_points >= 5 else 0
                true_cls = 1 if true_points >= 5 else 0
                if pred_cls==true_cls:
                    acc+=1

                del score, eeg, label_cls
                torch.cuda.empty_cache()

                # Calculate Loss Function
                loss = LOSS(pred_points, true_points)

                del pred_probs, pred_points, true_points
                torch.cuda.empty_cache()
                
                test_loss += loss.item()

                del loss, num_batch
                torch.cuda.empty_cache()
            
            test_loss /= num_test_data
            test_rmse = test_loss ** (1/2)
            acc /= num_test_data

            results.append(test_rmse)
            accuracies.append(acc)

            print(f'\t[Test] LOSS: {test_loss} | RMSE: {test_rmse} | Accuracy: {acc}')

        del test_loss, test_rmse
        torch.cuda.empty_cache()

    print("\n[Results]")
    for fold, (result, accuracy) in enumerate(zip(results, accuracies)):
        print(f"Fold {fold+1}: {result:.4f}, {(accuracy*100):.3f} %")
    print(f"Mean RMSE of 5 folds CV: {np.mean(results):.4f}")
    print(f"Mean Accuracy of 5 folds CV: {(np.mean(accuracies)*100):.3f} %")

if __name__=="__main__":

    # Basic & Model Options
    parser  = argparse.ArgumentParser(description="Alzheimer's Disease Detection")
    parser.add_argument('--test_params', type=str2bool, default=False, help='Select whether to test hyperparameters')
    parser.add_argument('--gspread_offset', type=int, default=None, help='Select offset to save the results on google spread sheet')

    parser.add_argument('--mode', type=str, default='Train', choices=['Train', 'Test'], help='Select the Mode')
    parser.add_argument('--method', type=str, default='CSI', choices=['SD', 'CSV', 'CSI', 'US'], help='Select tahe method for testing')
    parser.add_argument('--subject', type=int, default=None, help='Select a subject to train or test the model(Subject-Independent Model)')
    parser.add_argument('--folds', type=int, default=5, help='Select the number of folds')

    parser.add_argument('--epochs', type=int, default=1000, help='Set the number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Set the batch size')
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam', 'AdamW'], help='Select the Optimizer')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Set the Learning Rate')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu','cuda:0','cuda:1','cuda:2','cuda:3'], help='Select GPU or CPU')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Set the Weight decay')
    parser.add_argument('--beta1', type=float, default=0.9, help='Set the beta1')
    parser.add_argument('--beta2', type=float, default=0.999, help='Set the beta2')

    parser.add_argument('--formula', type=str, default='Expectation', choices=['None', 'Expectation', 'Bernoulli'], help='Select the Calculator Type')
    parser.add_argument('--weight', type=str, default='None', choices=['None', 'Absolute', 'Square'], help='Select the Weight Type')
    parser.add_argument('--alpha', type=int, default=1, help='Select the weight of CE')
    parser.add_argument('--n_classes', type=int, default=9, choices=[9, 5, 3, 1], help='Select the number of intervals')

    # Data Options
    parser.add_argument('--target', type=str, default='valence', choices=['valence', 'arousal'], help='Select the Target Type')
    parser.add_argument('--dataset', type=str, default='DEAP', choices=['DEAP'], help='Select a Dataset')
    parser.add_argument('--feature', type=str, default='EEG', choices=['EEG', 'DE'], help='Select a Feature Type')
    parser.add_argument('--basemean', type=str2bool, default=False, help='Select whether to use Basemean')
    parser.add_argument('--file_length', type=int, default=10, help='Select the Signal Length')

    # Save Options
    parser.add_argument('--output_dir', type=str, default='TEST', help='Select a Dataset')
    parser.add_argument('--save_gspread', type=str2bool, default=False, help='Select whether to save the results on google spread sheet')
    parser.add_argument('--test_weights', type=str, help='Set the weights of the model for Test')

    args = parser.parse_args()

    config = {
        # Basic Options
        "test_params": args.test_params,
        "gspread_offset": args.gspread_offset,

        "mode": args.mode,
        "method": args.method,
        "subject": args.subject,
        "folds": args.folds,

        "device": args.device,
        "epochs": args.epochs,
        "batch": args.batch_size,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "betas": (args.beta1, args.beta2),
        "weight_decay": args.weight_decay,

        "formula": args.formula,
        "weight": args.weight,
        "alpha": args.alpha,
        "n_classes": args.n_classes,

        # dataset
        "dataset": args.dataset,
        "target": args.target,
        "basemean": args.basemean,
        "file_length": args.file_length,
        "feature_type": args.feature,

        # Test Model
        "test_weights": args.test_weights,
        "output_dir": args.output_dir,
        "save_gspread":args.save_gspread, 

        # Model Options
        "input_image": INPUT_IMAGE, 
        "kernel": KERNEL, 
        "stride": STRIDE, 
        "padding": PADDING, 
        "n_units": N_UNITS
    }

    print("\n------------ Options ------------")
    print("--Test Parameters:", args.test_params)
    print("--Mode:", args.mode)
    print("--Method:", args.method)
    print("--Subject:", args.subject)
    print("--#. of folds:", args.folds)
    print("--Target:", args.target)
    print("--Epochs:", args.epochs)
    print("--Batch Size:", args.batch_size)
    print("--Optimizer:", args.optimizer)
    print("--Learning Rate:", args.learning_rate)
    print("--GPU/CPU:", args.device)
    print("--# of Intervals:", args.n_classes)
    print("--Dataset:", args.dataset)
    print("--Feature:", args.feature)
    print("--File Length:", args.file_length)
    print("--BaseMean:", args.basemean)
    print("--Formula:", args.formula)
    print("--Weight Type:", args.weight)
    print("--Weight Size:", args.alpha)
    print("--Output Directory:", args.output_dir)
    print("--Save Google Spread Sheet:", args.save_gspread)
    print("----------------------------------")

    if args.mode == "Train":
        if args.formula == "None":
            train_mse(config)
        elif args.formula in ['Expectation', 'Bernoulli']:
            train_cv(config)

        print("\n------------ Options ------------")
        print("--Test Parameters:", args.test_params)
        print("--Mode:", args.mode)
        print("--Method:", args.method)
        print("--Subject:", args.subject)
        print("--#. of folds:", args.folds)
        print("--Target:", args.target)
        print("--Epochs:", args.epochs)
        print("--Batch Size:", args.batch_size)
        print("--Optimizer:", args.optimizer)
        print("--Learning Rate:", args.learning_rate)
        print("--GPU/CPU:", args.device)
        print("--# of Intervals:", args.n_classes)
        print("--Dataset:", args.dataset)
        print("--Feature:", args.feature)
        print("--File Length:", args.file_length)
        print("--BaseMean:", args.basemean)
        print("--Formula:", args.formula)
        print("--Weight Type:", args.weight)
        print("--Weight Size:", args.alpha)
        print("--Output Directory:", args.output_dir)
        print("--Save Google Spread Sheet:", args.save_gspread)
        print("----------------------------------")

    elif args.mode == "Test":
        if args.formula == "None":
            test_mse(config)
        elif args.formula in ['Expectation', 'Bernoulli']:
            test_cv(config)
    else:
        print("Mode Error...!!")