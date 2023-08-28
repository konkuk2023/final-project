import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import Dataset

from options import INPUT_IMAGE, KERNEL, STRIDE, PADDING, N_CLASSES, N_UNITS, MOMENTUM, \
                    W_BASEMEAN_DATA_PATH, WO_BASEMEAN_DATA_PATH, DE_W_BASEMEAN_DATA_PATH, DE_WO_BASEMEAN_DATA_PATH, \
                    LABEL_PATH, WEIGHTS_PATH
from models import EmoticonNet
from dataloader import get_5fold, DEAP
from loss import Custom_Loss, Custom_Loss2

import os
import argparse
from tqdm import tqdm
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
    
def optimizer_to(optim, device):
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
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def create_directory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

def train_cv(config):
    # Basic Informations
    target = config["target"]
    formula = config["formula"]
    weight = config["weight"]
    alpha = config["alpha"]
    basemean = config["basemean"]

    torch.cuda.empty_cache()

    results = [0, 0, 0, 0, 0]
    
    kfold = get_5fold(LABEL_PATH)
    cal = Calculator(cal_type=config["formula"])

    if config["basemean"]:
        dataset = DEAP(data_dir=W_BASEMEAN_DATA_PATH, label_path=LABEL_PATH, target=config["target"])
    else:
        dataset = DEAP(data_dir=WO_BASEMEAN_DATA_PATH, label_path=LABEL_PATH, target=config["target"])

    # Devide folds
    for fold, (train_idx, test_idx) in enumerate(kfold):
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)

        trainloader = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=config["batch"], sampler=train_subsampler) 
        testloader = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=config["batch"], sampler=test_subsampler) 

        save_path = os.path.join(WEIGHTS_PATH, config["output_dir"], f'ALPHA_{alpha}', f'fold_{fold+1}')
        create_directory(save_path)

        f = open(os.path.join(save_path, 'output.csv'), 'w')
        f.write('epoch,loss,MSE,CE,RMSE,v_loss,v_MSE,v_CE,v_RMSE\n')

        print(f'\nTraining for fold {fold+1}')

        LOSS = Custom_Loss(device=config["device"], alpha=config["alpha"], dist_type=config["weight"])

        model = EmoticonNet(device=config["device"])
        if config["optimizer"] == "SGD":
            optimizer = SGD(model.parameters(), lr=1e-4, momentum=0.9)
        elif config["optimizer"] == "Adam":
            optimizer = Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.9))
            # optimizer = AdamW(model.parameters(), lr=1e-4)

        for epoch in range(config["epochs"]):
            print(f"[INFO] Target:{target}||Formula:{formula}||Loss:{weight}||Alpha:{alpha}||BaseMean:{basemean}||Fold:{fold+1}")
            torch.cuda.empty_cache()

            model.train()

            optimizer.zero_grad()
            train_mse = 0
            train_ce = 0
            train_loss = 0
            num_data = 0
            # print("MEMORY USAGE 1:", round(torch.cuda.memory_allocated(1)/1024**3, 2), "GB")
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

                val_mse = 0
                val_ce = 0
                val_loss = 0
                num_data = 0

                for _, eeg, score, label_cls in tqdm(testloader, desc=f"Epoch {epoch+1} / VALIDATE", ncols=100, ascii=" =", leave=False):

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
                    
                    val_mse += mse.item()*num_batch
                    val_ce += ce.item()*num_batch
                    val_loss += loss.item()*num_batch

                    del loss, mse, ce, num_batch
                    torch.cuda.empty_cache()
                
                val_mse /= num_data
                val_ce /= num_data
                val_loss /= num_data
                val_rmse = val_mse ** (1/2)

                if epoch==0:
                    best_rmse_val = val_rmse
                    results[fold] = best_rmse_val
                    best_mse_val = val_mse
                    best_ce_val = val_ce
                    best_loss_val = val_loss
                else:
                    if val_rmse <= best_rmse_val:
                        best_rmse_val = val_rmse
                        results[fold] = best_rmse_val
                        best_mse_val = val_mse
                        best_ce_val = val_ce
                        best_loss_val = val_loss
                        torch.save(model.state_dict(), os.path.join(save_path, f'best_model_{epoch+1}.pt'))
                print(f'\t[Validation] LOSS: {val_loss} | MSE: {val_mse} | CE: {val_ce} | RMSE: {val_rmse}\n\t[BEST] LOSS: {best_loss_val} | MSE: {best_mse_val} | CE: {best_ce_val} | RMSE: {best_rmse_val}')
            f.write(f'{epoch+1},{train_loss},{train_mse},{train_ce},{val_loss},{val_mse},{val_ce},{val_rmse}\n')
            del val_mse, val_ce, val_loss, val_rmse, train_mse, train_ce, train_loss, train_rmse
            torch.cuda.empty_cache()
        f.close()   
        del optimizer, model, LOSS, trainloader, testloader, train_subsampler, test_subsampler
        torch.cuda.empty_cache()
        print(f"\nFold {fold+1}: {best_rmse_val}")
    print(f"[Results]\nFold 1: {results[0]}\nFold 2: {results[1]}\nFold 3: {results[2]}\nFold 4: {results[3]}\nFold 5: {results[4]}")
    print(f"Mean RMSE of 5 folds CV: {np.mean(results)}")

def test_cv(config):
    # Basic Informations
    target = config["target"]
    formula = config["formula"]
    weight = config["weight"]
    alpha = config["alpha"]
    basemean = config["basemean"]

    torch.cuda.empty_cache()

    results = [0, 0, 0, 0, 0]
    weight_epochs = config["test_weight"].split(" ")
    
    kfold = get_5fold(LABEL_PATH)
    cal = Calculator(cal_type=config["formula"])

    if config["basemean"]:
        dataset = DEAP(data_dir=W_BASEMEAN_DATA_PATH, label_path=LABEL_PATH, target=config["target"])
    else:
        dataset = DEAP(data_dir=WO_BASEMEAN_DATA_PATH, label_path=LABEL_PATH, target=config["target"])

    # Devide folds
    for fold, (_, test_idx) in enumerate(kfold):
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx) 
        testloader = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=config["batch"], sampler=test_subsampler) 

        print(f'\nTraining for fold {fold+1}')

        LOSS = Custom_Loss(device=config["device"], alpha=config["alpha"], dist_type=config["weight"])
        model = EmoticonNet(device=config["device"])
        ep = weight_epochs[fold]
        model.load_state_dict(torch.load(os.path.join(WEIGHTS_PATH, config["output_dir"], f'ALPHA_{alpha}', f'fold_{fold+1}', f'best_model_{ep}.pt')))

        for epoch in range(config["epochs"]):
            print(f"[INFO] Target:{target}||Formula:{formula}||Loss:{weight}||Alpha:{alpha}||BaseMean:{basemean}||Fold:{fold+1}")
            torch.cuda.empty_cache()

            model.eval()
            with torch.no_grad():

                test_mse = 0
                test_ce = 0
                test_loss = 0
                num_data = 0

                for _, eeg, score, label_cls in tqdm(testloader, desc=f"Epoch {epoch+1} / VALIDATE", ncols=100, ascii=" =", leave=False):

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
                    
                    test_mse += mse.item()*num_batch
                    test_ce += ce.item()*num_batch
                    test_loss += loss.item()*num_batch

                    del loss, mse, ce, num_batch
                    torch.cuda.empty_cache()
                
                test_mse /= num_data
                test_ce /= num_data
                test_loss /= num_data
                test_rmse = test_mse ** (1/2)

                results[fold] = test_rmse.detach().cpu().item()

                print(f'\t[Test] LOSS: {test_loss} | MSE: {test_mse} | CE: {test_ce} | RMSE: {test_rmse}')
            del test_mse, test_ce, test_loss, test_rmse
            torch.cuda.empty_cache()

        print(f"\nFold {fold+1}: {test_rmse}")
    print(f"[Results]\nFold 1: {results[0]}\nFold 2: {results[1]}\nFold 3: {results[2]}\nFold 4: {results[3]}\nFold 5: {results[4]}")
    print(f"Mean RMSE of 5 folds CV: {np.mean(results)}")

if __name__=="__main__":

    # python ./main.py --mode Train --dataset DEAP --epochs 500 --batch_size 8 --optimizer SGD --target valence --basemean True --alpha 1 --formula Expectation --weight None --device cuda:1 --output_dir TEST

    # Basic & Model Options
    parser  = argparse.ArgumentParser(description="Alzheimer's Disease Detection")
    parser.add_argument('--mode', type=str, default='Train', choices=['Train', 'Test'], help='Select the Mode')
    parser.add_argument('--epochs', type=int, default=1000, help='Set the number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Set the batch size')
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'], help='Select the Optimizer')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Set the Learning Rate')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu','cuda:0','cuda:1','cuda:2','cuda:3'], help='Select GPU or CPU')

    parser.add_argument('--formula', type=str, default='Expectation', choices=['Expectation', 'Bernoulli'], help='Select the Calculator Type')
    parser.add_argument('--weight', type=str, default='None', choices=['None', 'Absolute', 'Square'], help='Select the Weight Type')
    parser.add_argument('--alpha', type=int, default=1, help='Select the weight of CE')

    # Data Options
    parser.add_argument('--target', type=str, default='valence', choices=['valence', 'arousal'], help='Select the Target Type')
    parser.add_argument('--dataset', type=str, default='DEAP', choices=['DEAP'], help='Select a Dataset')
    parser.add_argument('--basemean', type=str2bool, default=False, help='Select whether to use Basemean')

    # Save Options
    parser.add_argument('--output_dir', type=str, default='TEST', help='Select a Dataset')
    parser.add_argument('--test_weights', type=str, help='Set the weights of the model for Test')

    args = parser.parse_args()

    config = {
        # Basic Options
        "mode": args.mode,
        "device": args.device,
        "epochs": args.epochs,
        "batch": args.batch_size,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "momentum": MOMENTUM,

        "formula": args.formula,
        "weight": args.weight,
        "alpha": args.alpha,

        # dataset
        "dataset": args.dataset,
        "target": args.target,
        "basemean": args.basemean,

        # Test Model
        "test_weights": args.test_weights,
        "output_dir": args.output_dir,

        # Model Options
        "input_image": INPUT_IMAGE, 
        "kernel": KERNEL, 
        "stride": STRIDE, 
        "padding": PADDING, 
        "n_classes": N_CLASSES, 
        "n_units": N_UNITS
    }

    print("\n------------ Options ------------")
    print("--Mode:", args.mode)
    print("--Target:", args.target)
    print("--Epochs:", args.epochs)
    print("--Batch Size:", args.batch_size)
    print("--Optimizer:", args.optimizer)
    print("--Learning Rate:", args.learning_rate)
    print("--GPU/CPU:", args.device)
    print("--Dataset:", args.dataset)
    print("--BaseMean:", args.basemean)
    print("--Formula:", args.formula)
    print("--Weight Type:", args.weight)
    print("--Weight Size:", args.alpha)
    print("--Output Directory:", args.output_dir)
    print("----------------------------------")

    if args.mode == "Train":
        train_cv(config)

        print("\n------------ Options ------------")
        print("--Mode:", args.mode)
        print("--Target:", args.target)
        print("--Epochs:", args.epochs)
        print("--Batch Size:", args.batch_size)
        print("--Optimizer:", args.optimizer)
        print("--Learning Rate:", args.learning_rate)
        print("--GPU/CPU:", args.device)
        print("--Dataset:", args.dataset)
        print("--BaseMean:", args.basemean)
        print("--Formula:", args.formula)
        print("--Weight Type:", args.weight)
        print("--Weight Size:", args.alpha)
        print("--Output Directory:", args.output_dir)
        print("----------------------------------")

    elif args.mode == "Test":
        test_cv(config)
    else:
        print("Mode Error...!!")