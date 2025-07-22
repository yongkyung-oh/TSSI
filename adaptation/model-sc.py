#-*- coding:utf-8 -*-

import os
import sys
import copy
import time
import datetime
import pickle
import random

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms

import spconv.pytorch as spconv
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from sktime.datasets import load_UCR_UEA_dataset
from scipy.interpolate import interp1d

sys.path.append('..')
from ts2img import *

# Set random seed
SEED = 12345
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# get different data
data_info = pd.read_csv('../dataset_summary_multivariate.csv', index_col=0)
dataname_list = data_info['problem'].tolist()
dataname_list = np.unique(dataname_list)


if not os.path.exists('out_mc-sc'):
    os.mkdir('out_mc-sc')
    
for dataname in dataname_list:
    if not os.path.exists(os.path.join('out_mc-sc', dataname)):
        os.mkdir(os.path.join('out_mc-sc', dataname))
        
try:
    model_name = str(sys.argv[1])
except:
    model_name = 'Resnet18'
transfer = True
sparse = False
    
dataname_todo = []
for dataname in dataname_list:
    for k in range(10):
        try:
            out_name = '{}_{}_{}_{}'.format(model_name, str(transfer), str(sparse), k)
            with open(os.path.join('out_mc-sc', dataname, out_name), 'rb') as f:
                out = pickle.load(f)
        except:
            dataname_todo.append(dataname)

dataname_todo = np.unique(dataname_todo)
np.random.shuffle(dataname_todo)
print(dataname_todo)

### Run experiments 
for dataname in dataname_todo:
    try:
        try:
            X_train, y_train = load_UCR_UEA_dataset(dataname, 'train', return_X_y=True)
            X_test, y_test = load_UCR_UEA_dataset(dataname, 'test', return_X_y=True)
        except:
            X_train, y_train = load_from_tsfile_to_dataframe(os.path.join('../data/Multivariate_ts', "{}/{}_TRAIN.ts".format(dataname,dataname)))
            X_test, y_test = load_from_tsfile_to_dataframe(os.path.join('../data/Multivariate_ts', "{}/{}_TEST.ts".format(dataname,dataname)))

        label_dict = dict(zip(np.unique(y_train), range(len(np.unique(y_train)))))
        y_train = np.array([label_dict[y] for y in y_train])
        y_test = np.array([label_dict[y] for y in y_test])

        num_dim = X_train.shape[1]
        num_class = len(np.unique(y_train))

    except:
        print(dataname)
        continue
        
    # interpolate to resize
    try:
        len_list = []
        for i in range(X_train.shape[0]):
            for j in range(X_train.shape[1]):
                len_list.append(len(X_train.iloc[i,j]))
        max_len = max(len_list)

        if min(len_list) == max(len_list):
            pass
        else:
            max_len = max(len_list)

            def get_same_len(s):
                s_re = np.interp(
                       np.arange(0,max_len),
                       np.linspace(0,max_len,num=len(s)),
                       s)
                return pd.Series(s_re)

            X_train = X_train.applymap(lambda s: get_same_len(s))
            X_test = X_test.applymap(lambda s: get_same_len(s))
        
        num_dim = X_train.shape[1]
        num_class = len(np.unique(y_train))

    except:
        print(dataname)
        continue

    print(dataname, len(X_train), len(X_test), num_dim, num_class, max(len_list))
        
    X_train, X_test = normalize(X_train, X_test)
    X_train_np = get_np_format(X_train)
    X_test_np = get_np_format(X_test)
    
    # set size
    size = 2**8 # default 256 * 256
    tensor_size = 224
    if num_dim > 100:
        tensor_size = 112 # compressed tensor for high-dimensional

    # get scale 
    min_val, max_val = np.min(X_train_np), np.max(X_train_np)
    scale = (min_val, max_val)

    # transform
    kwargs = {
        'size': size,
        'image_size': 256 if max_len > 256 else max_len,
        'scale': (min_val, max_val), # TSSI
        'sample_range': (-1,1), # GADF, GASF
        'n_bins': 8 if max_len > 16 else 3, # MTF
        'time_delay': 1 if max_len < 1e3 else 10, # RP
    }
    
    def transform_function(sample, kwargs=kwargs):
        return transform_SC(sample, **kwargs)

    image_transform = transforms.Compose([
            transforms.Resize((tensor_size, tensor_size)), # for CNN model 
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
            ])
    BATCH_SIZE = 16 if len(X_train) > 16 else 4
    
    for k in range(10):
        # Load and setup model
        print(sys.argv[0], model_name, dataname)
        # check exists
        if os.path.exists(os.path.join('out_mc-sc', dataname, '{}_{}_{}_{}'.format(model_name, str(transfer), str(sparse), k))):
            continue
        
        # Split train / valid / test
        train_idx, valid_idx = train_test_split(range(len(y_train)), train_size=0.8, shuffle=True, stratify=y_train)

        train_dataset = MTSC_Dataset(X_train_np[train_idx], y_train[train_idx], image_transform, transform_function)
        valid_dataset = MTSC_Dataset(X_train_np[valid_idx], y_train[valid_idx], image_transform, transform_function)
        test_dataset = MTSC_Dataset(X_test_np, y_test, image_transform, transform_function)
        
        train_batch = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=16)
        valid_batch = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=16)
        test_batch = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=16)

        # Set model
        num_input_channel = num_dim
        num_output_class = num_class
        dropout = 0.1

        model = TransferModel(model_name, transfer, num_input_channel, num_output_class, dropout, sparse, size=tensor_size)
        model.to(device)

        # Train and Evaluate Model
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
        criterion = nn.CrossEntropyLoss()

        # lr scheduler 
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        best_loss = np.infty
        best_model_wts = copy.deepcopy(model.state_dict())
        patient = 0

        for e in tqdm(range(0, 100)):
            train_loss, train_accuracy = train(model, optimizer, criterion, train_batch, device)
            valid_loss, valid_accuracy, _ ,_  = evaluate(model, criterion, valid_batch, device)
            test_loss, test_accuracy, _ ,_  = evaluate(model, criterion, test_batch, device)

            if e % 10 == 0:
                print('{:00d}|{:4.4f}|{:4.4f}|{:4.4f}|{:2.2f}|{:2.2f}|{:2.2f}'.format(e, train_loss, valid_loss, test_loss, train_accuracy, valid_accuracy, test_accuracy))

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                patient = 0
            else:
                patient += 1

            if (e > 20) & (patient > 10):
                break

            scheduler.step()

        # using trained model
        model.load_state_dict(best_model_wts)

        # pred
        model.eval()
        corrects, total_loss = 0, 0
        true_list, pred_list, proba_list = [], [], []
        with torch.no_grad():
            for batch in test_batch:
                tensor = batch['tensor'].to(device)
                y = batch['label'].type(torch.LongTensor).to(device)

                outputs = model(tensor)
                _, preds = torch.max(outputs, 1)

                true_list.append(y.tolist())
                pred_list.append(preds.tolist())
                proba_list.append(outputs.tolist())

        y_test = [x for y in true_list for x in y]
        y_pred = [x for y in pred_list for x in y]
        y_proba = [x for y in proba_list for x in y]

        # metrics
        train_loss, train_accuracy, _ ,_ = evaluate(model, criterion, train_batch, device)
        valid_loss, valid_accuracy, _ ,_  = evaluate(model, criterion, valid_batch, device)
        test_loss, test_accuracy, _ ,_  = evaluate(model, criterion, test_batch, device)

        metric = [
            train_loss, train_accuracy,
            valid_loss, valid_accuracy, 
            test_loss, test_accuracy,
        ]


        with open(os.path.join('out_mc-sc', dataname, '{}_{}_{}_{}'.format(model_name, str(transfer), str(sparse), k)), 'wb') as f:
            pickle.dump([y_test, y_pred, y_proba, metric], f)

    
     
    
    
    
    

